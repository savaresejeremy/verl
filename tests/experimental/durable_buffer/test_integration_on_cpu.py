# Copyright 2026 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration: verl's real ``ReplayBufferAsync`` over the durable buffer.

Two properties are asserted here, and they are the whole point of the package:

1. After a crash, the buffer can be *sampled and trained on* with no rollout workers
   alive — the finished-but-untrained trajectories are not regenerated.
2. Journaling the dataloader state removes the duplicate dispatch that a
   trainer-checkpoint-only restore causes (the loader would re-serve prompts whose
   trajectories are already in the buffer).
"""

from __future__ import annotations

import time

import torch
from helpers import crash_and_restart, finish_group, submit_prompt, tag_prompt_finished
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.experimental.durable_buffer import recover
from verl.experimental.durable_buffer.journal import JournalWriter, latest_journal
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync


def _buffer(poll_interval=0.05, threshold=64):
    return ReplayBufferAsync(
        trainer_mode="separate_async",
        trainer_config=OmegaConf.create({}),
        max_off_policy_threshold=threshold,
        max_off_policy_strategy="drop",
        sampler_kwargs=OmegaConf.create({}),
        poll_interval=poll_interval,
        refill_fn=None,
    )


def test_replay_buffer_trains_on_crash_survivors(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    for i, uid in enumerate(("s1", "s2", "s3")):
        submit_prompt(tq, partition_id, uid, status="running", global_steps=1)
        finish_group(tq, partition_id, uid, n=2, value=i + 1, global_steps=1)
        tag_prompt_finished(tq, partition_id, uid, global_steps=1)
    journal = tmp_path / "journal"
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1, "fence_ts": time.time()})

    tq = crash_and_restart(tq_conf)
    recover(tq, store_root, journal, partition_id, group_size=2)

    rb = _buffer()
    t0 = time.perf_counter()
    batch, _metrics = rb.sample(global_steps=2, partition_id=partition_id, batch_size=3)
    sample_s = time.perf_counter() - t0

    groups = sorted({k.split("_", 1)[0] for k in batch.keys})
    assert groups == ["s1", "s2", "s3"], f"expected all crash survivors, got {groups}"
    assert sample_s < 5.0, "sampling should be immediate: no generation is needed"
    data = tq.kv_batch_get(keys=batch.keys, partition_id=partition_id, select_fields=["response_ids"])
    resp = data["response_ids"]
    rows = resp.unbind(0) if getattr(resp, "is_nested", False) else list(resp)
    assert all(int(r.reshape(-1)[0]) in (1, 2, 3) for r in rows), "trajectory bytes are wrong after recovery"


def test_journal_writer_ring_keeps_a_good_snapshot(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    finish_group(tq, partition_id, "ring", n=2, value=1)
    root = tmp_path / "journals"
    jw = JournalWriter(root, interval_s=0.0, ring=3)
    for _ in range(5):
        info = jw.tick()
        assert info is not None
    dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert dirs == ["journal-000", "journal-001", "journal-002"], dirs
    newest = latest_journal(root)
    assert newest is not None and (newest / "controller_state.pkl").exists()
    assert jw.summary()["journal_ticks"] == 5

    # a torn snapshot (no stamp) must be ignored by latest_journal
    torn = root / "journal-torn"
    torn.mkdir()
    (torn / "controller_state.pkl").write_bytes(b"garbage")
    assert latest_journal(root) == newest


class _Items(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"item_id": i}


def test_dataloader_state_journaling_removes_duplicate_dispatch(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    """v1 (loader state from the trainer checkpoint) duplicates work; journaling it does not."""
    tq = tq_session
    n_items, batch, group = 32, 4, 2
    ckpt_state = tmp_path / "data.pt"
    journal = tmp_path / "journal"

    def loader():
        return StatefulDataLoader(_Items(n_items), batch_size=batch, shuffle=False, num_workers=0)

    dl = loader()
    it = iter(dl)
    dispatched: list[int] = []
    for step in (1, 2, 3, 4):
        items = [int(x) for x in next(it)["item_id"]]
        dispatched.extend(items)
        for k, item in enumerate(items):
            uid = f"s{step}i{item}"
            submit_prompt(tq, partition_id, uid, status="running", global_steps=step)
            if k < len(items) - 1:  # last prompt of each step stays in flight
                finish_group(tq, partition_id, uid, n=group, value=item, global_steps=step)
                tag_prompt_finished(tq, partition_id, uid, global_steps=step)
        if step == 2:
            torch.save(dl.state_dict(), ckpt_state)  # trainer checkpoint
        if step == 4:
            time.sleep(0.05)
            tq.save_checkpoint(str(journal), metadata={"global_steps": step, "fence_ts": time.time()})
            torch.save(dl.state_dict(), journal / "dataloader_state.pt")  # the v1.1 addition

    tq = crash_and_restart(tq_conf)
    recover(tq, store_root, journal, partition_id, group_size=group)
    live = tq.kv_list().get(partition_id, {})
    recoverable = {int(k.split("i")[1].split("_")[0]) for k in live if "i" in k}

    def will_serve(state_path):
        d = loader()
        d.load_state_dict(torch.load(state_path, weights_only=False))
        out = []
        for b in d:
            out.extend(int(x) for x in b["item_id"])
        return set(out)

    dup_v1 = recoverable & will_serve(ckpt_state)
    dup_v11 = recoverable & will_serve(journal / "dataloader_state.pt")
    lost_v11 = set(dispatched) - recoverable - will_serve(journal / "dataloader_state.pt")

    assert dup_v1, "expected the trainer-checkpoint loader state to duplicate buffered prompts"
    assert not dup_v11, f"journaled loader state must not duplicate: {sorted(dup_v11)}"
    assert not lost_v11, f"journaled loader state must not lose prompts: {sorted(lost_v11)}"
