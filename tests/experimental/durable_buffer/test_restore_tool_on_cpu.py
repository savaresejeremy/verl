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
"""End-to-end test of the offline restore tool (the exact cluster resume flow)."""

from __future__ import annotations

import time

import torch
from helpers import crash_and_restart, finish_group, objects, submit_prompt, tag_prompt_finished

from verl.experimental.durable_buffer.journal import JournalWriter
from verl.experimental.durable_buffer.restore_tool import restore


def test_restore_tool_rewrites_checkpoint_tq_state(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session

    # --- a fake trainer checkpoint layout at step 3 (older than the journal)
    ckpt_dir = tmp_path / "ckpts"
    step_dir = ckpt_dir / "global_step_3"
    step_dir.mkdir(parents=True)
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text("3")
    torch.save({"pos": "step3"}, step_dir / "data.pt")
    tq.save_checkpoint(str(step_dir / "transfer_queue"), metadata={"global_steps": 3})

    # --- work after the checkpoint: two more groups, then a journal tick with dataloader state
    for i, uid in enumerate(("late1", "late2")):
        submit_prompt(tq, partition_id, uid, status="running", global_steps=4)
        finish_group(tq, partition_id, uid, n=2, value=40 + i, global_steps=4)
        tag_prompt_finished(tq, partition_id, uid, global_steps=4)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": "step4-late"},
        metadata_fn=lambda: {"global_steps": 4},
    )
    assert jw.tick() is not None
    time.sleep(0.05)
    # post-journal churn that fsck must fence
    finish_group(tq, partition_id, "post_tick", n=2, value=99, global_steps=5)

    # --- crash
    tq = crash_and_restart(tq_conf)
    tq.close()  # restore_tool owns its own TQ lifecycle

    result = restore(
        store=store_root,
        journal_root=journal_root,
        ckpt_dir=ckpt_dir,
        partition=partition_id,
        group_size=2,
    )
    assert result["status"] == "reconciled", result
    assert result["source_step"] == 4
    assert result["dataloader_state"] == "journal"
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "step4-late"}
    assert (step_dir / "data.pt.pre_reconcile").exists()

    # --- now do exactly what stock verl does on resume: load the checkpoint's TQ dir
    import transfer_queue as tq2

    tq2.init(tq_conf)
    tq2.load_checkpoint(str(step_dir / "transfer_queue"))
    live = tq2.kv_list().get(partition_id, {})
    assert {"late1_0_0", "late1_1_0", "late2_0_0", "late2_1_0"} <= set(live), (
        "post-checkpoint groups must be visible to a stock verl resume"
    )
    assert not any(k.startswith("post_tick") for k in live), "post-journal churn must stay fenced"
    data = tq2.kv_batch_get(keys=["late1_0_0"], partition_id=partition_id, select_fields=["rewards"])
    rewards = data["rewards"]
    val = float(rewards.reshape(-1)[0]) if hasattr(rewards, "reshape") else float(rewards[0])
    assert val == 40.0


def test_restore_tool_dry_run_touches_nothing(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    ckpt_dir = tmp_path / "ckpts"
    step_dir = ckpt_dir / "global_step_1"
    step_dir.mkdir(parents=True)
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text("1")
    tq.save_checkpoint(str(step_dir / "transfer_queue"), metadata={"global_steps": 1})
    before = sorted(p.name for p in (step_dir / "transfer_queue").iterdir())

    finish_group(tq, partition_id, "dr", n=2, value=1)
    journal_root = tmp_path / "journals"
    JournalWriter(journal_root, interval_s=0.0, ring=2).tick()
    time.sleep(0.05)
    # post-journal churn: a real run would fence these -- a dry run must NOT
    finish_group(tq, partition_id, "dr_post", n=2, value=2)
    store_before = objects(store_root)

    tq = crash_and_restart(tq_conf)
    tq.close()
    result = restore(
        store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True
    )
    assert result["status"] == "dry_run"
    assert result["fsck"]["fenced_objects"] >= 4, "dry run must still REPORT the pending fences"
    assert objects(store_root) == store_before, "dry run deleted store objects"
    assert sorted(p.name for p in (step_dir / "transfer_queue").iterdir()) == before
    # keep the session fixture usable for any later tests
    import transfer_queue as tq3

    tq3.init(tq_conf)
