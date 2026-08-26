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
"""Storage-backend tests: write-as-produced, atomic publish, and clear semantics.

The clear tests are regressions for a bug class that is invisible in the in-memory
backends: TQ computes a batch's field set as the *intersection* over its samples, so
verl's mixed ``_clear_groups`` call carries no field names at all. A backend that trusts
``metadata.field_names`` (or early-returns when it is empty) leaks every consumed
group's bytes and lets consumed samples come back after a restore.
"""

from __future__ import annotations

import json

import torch
from helpers import consume_group, finish_group, indexes_of, submit_prompt
from helpers import objects as _objects

from verl.experimental.durable_buffer.fs_kv import (
    FIELDS_REGISTRY,
    MANIFEST_FILE,
    OPLOG_DIR,
    OPLOG_OFFSETS_FILE,
    DurableFSClient,
)


def test_put_writes_objects_and_get_round_trips(tq_session, store_root, partition_id):
    tq = tq_session
    keys = finish_group(tq, partition_id, "uid1", n=2, value=5)
    objs = _objects(store_root)
    assert objs, "no durable objects were written"
    assert all(o.endswith(".bin") for o in objs)
    assert not list(store_root.glob("*.tmp")), "atomic publish left temp files behind"

    data = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["response_ids", "rewards"])
    rewards = data["rewards"]
    assert torch.allclose(rewards.reshape(-1).float(), torch.full((2,), 5.0))
    resp = data["response_ids"]
    rows = resp.unbind(0) if getattr(resp, "is_nested", False) else list(resp)
    assert int(rows[0].reshape(-1)[0]) == 5


def test_nested_and_nontensor_fields_round_trip(tq_session, store_root, partition_id):
    """verl stores jagged responses and NonTensor extra_fields; both must survive."""
    tq = tq_session
    from tensordict import TensorDict

    fields = TensorDict(
        {
            "response_ids": torch.arange(6).reshape(2, 3),
            "extra_fields": [
                {"uid": "a", "reward_extra_info": {"acc": 1.0}},
                {"uid": "b", "reward_extra_info": {"acc": 0.0}},
            ],
        },
        batch_size=[2],
    )
    keys = ["n_0_0", "n_1_0"]
    tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=[{"status": "success"}] * 2)
    out = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["extra_fields", "response_ids"])
    extras = out["extra_fields"]
    extras = extras.tolist() if hasattr(extras, "tolist") else list(extras)
    assert extras[0]["uid"] == "a" and extras[1]["reward_extra_info"]["acc"] == 0.0


def test_mixed_clear_deletes_trajectory_objects(tq_session, store_root, partition_id):
    """REGRESSION: verl clears prompt + trajectory keys in one call (empty field schema)."""
    tq = tq_session
    submit_prompt(tq, partition_id, "gA", status="finished", payload=False)  # tag-only prompt
    finish_group(tq, partition_id, "gA", n=2, value=7)
    submit_prompt(tq, partition_id, "gB", status="finished", payload=False)
    finish_group(tq, partition_id, "gB", n=2, value=8)

    before = set(_objects(store_root))
    consume_group(tq, partition_id, "gA", n=2)
    after = set(_objects(store_root))

    removed = before - after
    assert len(removed) == 4, f"expected group A's 4 objects deleted, removed={sorted(removed)}"
    # group B untouched
    live = tq.kv_list().get(partition_id, {})
    assert {"gB", "gB_0_0", "gB_1_0"} <= set(live)
    assert not any(k.startswith("gA") for k in live)


def test_clear_by_index_uses_field_registry_not_a_store_scan(tq_session, store_root, partition_id):
    """A consumer-only process must be able to delete a sample without listing the store."""
    tq = tq_session
    finish_group(tq, partition_id, "reg", n=1, value=3)
    assert (store_root / FIELDS_REGISTRY).exists(), "field registry not written"

    from verl.experimental.durable_buffer import fs_kv

    fresh = DurableFSClient({"root": str(store_root), "serializer": "tq"})  # never wrote anything
    scans_before = fs_kv.STATS["clear_index_scans"]
    fields = fresh._known_fields()
    assert "response_ids" in fields and "rewards" in fields
    idx = indexes_of(tq, partition_id, ["reg_0_0"])
    deleted = fresh.clear_by_index(idx)
    assert deleted >= 1
    assert fs_kv.STATS["clear_index_scans"] == scans_before, "should not have fallen back to a store scan"


def test_save_checkpoint_records_oplog_offsets_not_data(tq_session, store_root, partition_id, tmp_path):
    """Default snapshot artifact is O(writers), not O(store): each op log's length."""
    tq = tq_session
    finish_group(tq, partition_id, "man", n=2, value=4)
    ckpt = tmp_path / "tq_ckpt"
    tq.save_checkpoint(str(ckpt), metadata={"global_steps": 1})
    names = sorted(p.name for p in ckpt.iterdir())
    assert OPLOG_OFFSETS_FILE in names and "fence.json" in names and "controller_state.pkl" in names
    assert MANIFEST_FILE not in names, "the O(store) manifest must be opt-in"
    offsets = json.loads((ckpt / OPLOG_OFFSETS_FILE).read_text())
    assert offsets, "no op logs recorded"
    for logname, offset in offsets.items():
        assert (store_root / OPLOG_DIR / logname).stat().st_size >= offset
    assert not list(ckpt.rglob("*.bin")), "checkpoint must not contain store objects"


def test_full_manifest_is_available_for_deep_audits(tq_session, store_root, partition_id, tmp_path):
    tq = tq_session
    finish_group(tq, partition_id, "full", n=2, value=4)
    client = DurableFSClient({"root": str(store_root), "serializer": "tq"})
    out = tmp_path / "deep"
    rows_written = client.write_manifest(out, full=True)
    rows = [r for r in (out / MANIFEST_FILE).read_text().splitlines() if r]
    assert rows_written == len(rows) == len(_objects(store_root))
    assert all(len(r.split("\t")) == 4 for r in rows)  # name, inode, size, mtime


def test_store_is_empty_after_consuming_everything(tq_session, store_root, partition_id):
    """Storage hygiene: no object outlives its sample (the leak test verl's suite lacks)."""
    tq = tq_session
    my_indexes: set[int] = set()
    for uid in ("h1", "h2", "h3"):
        submit_prompt(tq, partition_id, uid, status="finished", payload=False)
        keys = finish_group(tq, partition_id, uid, n=2, value=1)
        my_indexes.update(indexes_of(tq, partition_id, keys))
    mine_before = {o for o in _objects(store_root) if int(o.split("@", 1)[0]) in my_indexes}
    assert len(mine_before) == 12, f"expected 3 groups x 2 trajectories x 2 fields, got {sorted(mine_before)}"
    for uid in ("h1", "h2", "h3"):
        consume_group(tq, partition_id, uid, n=2)
    mine_after = {o for o in _objects(store_root) if int(o.split("@", 1)[0]) in my_indexes}
    assert not mine_after, f"objects leaked after consume: {sorted(mine_after)}"


def test_torch_serializer_is_supported(store_root, partition_id):
    """The torch codec stays available as an escape hatch (TQ's own codec is the default)."""
    client = DurableFSClient({"root": str(store_root / "torch_codec"), "serializer": "torch"})
    client.put(["9@response_ids"], [torch.arange(4)])
    got = client.get(["9@response_ids"])
    assert torch.equal(got[0], torch.arange(4))
