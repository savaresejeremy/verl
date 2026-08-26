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
"""Crash-recovery tests: journal + fsck + group rescue, against a real TransferQueue.

Every test follows the production timeline — produce, journal, (maybe consume/produce
more), crash, recover — and then asserts what the buffer may and may not contain.
Several of these are regressions for bugs that only appear with a *durable* store:
index reuse across a snapshot, in-flight prompts with no trajectories, and samples
older than the controller's most recent index-allocation batch.
"""

from __future__ import annotations

import shutil
import time

import pytest
from helpers import (
    consume_group,
    crash_and_restart,
    finish_group,
    indexes_of,
    objects,
    submit_prompt,
    tag_prompt_finished,
)

from verl.experimental.durable_buffer import fsck, recover


def _journal(tq, path, step=1):
    """A journal tick: fence token first, then the metadata-only snapshot."""
    time.sleep(0.05)
    tq.save_checkpoint(str(path), metadata={"global_steps": step, "fence_ts": time.time()})
    time.sleep(0.05)


def _rewards_of(tq, partition_id, keys):
    data = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["rewards"])
    r = data["rewards"]
    vals = r.tolist() if hasattr(r, "tolist") else list(r)
    return [float(v[0]) if isinstance(v, list) else float(v) for v in vals]


def test_finished_trajectories_survive_a_crash(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    submit_prompt(tq, partition_id, "keep", status="running")
    keys = finish_group(tq, partition_id, "keep", n=4, value=7)
    tag_prompt_finished(tq, partition_id, "keep")
    journal = tmp_path / "journal"
    _journal(tq, journal)

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, journal, partition_id)

    live = tq.kv_list().get(partition_id, {})
    assert set(keys) <= set(live), f"finished trajectories lost: {report}"
    assert _rewards_of(tq, partition_id, keys) == [7.0] * 4, "restored trajectories must hold their own bytes"
    assert report["dropped_samples"] == 0 and report["fenced_objects"] == 0


def test_consumed_group_does_not_resurrect(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    for uid, val in (("gone", 1), ("stay", 2)):
        submit_prompt(tq, partition_id, uid, status="running")
        finish_group(tq, partition_id, uid, n=2, value=val)
        tag_prompt_finished(tq, partition_id, uid)
    journal = tmp_path / "journal"
    _journal(tq, journal)
    consume_group(tq, partition_id, "gone", n=2)  # trained AFTER the snapshot

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, journal, partition_id)

    live = tq.kv_list().get(partition_id, {})
    assert not any(k.startswith("gone") for k in live), "consumed group came back after restore"
    assert {"stay", "stay_0_0", "stay_1_0"} <= set(live)
    assert report["dropped_samples"] >= 2


def test_post_journal_writes_are_fenced(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    submit_prompt(tq, partition_id, "old", status="running")
    finish_group(tq, partition_id, "old", n=2, value=3)
    tag_prompt_finished(tq, partition_id, "old")
    journal = tmp_path / "journal"
    _journal(tq, journal)
    finish_group(tq, partition_id, "new", n=2, value=4)  # produced after the snapshot

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, journal, partition_id)

    live = tq.kv_list().get(partition_id, {})
    assert {"old", "old_0_0", "old_1_0"} <= set(live)
    assert not any(k.startswith("new") for k in live), "unknown post-snapshot samples must not appear"
    assert report["fenced_objects"] >= 4


def test_index_reuse_after_journal_never_serves_wrong_bytes(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """REGRESSION: TQ recycles freed indexes, so an index the journal still attributes to
    a consumed sample can hold another sample's bytes. Fencing on write identity, not on
    index membership, is what keeps the restore honest."""
    tq = tq_session
    submit_prompt(tq, partition_id, "victim", status="running")
    victim_keys = finish_group(tq, partition_id, "victim", n=2, value=11)
    tag_prompt_finished(tq, partition_id, "victim")
    victim_idx = set(indexes_of(tq, partition_id, victim_keys))
    journal = tmp_path / "journal"
    _journal(tq, journal)

    consume_group(tq, partition_id, "victim", n=2)  # frees the indexes
    finish_group(tq, partition_id, "usurper", n=2, value=99)  # very likely reuses them
    reused = victim_idx & set(indexes_of(tq, partition_id, ["usurper_0_0", "usurper_1_0"]))

    tq = crash_and_restart(tq_conf)
    recover(tq, store_root, journal, partition_id)
    live = tq.kv_list().get(partition_id, {})
    survivors = [k for k in live if k.startswith("victim_")]
    if reused:
        assert not survivors, "recycled-index samples must be dropped, not served"
    for key in survivors:  # if no reuse happened, at least the bytes must be the victim's
        assert _rewards_of(tq, partition_id, [key]) == [11.0]


def test_samples_older_than_the_last_allocation_batch_survive(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """REGRESSION: liveness must come from the partition, not from
    ``index_manager['allocated_indexes']`` (which is only the newest allocation batch)."""
    tq = tq_session
    uids = [f"batch{i}" for i in range(12)]
    for i, uid in enumerate(uids):
        submit_prompt(tq, partition_id, uid, status="running")
        finish_group(tq, partition_id, uid, n=2, value=i)
        tag_prompt_finished(tq, partition_id, uid)
    journal = tmp_path / "journal"
    _journal(tq, journal)

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, journal, partition_id)

    live = tq.kv_list().get(partition_id, {})
    kept = [u for u in uids if f"{u}_0_0" in live and f"{u}_1_0" in live]
    assert len(kept) == len(uids), f"buffer lost older samples: kept {len(kept)}/{len(uids)}, report={report}"
    assert report["fenced_objects"] == 0


def test_inflight_prompts_survive_recovery(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """In-flight prompts have no trajectories by design; they are what verl re-issues."""
    tq = tq_session
    submit_prompt(tq, partition_id, "running1", status="running")
    submit_prompt(tq, partition_id, "pending1", status="pending")
    submit_prompt(tq, partition_id, "done1", status="running")
    finish_group(tq, partition_id, "done1", n=2, value=5)
    tag_prompt_finished(tq, partition_id, "done1")
    journal = tmp_path / "journal"
    _journal(tq, journal)

    tq = crash_and_restart(tq_conf)
    recover(tq, store_root, journal, partition_id)

    live = tq.kv_list().get(partition_id, {})
    assert live.get("running1", {}).get("status") == "running", "in-flight prompt dropped by fsck"
    assert live.get("pending1", {}).get("status") == "pending"
    assert {"done1_0_0", "done1_1_0"} <= set(live)


def test_promote_complete_groups_rescues_untagged_group(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """A crash between the last trajectory write and the terminal prompt tag must not
    throw the group away."""
    tq = tq_session
    submit_prompt(tq, partition_id, "complete", status="running")
    finish_group(tq, partition_id, "complete", n=4, value=6)  # all present, prompt still 'running'
    submit_prompt(tq, partition_id, "partial", status="running")
    finish_group(tq, partition_id, "partial", n=2, value=6)  # half a group
    submit_prompt(tq, partition_id, "failed", status="running")
    finish_group(tq, partition_id, "failed", n=4, value=6, status="failure")
    journal = tmp_path / "journal"
    _journal(tq, journal)

    tq = crash_and_restart(tq_conf)
    # These prompts carry no persisted per-prompt count (verl only persists per-sample
    # __rollout_n__ columns), so rescue needs the operator's uniform-n assertion —
    # exactly what restore_tool's --group-size wires up.
    report = recover(tq, store_root, journal, partition_id, group_size=4, assume_uniform_group_size=True)
    promo = report["promotion"]

    assert promo["promoted"] == ["complete"], promo
    assert "partial" in promo["partial_left_for_reissue"]
    assert "failed" in promo["had_failures_left_for_reissue"], "all-failed groups belong to verl's refill path"
    live = tq.kv_list().get(partition_id, {})
    assert live["complete"]["status"] == "finished"
    assert live["partial"]["status"] == "running"


def test_fsck_does_not_fence_other_partitions_objects(tq_session, tq_conf, store_root, tmp_path):
    """REGRESSION: one durable store backs every partition (train/val/...), so a `train`
    restore must not delete `val`'s objects just because they are not in `train`'s
    key map."""
    tq = tq_session
    train_p, val_p = "train-multi", "val-multi"
    finish_group(tq, train_p, "t1", n=2, value=1)
    finish_group(tq, val_p, "v1", n=2, value=2)
    val_keys = ["v1_0_0", "v1_1_0"]
    journal = tmp_path / "journal"
    _journal(tq, journal)

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, journal, train_p)

    assert report["fenced_objects"] == 0, "fencing must be store-wide, not per-partition"
    live_val = tq.kv_list().get(val_p, {})
    assert set(val_keys) <= set(live_val), "val samples disappeared during a train restore"
    assert _rewards_of(tq, val_p, val_keys) == [2.0, 2.0]


def test_oplog_fsck_needs_no_per_object_stat(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """The default reconciliation cost must scale with churn plus one NAMES-ONLY
    readdir — never with per-object ``stat`` calls, which are what make full scans
    expensive on FSx (~0.3 ms each). The listing exists so snapshot/mutation straddles
    resolve to fence/drop (see fsck docstring)."""
    tq = tq_session
    for i in range(6):
        submit_prompt(tq, partition_id, f"big{i}", status="running")
        finish_group(tq, partition_id, f"big{i}", n=4, value=i)
        tag_prompt_finished(tq, partition_id, f"big{i}")
    journal = tmp_path / "journal"
    _journal(tq, journal)
    finish_group(tq, partition_id, "after", n=4, value=9)  # the only churn after the tick

    tq = crash_and_restart(tq_conf)
    tq.load_checkpoint(str(journal))
    report = fsck(tq, store_root, journal, partition_id)

    assert report["mode"] == "oplog"
    assert report["objects_statted"] == 0, "oplog mode must never stat objects"
    assert report["objects_listed"] > 0, "the names-only membership listing must run"
    assert report["fenced_objects"] == 8, report  # 4 trajectories x 2 fields
    assert report["oplog_records_replayed"] >= 8
    live = tq.kv_list().get(partition_id, {})
    assert all(f"big{i}_0_0" in live for i in range(6)), "pre-tick buffer must be intact"
    assert not any(k.startswith("after") for k in live)


def test_unusable_op_logs_abort_recovery_never_silent_mtime(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """A lost/truncated op log must ABORT recovery (fail closed) — never degrade to
    wall-clock fencing on its own. ``mode="mtime"`` remains available only as an
    EXPLICITLY-requested legacy flow that conservatively over-fences."""
    from verl.experimental.durable_buffer import UnrecoverableJournalError

    tq = tq_session
    submit_prompt(tq, partition_id, "fb", status="running")
    finish_group(tq, partition_id, "fb", n=2, value=3)
    tag_prompt_finished(tq, partition_id, "fb")
    journal = tmp_path / "journal"
    _journal(tq, journal)
    finish_group(tq, partition_id, "fb_post", n=2, value=4)

    # simulate a writer whose log vanished (e.g. store cloned without _oplog)
    oplog_dir = store_root / "_oplog"
    stash = tmp_path / "stashed_oplog"
    shutil.move(str(oplog_dir), str(stash))
    try:
        tq = crash_and_restart(tq_conf)
        with pytest.raises(UnrecoverableJournalError, match="disappeared"):
            recover(tq, store_root, journal, partition_id)
        # fail closed = the controller was never loaded: nothing can serve stale keys
        assert tq.kv_list().get(partition_id, {}) == {}

        # the explicit legacy escape hatch still reconciles, conservatively
        tq.load_checkpoint(str(journal))
        report = fsck(tq, store_root, journal, partition_id, mode="mtime")
        assert report["mode"] == "mtime", report
        assert report["objects_listed"] > 0, "the legacy flow must actually scan"
        live = tq.kv_list().get(partition_id, {})
        assert not any(k.startswith("fb_post") for k in live), "post-tick samples still fenced"
    finally:
        shutil.move(str(stash), str(oplog_dir))  # restore for later tests


def test_fsck_dry_run_changes_nothing(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    finish_group(tq, partition_id, "dry", n=2, value=2)
    journal = tmp_path / "journal"
    _journal(tq, journal)
    finish_group(tq, partition_id, "dry_post", n=2, value=2)

    tq = crash_and_restart(tq_conf)
    tq.load_checkpoint(str(journal))
    before = objects(store_root)
    report = fsck(tq, store_root, journal, partition_id, dry_run=True)
    assert report["fenced_objects"] > 0
    assert objects(store_root) == before, "dry run must not delete anything"
