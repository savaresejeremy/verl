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
"""Regressions for the round-2 CR review findings (round 3 of hardening).

Each section pins one review comment:

* snapshot/mutation straddles — a snapshot landing between a TQ storage mutation and
  its controller-metadata mutation resolves to fence (put-straddle) / drop
  (clear-straddle), never to a kept inconsistency;
* oplog=False fail-closed — superseded in round 4: the mode is removed entirely and
  recovery refuses such snapshots (see ``test_round4_on_cpu.py``);
* identity preflight — recovery rejects a journal whose store identity is missing,
  invalid or different BEFORE ``tq.load_checkpoint`` (which already mutates the
  caller's controller);
* restore-tool transaction — the two irreversible publication steps run inside a
  ``reconcile.state`` transaction; an interruption between them is completed on the
  next run (the freshness short-circuit alone would strand the mixed state), and
  dry-run repairs nothing;
* freshness matrix — equal snapshots and same-step fence_ts tie-breaks;
* cross-host name collisions — writer tokens keep tmp/op-log names unique when PID
  and counter collide between hosts;
* hostile names in the full manifest — the manifest round-trips names containing
  every separator ``str.splitlines()`` would split on;
* per-prompt ``__rollout_n__`` promotion — the expected session count comes from the
  prompt's own record when persisted, and promotion fails closed when the count is
  unverifiable or provably wrong;
* custom-sampler kwargs — re-supplied DAPO/failure-refill kwargs reach the base
  constructor (verl drops them for custom samplers), unknown keys error out, and
  ``background: false`` is honest poll-tick mode.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
import uuid
from pathlib import Path

import pytest
import torch
from helpers import (
    crash_and_restart,
    finish_group,
    indexes_of,
    make_fields,
    objects,
    submit_prompt,
    tag_prompt_finished,
)
from omegaconf import OmegaConf

from verl.experimental.durable_buffer import StoreMismatchError, fsck, recover
from verl.experimental.durable_buffer import restore_tool as restore_tool_mod
from verl.experimental.durable_buffer.buffer import DurableReplayBuffer
from verl.experimental.durable_buffer.fs_kv import (
    FENCE_FILE,
    OPLOG_DIR,
    DurableFSClient,
)
from verl.experimental.durable_buffer.fsck import _parse_object_name, _read_manifest, promote_complete_groups
from verl.experimental.durable_buffer.journal import STAMP_FILE, JournalWriter
from verl.experimental.durable_buffer.restore_tool import RECONCILE_STATE_FILE, restore

HOSTILE_FIELDS_SEPARATORS = ["nel\x85nel", "ls\u2028ls", "ps\u2029ps", "vt\x0bvt", "ff\x0cff"]


def _storage_client(tq):
    return tq.get_client().storage_manager.storage_client


def _tree_bytes(root: Path) -> dict[str, str]:
    """Relative path -> content hash for every file under root (byte-identity check)."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ------------------------------------------------- comment 2: snapshot straddles


def test_put_straddle_is_fenced_on_recovery(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """STORAGE put published before the snapshot, controller metadata never updated
    (the put-straddle): the object's op-log record falls BEFORE the captured offsets,
    so only the membership check can see it — it must be FENCED, and the pre-existing
    buffer must survive untouched."""
    tq = tq_session
    submit_prompt(tq, partition_id, "base", status="running")
    keys = finish_group(tq, partition_id, "base", n=2, value=6)
    tag_prompt_finished(tq, partition_id, "base")
    live_idx = indexes_of(tq, partition_id, [keys[0]])[0]

    client = _storage_client(tq)
    # (a) a foreign index the controller never allocated, (b) a live index with a
    # field the controller never registered — both storage-only mutations.
    client.put(["424242@rewards"], [torch.tensor([13.0])])
    client.put([f"{live_idx}@straddle_extra"], [torch.tensor([14.0])])

    journal = tmp_path / "journal"
    time.sleep(0.05)
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1, "fence_ts": time.time()})

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, journal, partition_id)

    assert report["mode"] == "oplog"
    assert not (store_root / "424242@rewards.bin").exists(), "unexpected object must be fenced"
    assert not (store_root / f"{live_idx}@straddle_extra.bin").exists(), "unregistered field must be fenced"
    assert report["fenced_objects"] >= 2, report
    live = tq.kv_list().get(partition_id, {})
    assert set(keys) <= set(live), "the registered group must survive the straddle cleanup"


def test_clear_straddle_drops_metadata_without_objects(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """STORAGE clear executed before the snapshot, controller metadata never cleared
    (the clear-straddle): its D records fall BEFORE the captured offsets, so only the
    missing-object check can see it — the stale metadata must be DROPPED (a kept key
    would be re-served and its read could only fail)."""
    tq = tq_session
    submit_prompt(tq, partition_id, "strad", status="running")
    gone_keys = finish_group(tq, partition_id, "strad", n=2, value=7)
    tag_prompt_finished(tq, partition_id, "strad")
    submit_prompt(tq, partition_id, "keep", status="running")
    keep_keys = finish_group(tq, partition_id, "keep", n=2, value=8)
    tag_prompt_finished(tq, partition_id, "keep")

    gone_idx = indexes_of(tq, partition_id, gone_keys)
    client = _storage_client(tq)
    client.clear([f"{i}@{f}" for i in gone_idx for f in ("rewards", "response_ids")])

    journal = tmp_path / "journal"
    time.sleep(0.05)
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1, "fence_ts": time.time()})

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, journal, partition_id)

    live = tq.kv_list().get(partition_id, {})
    assert not any(k.startswith("strad") for k in live), f"stale metadata kept after clear-straddle: {report}"
    assert set(keep_keys) <= set(live)
    assert report["dropped_samples"] >= 2, report
    assert not any(name.startswith(f"{gone_idx[0]}@") for name in objects(store_root))


# ---------------------------------------------- comment 3: oplog=False fail-closed
# (Superseded in round 4: the oplog=False tick mode is REMOVED — config rejection,
# journal refusal, and the controller-save->storage-hook gap regression live in
# test_round4_on_cpu.py.)


# ------------------------------------------------ comment 4: identity preflight


def test_recover_rejects_identityless_journal_before_any_mutation(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    tq = tq_session
    submit_prompt(tq, partition_id, "pre", status="running")
    finish_group(tq, partition_id, "pre", n=2, value=3)
    tag_prompt_finished(tq, partition_id, "pre")
    journal = tmp_path / "journal"
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1})

    # strip the identity from the snapshot (what a hand-assembled journal looks like)
    fence = json.loads((journal / FENCE_FILE).read_text())
    fence.pop("store_id")
    (journal / FENCE_FILE).write_text(json.dumps(fence))

    finish_group(tq, partition_id, "livework", n=2, value=9)  # state that load_checkpoint would wipe
    live_before = {p: dict(tags) for p, tags in tq.kv_list().items()}
    store_before = _tree_bytes(store_root)

    with pytest.raises(StoreMismatchError):
        recover(tq, store_root, journal, partition_id)

    assert {p: dict(tags) for p, tags in tq.kv_list().items()} == live_before, (
        "controller metadata mutated on a rejected recover (load_checkpoint must not have run)"
    )
    assert _tree_bytes(store_root) == store_before, "store bytes mutated on a rejected recover"

    # empty identity is as invalid as a missing one
    fence["store_id"] = ""
    (journal / FENCE_FILE).write_text(json.dumps(fence))
    with pytest.raises(StoreMismatchError):
        recover(tq, store_root, journal, partition_id)
    # ... and a malformed one
    fence["store_id"] = "not-a-uuid"
    (journal / FENCE_FILE).write_text(json.dumps(fence))
    with pytest.raises(StoreMismatchError):
        recover(tq, store_root, journal, partition_id)

    # the override is NARROW (round-5): it never covers a missing/empty/invalid
    # identity — there is no lineage to vouch for, and an unstamped/invalid side is
    # indistinguishable from a wrong --store path
    with pytest.raises(StoreMismatchError, match="does not apply"):
        recover(tq, store_root, journal, partition_id, dry_run=True, force_store_mismatch=True)
    # ... but two VALID, different identities remain overridable, and the ordinary
    # matched path still works
    fence["store_id"] = uuid.uuid4().hex
    (journal / FENCE_FILE).write_text(json.dumps(fence))
    report = recover(tq, store_root, journal, partition_id, dry_run=True, force_store_mismatch=True)
    assert report["mode"] == "oplog"


def test_fsck_rejects_a_store_without_identity(tq_session, partition_id, tmp_path):
    """One-sided missing on the STORE side fails closed too (a wrong --store path is
    exactly what an unstamped directory looks like)."""
    tq = tq_session
    finish_group(tq, partition_id, "idcheck", n=2, value=1)
    journal = tmp_path / "journal"
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1})

    bare = tmp_path / "unstamped_store"
    bare.mkdir()
    (bare / "0@rewards.bin").write_bytes(b"bytes that must survive")
    with pytest.raises(StoreMismatchError):
        fsck(tq, bare, journal, partition_id)
    assert (bare / "0@rewards.bin").read_bytes() == b"bytes that must survive"


# --------------------------------------- comment 5: restore-tool transaction marker


def _make_ckpt(tq, tmp_path, step: int):
    ckpt_dir = tmp_path / "ckpts"
    step_dir = ckpt_dir / f"global_step_{step}"
    step_dir.mkdir(parents=True)
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text(str(step))
    torch.save({"pos": f"step{step}"}, step_dir / "data.pt")
    tq.save_checkpoint(str(step_dir / "transfer_queue"), metadata={"global_steps": step})
    return ckpt_dir, step_dir


def test_interrupted_two_step_publication_is_completed_before_freshness(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    """Crash AFTER the TQ-dir swap but BEFORE data.pt: the freshness short-circuit
    alone would judge the half-published state 'equal' and strand it; the transaction
    marker makes the next run COMPLETE the data.pt half first."""
    tq = tq_session
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    finish_group(tq, partition_id, "txn", n=2, value=9)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": "journal6"},
        metadata_fn=lambda: {"global_steps": 6},
    )
    assert jw.tick() is not None
    tq.close()

    original = restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK

    def crash():
        raise RuntimeError("killed between the two publications")

    restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK = crash
    try:
        with pytest.raises(RuntimeError, match="between the two publications"):
            restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK = original

    # mixed state: new TQ dir, old data.pt, transaction not DONE
    state = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert state["state"] == "BEGIN"
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "step5"}

    # the re-run completes the transaction BEFORE its freshness logic would block it
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "checkpoint_fresher", result
    assert result["completed_interrupted_transaction"]["dataloader_state"] == "journal", result
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "journal6"}
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE"

    # convergence: a further re-run changes nothing
    snapshot = _tree_bytes(step_dir)
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "checkpoint_fresher"
    assert "completed_interrupted_transaction" not in result
    assert _tree_bytes(step_dir) == snapshot

    import transfer_queue as tq2

    tq2.init(tq_conf)
    tq2.load_checkpoint(str(step_dir / "transfer_queue"))
    assert {"txn_0_0", "txn_1_0"} <= set(tq2.kv_list().get(partition_id, {}))


def test_restore_tool_dry_run_reports_pending_repairs_and_touches_nothing(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    tq = tq_session
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    finish_group(tq, partition_id, "dryrepair", n=2, value=4)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
    assert jw.tick() is not None
    tq.close()

    original = restore_tool_mod._PUBLISH_CRASH_HOOK

    def crash():
        raise RuntimeError("killed mid-publication")

    restore_tool_mod._PUBLISH_CRASH_HOOK = crash
    try:
        with pytest.raises(RuntimeError, match="killed mid-publication"):
            restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._PUBLISH_CRASH_HOOK = original

    before_ckpt = _tree_bytes(ckpt_dir)
    before_store = _tree_bytes(store_root)
    result = restore(
        store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True
    )
    assert result["status"] == "dry_run", result
    assert result["would_heal_publication"] is True, "dry run must REPORT the pending heal"
    assert result["pending_transaction"]["state"] == "BEGIN", result
    # the completion decision is part of the reported plan (here: the swap never
    # became visible, so the marker would be discarded and the run decides fresh)
    assert result["would_complete_transaction"]["action"] == "discard_marker", result
    assert _tree_bytes(ckpt_dir) == before_ckpt, "dry run repaired/mutated the checkpoint"
    assert _tree_bytes(store_root) == before_store, "dry run mutated the store (byte-level)"

    # a real run then converges
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "reconciled", result
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE"

    import transfer_queue as tq2

    tq2.init(tq_conf)


# --------------------------------------------- comment 6: freshness matrix gaps


def _set_snapshot_freshness(snapshot_dir: Path, step: int, fence_ts: float) -> None:
    """Rewrite a snapshot's freshness signals, keeping the slot internally coherent.

    A journal slot's ``fence_ts`` lives in TQ's ``metadata.json`` AND in
    ``journal_stamp.json``, and the two must agree or the slot is refused as the product
    of two different ticks (``fsck.check_tick_coherence``). Forging one side only would
    make every freshness case here fail for the wrong reason, so the writer id is
    preserved and the stamp is moved with the metadata. A checkpoint snapshot has no
    stamp and needs no second write.
    """
    meta_path = snapshot_dir / "metadata.json"
    previous = {}
    if meta_path.exists():
        previous = json.loads(meta_path.read_text()).get("user_metadata", {}) or {}
    user_metadata = {"global_steps": step, "fence_ts": fence_ts}
    if "journal_writer" in previous:
        user_metadata["journal_writer"] = previous["journal_writer"]
    meta_path.write_text(json.dumps({"storage_saved": True, "user_metadata": user_metadata}))
    stamp_path = snapshot_dir / STAMP_FILE
    if stamp_path.exists():
        stamp = json.loads(stamp_path.read_text())
        stamp["fence_ts"] = fence_ts
        stamp_path.write_text(json.dumps(stamp))


def _freshness_case(tq, tq_conf, store_root, partition_id, tmp_path, journal_ts, ckpt_ts, step=5):
    finish_group(tq, partition_id, "fresh", n=2, value=1)  # the journal must know this partition
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=step)
    _set_snapshot_freshness(step_dir / "transfer_queue", step, ckpt_ts)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": step})
    info = jw.tick()
    assert info is not None
    _set_snapshot_freshness(Path(info["dir"]), step, journal_ts)
    tq.close()
    before_ckpt = _tree_bytes(ckpt_dir)
    before_store = _tree_bytes(store_root)
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    return result, before_ckpt, _tree_bytes(ckpt_dir), before_store, _tree_bytes(store_root)


def test_equal_snapshots_cause_no_mutation(tq_session, tq_conf, store_root, partition_id, tmp_path):
    result, ckpt0, ckpt1, store0, store1 = _freshness_case(
        tq_session, tq_conf, store_root, partition_id, tmp_path, journal_ts=1000.0, ckpt_ts=1000.0
    )
    assert result["status"] == "checkpoint_fresher", result
    assert ckpt1 == ckpt0, "an equal journal must leave the checkpoint byte-identical"
    assert store1 == store0, "an equal journal must leave the store byte-identical"

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_same_step_older_fence_ts_causes_no_mutation(tq_session, tq_conf, store_root, partition_id, tmp_path):
    result, ckpt0, ckpt1, store0, store1 = _freshness_case(
        tq_session, tq_conf, store_root, partition_id, tmp_path, journal_ts=999.0, ckpt_ts=1000.0
    )
    assert result["status"] == "checkpoint_fresher", result
    assert ckpt1 == ckpt0 and store1 == store0, "a same-step OLDER journal must not touch anything"

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_same_step_newer_fence_ts_applies_the_journal(tq_session, tq_conf, store_root, partition_id, tmp_path):
    result, ckpt0, ckpt1, _, _ = _freshness_case(
        tq_session, tq_conf, store_root, partition_id, tmp_path, journal_ts=1001.0, ckpt_ts=1000.0
    )
    assert result["status"] == "reconciled", result
    assert ckpt1 != ckpt0, "a same-step NEWER journal must be applied"

    import transfer_queue as tq2

    tq2.init(tq_conf)


# --------------------------------------- comment 7: cross-host name collisions


def test_writer_token_disambiguates_identical_pid_and_counter(tmp_path, monkeypatch):
    """Two hosts can hand two writers the same PID and the same counter values; only
    the per-writer random token keeps their generated names apart, in BOTH name
    families (tmp publish names and op-log file names)."""
    import os as os_mod

    monkeypatch.setattr(os_mod, "getpid", lambda: 4242)
    root = tmp_path / "store"
    a = DurableFSClient({"root": str(root), "serializer": "tq"})
    b = DurableFSClient({"root": str(root), "serializer": "tq"})
    a._tmp_counter = itertools.count()  # instance attrs shadow the class counter:
    b._tmp_counter = itertools.count()  # both clients now yield identical sequences

    path = a._path("3@rewards")
    assert a._tmp_path(path) != b._tmp_path(path), "identical pid+counter must still yield distinct tmp names"

    # inject: with the SAME token the names collide again — the token is load-bearing
    c = DurableFSClient({"root": str(root), "serializer": "tq"})
    c._writer_token = a._writer_token
    a._tmp_counter = itertools.count(7)
    c._tmp_counter = itertools.count(7)
    assert a._tmp_path(path) == c._tmp_path(path)

    a.put(["3@rewards"], [torch.tensor([1.0])])
    b.put(["3@rewards"], [torch.tensor([2.0])])
    logs = sorted(p.name for p in (root / OPLOG_DIR).glob("*.log"))
    assert len(logs) == 2, f"two same-pid writers must keep separate op logs, got {logs}"
    for client in (a, b, c):
        client.close()


# ------------------------------------ comment 8: hostile names in the manifest


def test_hostile_names_round_trip_through_the_full_manifest(tmp_path):
    root = tmp_path / "store"
    client = DurableFSClient({"root": str(root), "serializer": "tq"})
    keys = [f"9@{f}" for f in HOSTILE_FIELDS_SEPARATORS]
    client.put(keys, [torch.tensor([float(i)]) for i in range(len(keys))])

    snap = tmp_path / "snap"
    client.write_manifest(snap, full=True)
    manifest = _read_manifest(snap)
    assert manifest is not None, "hostile names must not make the manifest unusable"
    parsed = {_parse_object_name(name) for name in manifest}
    assert parsed == {(9, f) for f in HOSTILE_FIELDS_SEPARATORS}, parsed
    on_disk = {p.name for p in root.glob("*.bin")}
    assert set(manifest) == on_disk, "every object must appear in the manifest exactly once"
    client.close()


# --------------------------------- comment 10: per-prompt __rollout_n__ promotion


def _submit_prompt_with_rollout_n(tq, partition_id, uid, n):
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": "running", "global_steps": 1},
        fields={"prompt_ids": torch.arange(4).unsqueeze(0), "__rollout_n__": torch.tensor([n])},
    )


def test_promotion_honors_a_smaller_per_prompt_rollout_n(tq_session, partition_id):
    """actual < global: a prompt whose record says n=2 completes at 2 sessions even
    though the run-global rollout.n is 4."""
    tq = tq_session
    _submit_prompt_with_rollout_n(tq, partition_id, "small", 2)
    finish_group(tq, partition_id, "small", n=2, value=5)

    report = promote_complete_groups(tq, partition_id, group_size=4)
    assert "small" in report["promoted"], report
    assert tq.kv_list()[partition_id]["small"]["status"] == "finished"
    assert report["trajectories_rescued"] == 2, "rescue accounting must use the per-prompt n"


def test_promotion_honors_a_larger_per_prompt_rollout_n(tq_session, partition_id):
    """actual > global: a prompt whose record says n=8 must NOT promote at the global
    4 complete sessions — only once all 8 are successful."""
    tq = tq_session
    _submit_prompt_with_rollout_n(tq, partition_id, "large", 8)
    finish_group(tq, partition_id, "large", n=4, value=6)  # sessions 0..3 only

    report = promote_complete_groups(tq, partition_id, group_size=4)
    assert "large" not in report["promoted"], report
    assert "large" in report["partial_left_for_reissue"], report
    assert tq.kv_list()[partition_id]["large"]["status"] == "running"

    # complete sessions 4..7 -> now it promotes
    tq.kv_batch_put(
        keys=[f"large_{s}_0" for s in range(4, 8)],
        partition_id=partition_id,
        fields=make_fields(4, 6),
        tags=[{"status": "success", "global_steps": 1}] * 4,
    )
    report = promote_complete_groups(tq, partition_id, group_size=4)
    assert "large" in report["promoted"], report


def test_promotion_fails_closed_when_sessions_exceed_the_assumed_n(tq_session, partition_id):
    """Even under the operator's uniform-n assertion, observed evidence wins: sessions
    beyond the asserted n prove the assumption wrong for this prompt — promotion must
    be disabled, not guessed. (Without the assertion the prompt is already skipped as
    skipped_no_durable_n; see test_round4_on_cpu.py.)"""
    tq = tq_session
    submit_prompt(tq, partition_id, "overflow", status="running")  # payload without __rollout_n__
    finish_group(tq, partition_id, "overflow", n=6, value=7)  # sessions 0..5, all success

    report = promote_complete_groups(tq, partition_id, group_size=4, assume_uniform_group_size=True)
    assert "overflow" in report["skipped_unverifiable_n"], report
    assert "overflow" not in report["promoted"]
    assert tq.kv_list()[partition_id]["overflow"]["status"] == "running", "fail closed = leave for re-issue"


# ------------------------------------- comment 11: custom-sampler kwarg plumbing


def _buffer_kwargs(**durable):
    return dict(
        trainer_mode="separate_async",
        trainer_config=OmegaConf.create({}),
        max_off_policy_threshold=10**6,
        max_off_policy_strategy="drop",
        sampler_kwargs=OmegaConf.create({"durable_buffer": durable}),
        poll_interval=0.05,
    )


def test_resupplied_dapo_kwargs_reach_the_base_constructor():
    """verl passes filter_groups_metric / sync_refill_failed_groups only to non-custom
    samplers (trainer_base._build_replay_buffer); re-supplying them through
    sampler_kwargs.durable_buffer must land them on the base class, not drop them."""
    rb = DurableReplayBuffer(
        refill_fn=lambda n: n,
        **_buffer_kwargs(filter_groups_metric="acc", sync_refill_failed_groups=False),
    )
    assert rb.filter_groups_metric == "acc"
    assert rb.sync_refill_failed_groups is False
    assert rb._journal is None  # no journal_root -> journaling off, construction still fine


def test_resupplied_kwargs_get_base_class_validation():
    with pytest.raises(ValueError, match="requires refill_fn"):
        DurableReplayBuffer(refill_fn=None, **_buffer_kwargs(filter_groups_metric="acc"))


def test_unknown_durable_buffer_keys_error_instead_of_silently_dropping():
    with pytest.raises(ValueError, match="unknown keys.*journal_rot"):
        DurableReplayBuffer(refill_fn=None, **_buffer_kwargs(journal_rot="/tmp/x"))


def test_background_false_is_poll_tick_mode(tmp_path):
    """background: false must mean NO thread (ticks only from the sampler's poll
    loop, i.e. roughly once per training step) — the honest, documented semantics."""
    rb = DurableReplayBuffer(
        refill_fn=None,
        **_buffer_kwargs(journal_root=str(tmp_path / "j"), background=False),
    )
    assert rb._journal is not None
    assert rb._journal._thread is None, "poll-tick mode must not start a background thread"

    rb2 = DurableReplayBuffer(refill_fn=None, **_buffer_kwargs(journal_root=str(tmp_path / "j2")))
    try:
        assert rb2._journal._thread is not None, "default mode is the background thread"
    finally:
        rb2._journal.stop()
