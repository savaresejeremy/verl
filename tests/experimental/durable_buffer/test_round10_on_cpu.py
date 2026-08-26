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
"""Regressions for the round-10 design review (distributed-systems attack).

The eight behaviours it added, and what each test pins:

1.  **``DurableFS.oplog_fsync``** — the intent log is flushed, not fsynced, so the
    record-before-publish ordering holds across a killed process and not across a lost
    node. The opt-in barrier makes the batch durable before the caller publishes
    anything; it is OFF by default because it costs shared-filesystem throughput.
2.  **Store exclusivity for the restore tool** — ``tq.init`` ignores ``create=False``
    whenever a controller is reachable, and ``recover(..., dry_run=True)`` still calls
    ``load_checkpoint``, so a preview against a live job destroys its buffer index. The
    tool refuses with ``live_controller_present`` in BOTH modes and brings up its own
    Ray rather than joining a cluster through ``ray.get_actor``'s auto-init.
3.  **Split restore gate** — a refused journal no longer blocks healing this tool's own
    interrupted swap (status ``healed_only``), and any outcome leaving the checkpoint
    without a live ``transfer_queue`` dir exits non-zero, because verl's resume skips a
    missing one silently and starts with an empty buffer.
4.  **Journal dual-writer protection** — one writer per ring root (an ``O_CREAT|O_EXCL``
    lease) plus a read-time cross-check that a slot's stamp and its TQ metadata came
    from one tick, which is what a bypassed lease would otherwise produce: a slot whose
    op-log fence was captured after its controller snapshot.
5.  **``load_journal`` hoisted into ``recover``'s preflight** — the controller pickle is
    a plain file read, and parsing it late was the one rejection class evaluated after
    ``tq.load_checkpoint`` had already loaded the snapshot into the live controller.
6.  **Controller-schema pin** — TQ's pickle carries live class references and no schema
    version, so a TransferQueue upgrade between snapshot and restore was undetectable.
7.  **Ring-slot reach** — a mid-record offsets fence aligns down instead of aborting, a
    poisoned newest slot falls back to the next-older complete one, and
    ``--journal-slot`` names a slot instead of an operator moving directories by hand.
8.  **Ordering and coupling signals** — slots carry a monotonic tick sequence rather
    than being ranked by wall clock alone, ``start_journal`` records the trainer step by
    default, and a journal running without the dataloader position says so.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path

import pytest
import torch
from helpers import finish_group
from omegaconf import OmegaConf
from test_round6_on_cpu import _make_ckpt, _tree_bytes

from verl.experimental.durable_buffer import restore_tool as restore_tool_mod
from verl.experimental.durable_buffer.buffer import DurableReplayBuffer
from verl.experimental.durable_buffer.fs_kv import (
    OPLOG_DIR,
    OPLOG_OFFSETS_FILE,
    STATS,
    DurableFSClient,
    backend_config,
    bootstrap_durable_fs,
)
from verl.experimental.durable_buffer.fsck import (
    SNAPSHOT_ATTRIBUTES,
    UnrecoverableJournalError,
    _align_start_to_record_boundary,
    _read_oplog_tail,
    _transfer_queue_version,
    check_journal_recoverable,
    check_snapshot_schema,
    check_tick_coherence,
    load_journal,
    recover,
    snapshot_schema_fingerprint,
)
from verl.experimental.durable_buffer.journal import (
    STAMP_FILE,
    WRITER_LEASE_FILE,
    JournalWriter,
    JournalWriterLeaseError,
    complete_journal_slots,
    latest_journal,
    select_journal_slot,
)
from verl.experimental.durable_buffer.restore_tool import restore

# ----------------------------------------------------------------- 1. oplog_fsync


def _client(root: Path, **extra) -> DurableFSClient:
    return DurableFSClient({"root": str(root), "serializer": "tq", "create": True, **extra})


def test_oplog_fsync_defaults_off_and_is_config_reachable(tmp_path):
    """The barrier is opt-in from the same place every other backend knob is set, and
    absent from the config it stays off."""
    assert bootstrap_durable_fs(backend_config(tmp_path / "a"))["oplog_fsync"] is False
    assert bootstrap_durable_fs(backend_config(tmp_path / "b", oplog_fsync=True))["oplog_fsync"] is True
    # the raw YAML shape an operator writes, without the helper
    conf = OmegaConf.create(
        {"backend": {"storage_backend": "DurableFS", "DurableFS": {"root": str(tmp_path / "c"), "oplog_fsync": True}}}
    )
    assert bootstrap_durable_fs(conf)["oplog_fsync"] is True
    assert _client(tmp_path / "d").oplog_fsync is False
    assert _client(tmp_path / "e", oplog_fsync=True).oplog_fsync is True


def test_oplog_fsync_barriers_the_batch_before_the_publish(tmp_path, monkeypatch):
    """ON: one fsync per mutation BATCH (not per object), and it lands BEFORE the first
    object of that batch is published — the ordering the flush-only default cannot
    guarantee across a lost node, because a durable rename can outlive a log record
    still sitting in a client cache."""
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (events.append("replace"), real_replace(a, b))[1])

    client = _client(tmp_path / "on", oplog_fsync=True)
    client.put([f"{i}@rewards" for i in range(4)], [torch.tensor([float(i)]) for i in range(4)])
    client.close()
    assert events[0] == "fsync", f"the barrier must precede every publish in the batch: {events}"
    assert events.count("fsync") == 1, f"one barrier per batch, not per object: {events}"
    assert events.count("replace") == 4

    events.clear()
    off = _client(tmp_path / "off")
    off.put(["7@rewards"], [torch.tensor([7.0])])
    off.close()
    assert "fsync" not in events, "the default must not pay for a barrier"


def test_oplog_fsync_leaves_a_replayable_log(tmp_path):
    """The barrier changes durability, not the log's content: the records a fenced
    recovery reads back are exactly the ones a flush-only writer would have written."""
    before = STATS["oplog_fsyncs"]
    client = _client(tmp_path / "s", oplog_fsync=True)
    client.put(["3@rewards"], [torch.tensor([3.0])])
    client.clear(["3@rewards"])
    client.close()
    assert STATS["oplog_fsyncs"] == before + 2, "one counted barrier per logged batch"
    (log,) = list((tmp_path / "s" / OPLOG_DIR).glob("*.log"))
    assert log.read_text().splitlines() == ["P 3@rewards.bin", "D 3@rewards.bin"]


# ------------------------------------------------- 2. store exclusivity / live controller


def test_restore_refuses_a_reachable_controller_in_both_modes(tq_session, store_root, partition_id, tmp_path):
    """The session TQ is UP, which is exactly the partial-failure case (driver alive,
    trainer wedged, operator restoring). Both modes must refuse before reading anything
    out of the store, with byte-identical trees, because the dry run is NOT read-only
    with respect to a live controller: its own phase 2 calls tq.load_checkpoint on
    whatever controller tq.init resolved."""
    tq = tq_session
    finish_group(tq, partition_id, "live10", n=2, value=5)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)

    for dry in (True, False):
        out = restore(store=store_root, journal_root=None, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        assert out["status"] == "live_controller_present", (dry, out)
        assert out["dry_run"] is dry
        assert out["plan_available"] is False, "the refusal precedes the preflight"
        assert out["controller"]["actor"] == "TransferQueueController"
        assert out["controller"]["reachable"] is True
        assert restore_tool_mod._exit_ok(out) is False
        assert _tree_bytes(store_root) == before_store, f"dry={dry}: refused restore touched the store"
        assert _tree_bytes(ckpt_dir) == before_ckpt, f"dry={dry}: refused restore touched the checkpoint"
    assert not (step_dir / "reconcile.lock").exists(), "the refusal must precede even the lock"


def test_allow_live_controller_overrides_the_refusal(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """The override exists for a leftover controller an operator has verified. Probed at
    the seam (the real controller is closed, the probe is forced to report one), so the
    test proves the flag is honored without reconciling against a live job."""
    tq = tq_session
    finish_group(tq, partition_id, "ovr10", n=2, value=4)
    ckpt_dir, _ = _make_ckpt(tq, tmp_path, step=5)
    tq.close()

    original = restore_tool_mod._live_controller
    # The real probe also fills in a record of what it did (round 12), which the refusal
    # detail and the report read; a stub standing in for it has to accept that argument.
    restore_tool_mod._live_controller = lambda **_kwargs: {"actor": "TransferQueueController", "reachable": True}
    try:
        refused = restore(store=store_root, journal_root=None, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
        assert refused["status"] == "live_controller_present", refused
        allowed = restore(
            store=store_root,
            journal_root=None,
            ckpt_dir=ckpt_dir,
            partition=partition_id,
            dry_run=True,
            allow_live_controller=True,
        )
        assert allowed["status"] == "dry_run", allowed
        assert allowed["plan_available"] is True
    finally:
        restore_tool_mod._live_controller = original
        import transfer_queue as tq2

        tq2.init(tq_conf)


def test_isolated_ray_never_replaces_a_ray_the_caller_owns(tq_session):
    """The tool starts a PRIVATE Ray only when it has none. An initialized Ray belongs
    to the caller (in production, to nothing — the tool is offline; here, to the test
    session), and the exclusivity probe has already run inside it."""
    import ray

    assert ray.is_initialized(), "the session TQ must have brought Ray up"
    assert restore_tool_mod._ensure_isolated_ray() is False, "an existing Ray must be reused, never re-initialized"
    assert ray.is_initialized()


# ---------------------------------------------------------- 3. exit code / healed_only


@pytest.mark.parametrize(
    ("status", "tq_present", "expected"),
    [
        ("reconciled", True, True),
        ("dry_run", True, True),
        ("checkpoint_fresher", True, True),
        ("ordering_ambiguous", True, True),
        ("reconciled", False, False),  # a "success" that left no live dir is not one
        ("checkpoint_fresher", False, False),
        ("healed_only", True, False),  # resumable again, but the journal refusal is the operator's
        ("unrecoverable_journal", True, False),
        ("store_mismatch", True, False),
        ("live_controller_present", None, False),
        ("locked", None, False),
        ("no_checkpoint", None, False),
    ],
)
def test_exit_code_requires_both_an_ok_status_and_a_live_tq_dir(status, tq_present, expected):
    """``trainer_base._load_checkpoint`` does ``if os.path.exists(tq_ckpt_path)`` and
    skips a missing TransferQueue dir with no warning, so a checkpoint left without one
    resumes with an empty controller and every buffered trajectory becomes orphaned
    bytes. An automated restart wrapper must not be able to walk past that on a zero
    exit, whatever the status says."""
    assert restore_tool_mod._exit_ok({"status": status, "checkpoint_tq_present": tq_present}) is expected


# ------------------------------------------------------- 4. journal dual-writer guards


def test_one_writer_per_ring_root(tmp_path):
    """Both documented wirings (``start_journal`` and the buffer subclass) can be active
    in one driver, and a copy-pasted config points two drivers at one root. The second
    writer is refused the lease at CONSTRUCTION instead of interleaving its artifact
    writes; the wiring sites turn that refusal into journaling-off rather than a dead run
    (``test_round11_on_cpu.py``)."""
    root = tmp_path / "ring"
    first = JournalWriter(root, interval_s=0.0, ring=2)
    assert (root / WRITER_LEASE_FILE).exists()
    holder = json.loads((root / WRITER_LEASE_FILE).read_text())
    assert holder["writer"] == first.writer_id and holder["pid"] == os.getpid()

    with pytest.raises(JournalWriterLeaseError, match="exactly one writer"):
        JournalWriter(root, interval_s=0.0, ring=2)

    first.stop()
    assert not (root / WRITER_LEASE_FILE).exists(), "stop() releases the lease"
    second = JournalWriter(root, interval_s=0.0, ring=2)  # the root is free again
    second.stop()


def test_a_lease_whose_holder_stopped_ticking_is_taken_over(tmp_path, caplog):
    """A driver killed hard leaves its lease behind, and this package exists for exactly
    that restart: refusing forever would wedge journaling on the recovery path. Takeover
    is keyed on the heartbeat (decidable from the filesystem) rather than on pid liveness
    (which is not decidable across hosts, which is why the restore lock is never broken)."""
    root = tmp_path / "ring"
    dead = JournalWriter(root, interval_s=0.0, ring=2)
    stale = json.loads((root / WRITER_LEASE_FILE).read_text())
    stale["heartbeat_at"] = time.time() - 3600
    (root / WRITER_LEASE_FILE).write_text(json.dumps(stale))

    with caplog.at_level(logging.WARNING):
        heir = JournalWriter(root, interval_s=0.0, ring=2)
    assert "taking over the writer lease" in caplog.text
    assert json.loads((root / WRITER_LEASE_FILE).read_text())["writer"] == heir.writer_id
    # and the dispossessed writer must not remove its successor's lease
    dead.stop()
    assert json.loads((root / WRITER_LEASE_FILE).read_text())["writer"] == heir.writer_id
    heir.stop()


def test_a_tick_fails_when_the_lease_changed_hands(tq_session, tmp_path, partition_id):
    """The lease is re-asserted at the START of every tick, so a writer that lost it
    publishes no slot rather than racing the new holder's artifact writes."""
    finish_group(tq_session, partition_id, "lease10", n=2, value=1)
    root = tmp_path / "ring"
    writer = JournalWriter(root, interval_s=0.0, ring=2)
    assert writer.tick() is not None
    (root / WRITER_LEASE_FILE).write_text(json.dumps({"writer": "someone-else", "heartbeat_at": time.time()}))
    before = complete_journal_slots(root)
    assert writer.tick() is None, "a tick without the lease must fail"
    assert writer.stats["failures"] == 1
    assert complete_journal_slots(root) == before, "and publish nothing"


def test_a_real_tick_writes_one_coherent_fence_pair(tq_session, tmp_path, partition_id):
    """The free coherence check: one tick writes ONE fence_ts into TQ's metadata.json
    and into journal_stamp.json, plus the writer id on both sides."""
    finish_group(tq_session, partition_id, "pair10", n=2, value=2)
    root = tmp_path / "ring"
    writer = JournalWriter(root, interval_s=0.0, ring=2)
    info = writer.tick()
    assert info is not None
    slot = Path(info["dir"])
    stamp = json.loads((slot / STAMP_FILE).read_text())
    md = json.loads((slot / "metadata.json").read_text())["user_metadata"]
    assert stamp["fence_ts"] == md["fence_ts"] == info["fence_ts"]
    assert stamp["journal_writer"] == md["journal_writer"] == writer.writer_id
    check_tick_coherence(slot)  # the gate accepts it
    writer.stop()


def test_a_slot_assembled_from_two_ticks_is_refused(tq_session, tmp_path, partition_id):
    """What two writers on one root actually produce: the artifact writes after
    ``tq.save_checkpoint`` resolve their paths at write time, so one writer's stamp can
    land on top of the controller pickle the other just renamed in. Every completeness
    gate passes and the slot's fence is LATER than its pickle, which is the ordering the
    design refuses whenever it can see it. The pair is what lets it see it."""
    finish_group(tq_session, partition_id, "mix10", n=2, value=3)
    root = tmp_path / "ring"
    writer = JournalWriter(root, interval_s=0.0, ring=2)
    slot = Path(writer.tick()["dir"])
    writer.stop()

    for field, value in (("fence_ts", 1.0), ("journal_writer", "another-writer")):
        stamp = json.loads((slot / STAMP_FILE).read_text())
        original = stamp[field]
        stamp[field] = value
        (slot / STAMP_FILE).write_text(json.dumps(stamp))
        with pytest.raises(UnrecoverableJournalError, match="not the product of a single journal tick"):
            check_journal_recoverable(slot)
        stamp[field] = original
        (slot / STAMP_FILE).write_text(json.dumps(stamp))
    check_journal_recoverable(slot)  # restored: trustworthy again


def test_a_trainer_checkpoint_is_not_judged_on_a_pair_it_never_writes(tq_session, tmp_path, partition_id):
    """verl's own ``tq.save_checkpoint`` writes no journal stamp, so there is no pair to
    compare: the checkpoint source must stay recoverable (it is the no-journal fallback)."""
    finish_group(tq_session, partition_id, "ckpt10", n=2, value=4)
    _, step_dir = _make_ckpt(tq_session, tmp_path, step=5)
    ckpt_tq = step_dir / "transfer_queue"
    assert not (ckpt_tq / STAMP_FILE).exists()
    check_tick_coherence(ckpt_tq)
    check_snapshot_schema(ckpt_tq)
    check_journal_recoverable(ckpt_tq)


# ------------------------------------------- 5. the controller pickle in the preflight


class _RecordingTQ:
    """The two calls ``recover`` makes on a TQ instance, with nothing behind them."""

    def __init__(self):
        self.loaded = []

    def load_checkpoint(self, path):
        self.loaded.append(path)


def test_a_corrupt_controller_pickle_is_refused_before_load_checkpoint(tq_session, store_root, tmp_path, partition_id):
    """``load_journal`` needs no TQ instance: it is a pure file read, like the three
    checks that were already hoisted. Left inside ``fsck`` it ran AFTER
    ``tq.load_checkpoint``, so a truncated or version-drifted pickle loaded into the
    caller's live controller and only then blew up — as a raw traceback, past the point
    where the controller could still be left untouched."""
    finish_group(tq_session, partition_id, "pkl10", n=2, value=6)
    root = tmp_path / "ring"
    writer = JournalWriter(root, interval_s=0.0, ring=2)
    slot = Path(writer.tick()["dir"])
    writer.stop()
    (slot / "controller_state.pkl").write_bytes(b"\x80\x04 truncated garbage")

    stub = _RecordingTQ()
    with pytest.raises(UnrecoverableJournalError, match="cannot be parsed into the reconciliation view"):
        recover(stub, store_root, slot, partition_id)
    assert stub.loaded == [], "a refused journal must never reach the caller's controller"


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        (b"not a pickle at all", "unpicklable bytes"),
        (pickle.dumps({"no_partitions_key": 1}), "a payload without partitions"),
        (pickle.dumps({"partitions": {"other": object()}}), "a payload without the asked-for partition"),
    ],
)
def test_load_journal_classifies_every_parse_failure_as_unrecoverable(tmp_path, payload, why):
    """Any shape that cannot be parsed is an UNTRUSTED ARTIFACT and gets the same
    fail-closed classification as a bad offsets fence, so the restore tool reports a
    controlled status instead of an AttributeError/KeyError traceback."""
    slot = tmp_path / "slot"
    slot.mkdir()
    (slot / "controller_state.pkl").write_bytes(payload)
    with pytest.raises(UnrecoverableJournalError, match="cannot be parsed into the reconciliation view"):
        load_journal(slot, "train")


# --------------------------------------------------------------- 6. schema pin


def test_a_real_snapshot_carries_every_attribute_reconciliation_reads(tq_session, tmp_path, partition_id):
    """The schema pin itself: take a REAL controller snapshot and assert the attributes
    ``load_journal`` reads exist on it. An upstream rename then fails here, in CI, and
    not at 3am on a restore."""
    finish_group(tq_session, partition_id, "schema10", n=2, value=7)
    root = tmp_path / "ring"
    writer = JournalWriter(root, interval_s=0.0, ring=2)
    slot = Path(writer.tick()["dir"])
    writer.stop()

    state = pickle.loads((slot / "controller_state.pkl").read_bytes())
    assert partition_id in state["partitions"], sorted(state["partitions"])
    snapshot = state["partitions"][partition_id]
    missing = [name for name in SNAPSHOT_ATTRIBUTES if not hasattr(snapshot, name)]
    assert not missing, f"TransferQueue's controller snapshot no longer carries {missing}"
    # field_metadata's members carry the per-field index set reconciliation expands
    for meta in snapshot.field_metadata.values():
        assert hasattr(meta, "global_indexes")
    stamp = json.loads((slot / STAMP_FILE).read_text())
    assert stamp["snapshot_schema"] == snapshot_schema_fingerprint()
    assert stamp["tq_version"] == _transfer_queue_version()


def test_a_drifted_controller_schema_is_refused_naming_both_versions(tq_session, tmp_path, partition_id):
    """A fingerprint mismatch means the attributes the snapshot carries are not the ones
    reconciliation reads, so the pickle would either crash or, worse, read a renamed
    attribute's stale namesake and reconcile against drifted semantics.

    Forging the stamp covers the reader-contract half (verl changing what it reads) and
    the refusal message. The TransferQueue half — a rename moving the value the tick
    records — is exercised against a real snapshot in
    ``test_round11_on_cpu.py::test_a_transfer_queue_side_rename_moves_the_fingerprint_and_is_refused``.
    """
    finish_group(tq_session, partition_id, "drift10", n=2, value=8)
    root = tmp_path / "ring"
    writer = JournalWriter(root, interval_s=0.0, ring=2)
    slot = Path(writer.tick()["dir"])
    writer.stop()

    stamp = json.loads((slot / STAMP_FILE).read_text())
    stamp["snapshot_schema"] = "custom_meta,field_metadata,global_indexes,keys_map,revert_keys_mapping"
    stamp["tq_version"] = "0.1.4"
    (slot / STAMP_FILE).write_text(json.dumps(stamp))
    with pytest.raises(UnrecoverableJournalError, match="different TransferQueue controller schema") as exc:
        check_snapshot_schema(slot)
    assert "0.1.4" in str(exc.value) and str(_transfer_queue_version()) in str(exc.value)


def test_a_bare_version_difference_with_a_matching_fingerprint_proceeds(tq_session, tmp_path, partition_id, caplog):
    """The fields reconciliation reads are unchanged, so the snapshot is readable: warn,
    do not refuse. Refusing every patch bump would make the pin useless in practice."""
    finish_group(tq_session, partition_id, "ver10", n=2, value=9)
    root = tmp_path / "ring"
    writer = JournalWriter(root, interval_s=0.0, ring=2)
    slot = Path(writer.tick()["dir"])
    writer.stop()

    stamp = json.loads((slot / STAMP_FILE).read_text())
    stamp["tq_version"] = "0.0.1-ancient"
    (slot / STAMP_FILE).write_text(json.dumps(stamp))
    with caplog.at_level(logging.WARNING):
        check_snapshot_schema(slot)
    assert "0.0.1-ancient" in caplog.text and "unchanged, so the snapshot is readable" in caplog.text


def test_a_snapshot_without_a_fingerprint_is_not_judged(tq_session, tmp_path, partition_id):
    """Journals written before the pin existed stay recoverable: there is nothing to
    compare, and refusing them would make an upgrade destroy recoverability."""
    finish_group(tq_session, partition_id, "old10", n=2, value=1)
    root = tmp_path / "ring"
    writer = JournalWriter(root, interval_s=0.0, ring=2)
    slot = Path(writer.tick()["dir"])
    writer.stop()

    stamp = json.loads((slot / STAMP_FILE).read_text())
    stamp.pop("snapshot_schema")
    stamp.pop("tq_version")
    (slot / STAMP_FILE).write_text(json.dumps(stamp))
    check_snapshot_schema(slot)
    check_journal_recoverable(slot)


# ------------------------------------------------ 7a. mid-record offsets align down


def _fake_oplog(store_root: Path, name: str, records: list[str]) -> Path:
    d = store_root / OPLOG_DIR
    d.mkdir(parents=True, exist_ok=True)
    log = d / name
    log.write_text("".join(f"{r}\n" for r in records), encoding="utf-8")
    return log


def test_a_mid_record_offset_aligns_down_instead_of_aborting(tmp_path):
    """The fence is a file SIZE sampled in the driver while writers append from other
    nodes. Between calls a log rests on a record boundary, but a size sampled during a
    large multi-page append and observed from another node need not. Aborting there
    refuses a valid, internally coherent journal; replaying from the previous boundary
    treats the straddling record as post-cut, which over-fences — the direction every
    other decision here prefers."""
    store = tmp_path / "store"
    (store / "_store_id").parent.mkdir(parents=True, exist_ok=True)
    log = _fake_oplog(store, "w.log", ["P 1@rewards.bin", "P 2@rewards.bin", "D 3@rewards.bin"])
    boundary = len("P 1@rewards.bin\n")
    mid = boundary + 4  # inside "P 2@rewards.bin"

    journal = tmp_path / "slot"
    journal.mkdir()
    (journal / OPLOG_OFFSETS_FILE).write_text(json.dumps({"w.log": mid}))
    written, deleted = _read_oplog_tail(store, journal)
    assert written == {"2@rewards.bin"}, "the straddling record must be replayed, i.e. fenced"
    assert deleted == {"3@rewards.bin"}
    assert "1@rewards.bin" not in written, "alignment must go DOWN one record, not to the start"

    # an offset with no boundary behind it replays the whole log: fences more, loses nothing
    (journal / OPLOG_OFFSETS_FILE).write_text(json.dumps({"w.log": 3}))
    written, _ = _read_oplog_tail(store, journal)
    assert written == {"1@rewards.bin", "2@rewards.bin"}

    with open(log, "rb") as fh:
        assert _align_start_to_record_boundary(fh, boundary, "w.log") == boundary, "a boundary offset must not move"
        assert _align_start_to_record_boundary(fh, 0, "w.log") == 0


def test_a_malformed_record_still_aborts_after_alignment(tmp_path):
    """Alignment is not a relaxation of the fence contract: a record that is
    unparseable after it still fails closed, because an unparseable log cannot fence."""
    store = tmp_path / "store"
    _fake_oplog(store, "w.log", ["P 1@rewards.bin", "X bogus record"])
    journal = tmp_path / "slot"
    journal.mkdir()
    (journal / OPLOG_OFFSETS_FILE).write_text(json.dumps({"w.log": len("P 1@rewards.bin\n") + 2}))
    with pytest.raises(UnrecoverableJournalError, match="malformed op-log record"):
        _read_oplog_tail(store, journal)


# ------------------------------------------- 7b/8. slot reach, ordering, selection


def _two_slot_ring(tq, tmp_path, partition_id) -> tuple[Path, list[Path]]:
    """A ring holding two complete slots, newest first."""
    root = tmp_path / "ring"
    writer = JournalWriter(root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
    finish_group(tq, partition_id, "slotA", n=2, value=1)
    older = Path(writer.tick()["dir"])
    finish_group(tq, partition_id, "slotB", n=2, value=2)
    newer = Path(writer.tick()["dir"])
    writer.stop()
    assert older != newer
    return root, [newer, older]


def test_slots_are_ranked_by_tick_sequence_not_by_the_clock(tq_session, tmp_path, partition_id):
    """Slot selection was a pure wall-clock decision, and the recommended wiring
    recorded no step at all, so a backwards clock step chose the older slot. The
    persisted sequence orders them without a clock; ``completed_at`` stays the fallback
    for stamps written before it existed."""
    root, (newer, older) = _two_slot_ring(tq_session, tmp_path, partition_id)
    assert complete_journal_slots(root) == [newer, older]
    assert latest_journal(root) == newer

    # a clock that ran backwards during the newest tick must not demote it
    stamp = json.loads((newer / STAMP_FILE).read_text())
    assert stamp["tick_seq"] > json.loads((older / STAMP_FILE).read_text())["tick_seq"]
    stamp["completed_at"] = 1.0
    (newer / STAMP_FILE).write_text(json.dumps(stamp))
    assert latest_journal(root) == newer, "the sequence, not the timestamp, decides"

    # a stamp with no sequence ranks below one that has it
    old_stamp = json.loads((older / STAMP_FILE).read_text())
    old_stamp.pop("tick_seq")
    old_stamp["completed_at"] = time.time() + 3600
    (older / STAMP_FILE).write_text(json.dumps(old_stamp))
    assert latest_journal(root) == newer


def test_the_tick_sequence_continues_across_a_writer_restart(tq_session, tmp_path, partition_id):
    """It is persisted, not in-memory: a restarted driver continues above the highest
    sequence in the ring, so ordering survives the crash this package exists for."""
    root, (newer, _older) = _two_slot_ring(tq_session, tmp_path, partition_id)
    highest = json.loads((newer / STAMP_FILE).read_text())["tick_seq"]
    restarted = JournalWriter(root, interval_s=0.0, ring=2)
    try:
        assert restarted._tick_seq == highest + 1
        finish_group(tq_session, partition_id, "slotC", n=2, value=3)
        info = restarted.tick()
        assert info["tick_seq"] == highest + 1
        assert latest_journal(root) == Path(info["dir"])
    finally:
        restarted.stop()


def test_a_poisoned_newest_slot_falls_back_to_the_next_older_one(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    """The ring held history no consumer could reach: ``latest_journal`` returned one
    slot and the tool had no fallback, so one corrupt slot made the whole journal
    unusable and the operator's only recovery was to move it out of the ring by hand,
    mid-outage. The fallback recovers from the older slot and reports why."""
    tq = tq_session
    root, (newer, older) = _two_slot_ring(tq, tmp_path, partition_id)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    (newer / OPLOG_OFFSETS_FILE).write_text(json.dumps({"w.log": -1}))  # torn fence on the newest slot
    tq.close()

    out = restore(store=store_root, journal_root=root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert out["status"] == "dry_run", out
    assert out["source_journal"] == str(older), "the older complete slot must be used"
    assert out["source_selection"] == "journal"
    assert [r["slot"] for r in out["source_slot_rejections"]] == [str(newer)]
    assert out["source_slot_rejections"][0]["status"] == "unrecoverable_journal"
    assert out["journal_slots"] == [str(newer), str(older)]

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_journal_slot_selects_one_slot_and_reports_a_bad_name(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """``--journal-slot`` turns the undocumented mid-outage ``mv`` into a flag, with the
    same validation automatic selection applies (so it cannot reach a slot automatic
    selection would refuse to select), and a name the ring does not hold is a controlled
    status rather than a silent fallback to the newest."""
    tq = tq_session
    root, (newer, older) = _two_slot_ring(tq, tmp_path, partition_id)
    ckpt_dir, _ = _make_ckpt(tq, tmp_path, step=5)
    tq.close()

    assert select_journal_slot(root, older.name) == older
    assert select_journal_slot(root, "journal-777") is None

    pinned = restore(
        store=store_root,
        journal_root=root,
        ckpt_dir=ckpt_dir,
        partition=partition_id,
        dry_run=True,
        journal_slot=older.name,
    )
    assert pinned["status"] == "dry_run", pinned
    assert pinned["source_journal"] == str(older)
    assert pinned["journal_slot_request"] == older.name
    assert pinned["journal_slots"] == [str(older)], "an explicit selection considers that slot only"

    default = restore(store=store_root, journal_root=root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert default["source_journal"] == str(newer)

    missing = restore(
        store=store_root,
        journal_root=root,
        ckpt_dir=ckpt_dir,
        partition=partition_id,
        dry_run=True,
        journal_slot="journal-777",
    )
    assert missing["status"] == "journal_slot_not_found", missing
    assert missing["source_journal"] is None

    import transfer_queue as tq2

    tq2.init(tq_conf)


# --------------------------------------- 8. trainer step by default, loader signal


class _FakeTrainer:
    """Enough of a verl trainer for discovery: a step and a dataloader attribute."""

    def __init__(self, root):
        self.global_steps = 41
        self.train_dataloader = None
        self.root = root

    def start(self):
        from verl.experimental.durable_buffer import register

        # called from trainer code, exactly as the README's recommended wiring is
        return register.start_journal(str(self.root), interval_s=3600.0)


def test_start_journal_records_the_trainer_step_by_default(tmp_path):
    """The recommended wiring passed no ``metadata_fn``, so its slots carried no step
    and BOTH the slot ranking and the journal-versus-checkpoint freshness comparison
    fell back to wall clock: the recommended path was the weakest one. The step is now
    recorded whenever the calling trainer can be identified, and it is re-read per tick
    so it tracks progress."""
    from verl.experimental.durable_buffer import register

    trainer = _FakeTrainer(tmp_path / "ring")
    writer = trainer.start()
    try:
        assert writer.metadata_fn is not None
        assert writer.metadata_fn() == {"global_steps": 41}
        trainer.global_steps = 42
        assert writer.metadata_fn() == {"global_steps": 42}, "the step is read per tick, not captured once"
    finally:
        register.stop_journal()
    assert not (tmp_path / "ring" / WRITER_LEASE_FILE).exists(), "stop_journal releases the ring lease"


def test_an_explicit_metadata_fn_still_wins(tmp_path):
    from verl.experimental.durable_buffer import register

    try:
        writer = register.start_journal(
            str(tmp_path / "ring"), interval_s=3600.0, metadata_fn=lambda: {"global_steps": 7, "mine": True}
        )
        assert writer.metadata_fn() == {"global_steps": 7, "mine": True}
    finally:
        register.stop_journal()


def test_no_discoverable_trainer_is_reported_rather_than_silent(tmp_path, caplog):
    """A run whose journal records no step is not wrong, only weaker, so it is reported
    instead of being left for someone to infer from a freshness tie-break later."""
    from verl.experimental.durable_buffer import register

    try:
        with caplog.at_level(logging.WARNING):
            writer = register.start_journal(str(tmp_path / "ring"), interval_s=3600.0)
        assert writer.metadata_fn is None
        assert "found no trainer to read global_steps from" in caplog.text
    finally:
        register.stop_journal()


def test_a_caller_that_is_not_a_trainer_is_not_mistaken_for_one(tmp_path):
    """Discovery is narrow on purpose: an integer ``global_steps`` AND a
    ``train_dataloader`` together identify a verl trainer and nothing else. Probed at
    depth 1 so the frame under inspection is the calling method's, which is the frame
    ``start_journal`` inspects at the default depth."""
    from verl.experimental.durable_buffer.register import _discover_trainer

    class NotATrainer:
        global_steps = 3  # no train_dataloader

        def probe(self):
            return _discover_trainer(1)

    class AlsoNot:
        train_dataloader = object()  # no global_steps

        def probe(self):
            return _discover_trainer(1)

    class Both:
        global_steps = 5
        train_dataloader = object()

        def probe(self):
            return _discover_trainer(1)

    assert NotATrainer().probe() is None
    assert AlsoNot().probe() is None
    found = Both().probe()
    assert isinstance(found, Both), "a caller carrying both attributes IS the trainer"


def test_summary_reports_whether_the_loader_position_is_journaled(tmp_path):
    """``dataloader_journaled`` reaches the trainer's metrics under the existing
    ``durable_buffer/`` prefix, so a broken coupling is visible on a dashboard before
    the crash instead of in the postmortem."""
    writer = JournalWriter(tmp_path / "ring", interval_s=0.0, ring=2)
    try:
        assert writer.summary()["dataloader_journaled"] is False
        writer.dataloader_fn = lambda: {"pos": 1}
        assert writer.summary()["dataloader_journaled"] is True
    finally:
        writer.stop()


class _Loader:
    def state_dict(self):
        return {"pos": 3}


def _buffer_kwargs(**durable):
    return dict(
        trainer_mode="separate_async",
        trainer_config=OmegaConf.create({}),
        max_off_policy_threshold=10**6,
        max_off_policy_strategy="drop",
        sampler_kwargs=OmegaConf.create({"durable_buffer": durable}),
        poll_interval=0.05,
    )


def test_an_unwired_dataloader_warns_once_and_shows_up_in_the_metrics(tmp_path, caplog, monkeypatch):
    """``_wire_dataloader`` is best-effort through ``refill_fn.__self__`` and did nothing
    at all on failure — retried silently forever. An upstream refactor therefore turned
    loader journaling off with no signal, and the only symptom was duplicate generation
    after the next crash. It is reported at the moment it breaks, once (the sync runs per
    poll), and it clears as soon as a loader is attached."""
    from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

    monkeypatch.setattr(ReplayBufferAsync, "_sync_metadata_from_transfer_queue", lambda self: None)
    rb = DurableReplayBuffer(refill_fn=None, **_buffer_kwargs(journal_root=str(tmp_path / "ring"), background=False))
    try:
        assert rb._journal is not None and rb._dataloader_wired is False
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                rb._sync_metadata_from_transfer_queue()
        assert caplog.text.count("could not be discovered through refill_fn") == 1, "warn ONCE, not per poll"
        assert rb._journal.summary()["dataloader_journaled"] is False

        rb.attach_dataloader(_Loader())
        assert rb._journal.summary()["dataloader_journaled"] is True
    finally:
        rb._journal.stop()


def test_a_wired_dataloader_never_warns(tmp_path, caplog, monkeypatch):
    from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

    monkeypatch.setattr(ReplayBufferAsync, "_sync_metadata_from_transfer_queue", lambda self: None)

    class Trainer:
        train_dataloader = _Loader()

        def refill(self, n):  # bound method -> __self__ is this object
            return n

    trainer = Trainer()
    rb = DurableReplayBuffer(
        refill_fn=trainer.refill, **_buffer_kwargs(journal_root=str(tmp_path / "ring"), background=False)
    )
    try:
        assert rb._dataloader_wired is True
        with caplog.at_level(logging.WARNING):
            rb._sync_metadata_from_transfer_queue()
        assert "could not be discovered" not in caplog.text
        assert rb._journal.summary()["dataloader_journaled"] is True
    finally:
        rb._journal.stop()
