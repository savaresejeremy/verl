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
"""Regressions for the round-1 CR review findings (crash-window and safety hardening).

Each test pins one specific failure mode from the review:

* intent logging — a mutation is ALWAYS covered by the op log, even when the process
  dies between the log append and the publish/unlink;
* journal tick coherence — offsets are captured before the controller snapshot
  (mutations landing during the snapshot get fenced, the safe direction) and the
  dataloader state is captured before the TQ cut (duplication, never loss);
* tick serialization — the poll-driven backstop cannot overlap the background thread;
* store identity — a journal cannot destructively reconcile a store it was not taken
  against;
* store-wide DROP — consuming another partition's samples after the journal must drop
  that partition's metadata on restore, not just the requested partition's;
* session-aware promotion — multiple outputs from one session cannot impersonate a
  complete group;
* restore tool — an older/unordered journal never overwrites a fresher checkpoint, and
  a crash mid-publication is recoverable;
* resource/paths hygiene — op-log handles close, tmp names are collision-free, hostile
  field names cannot escape the store root or corrupt the op log.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import threading
import time
import warnings
from pathlib import Path

import pytest
import torch
from helpers import (
    consume_group,
    crash_and_restart,
    finish_group,
    indexes_of,
    make_fields,
    objects,
    submit_prompt,
    tag_prompt_finished,
)

from verl.experimental.durable_buffer import StoreMismatchError, fsck, recover
from verl.experimental.durable_buffer import restore_tool as restore_tool_mod
from verl.experimental.durable_buffer.fs_kv import (
    OPLOG_DIR,
    OPLOG_OFFSETS_FILE,
    STORE_ID_FILE,
    DurableFSClient,
    _dumps,
)
from verl.experimental.durable_buffer.fsck import _parse_object_name, _read_oplog_tail, promote_complete_groups
from verl.experimental.durable_buffer.journal import DATALOADER_FILE, JournalWriter
from verl.experimental.durable_buffer.restore_tool import restore

# --------------------------------------------------------------------------- comment 1


def test_put_intent_logs_before_publish(tmp_path, monkeypatch):
    """The op-log record must be durable BEFORE the object is published: a crash in
    between leaves a logged-but-unpublished mutation (fsck resolves it as a safe DROP),
    never a published-but-unlogged one (which op-log fencing could not see)."""
    client = DurableFSClient({"root": str(tmp_path / "store"), "serializer": "tq"})

    real_replace = os.replace

    def die_on_publish(src, dst):  # simulate death between log append and publish
        raise RuntimeError("killed before publish")

    monkeypatch.setattr(os, "replace", die_on_publish)
    with pytest.raises(RuntimeError, match="killed before publish"):
        client.put(["5@rewards"], [torch.tensor([1.0])])
    monkeypatch.setattr(os, "replace", real_replace)

    logs = list((client.root / OPLOG_DIR).glob("*.log"))
    assert logs, "no op log written"
    records = logs[0].read_text()
    assert "P 5@rewards.bin" in records, "the intent record must precede the publish"
    assert not list(client.root.glob("*.bin")), "object must not exist (publish was killed)"
    assert not list(client.root.glob("*.tmp*")), "failed publish must clean its tmp file"
    client.close()


def test_logged_but_unpublished_record_is_fenced_and_dropped(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """A crashed writer's intent record without a published object must resolve to
    fence+DROP (wasteful, safe) — the metadata never points at bytes fsck did not see."""
    tq = tq_session
    submit_prompt(tq, partition_id, "solid", status="running")
    keys = finish_group(tq, partition_id, "solid", n=2, value=6)
    tag_prompt_finished(tq, partition_id, "solid")
    victim_idx = indexes_of(tq, partition_id, [keys[0]])[0]
    journal = tmp_path / "journal"
    time.sleep(0.05)
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1, "fence_ts": time.time()})

    # a writer intent-logged a rewrite of the live sample's object, then died before
    # publishing: the record is in the log tail, the store still holds the OLD bytes.
    phantom_log = store_root / OPLOG_DIR / "99999-dead.log"
    phantom_log.write_text(f"P {victim_idx}@rewards.bin\n")

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, journal, partition_id)

    assert report["fenced_objects"] >= 1, report
    live = tq.kv_list().get(partition_id, {})
    assert keys[0] not in live, "a sample whose object may hold unknown bytes must be dropped"
    assert not (store_root / f"{victim_idx}@rewards.bin").exists(), "the suspect object must be fenced"
    phantom_log.unlink()  # keep the shared store's logs clean for later tests


def test_journal_offsets_are_captured_before_the_snapshot(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """Mutations landing between the offset capture and the controller pickle must be
    REPLAYED (and fenced) on restore — early offsets over-fence, which is the safe
    direction. Late offsets would let a delete+rewrite pair escape the fence."""
    tq = tq_session
    submit_prompt(tq, partition_id, "early", status="running")
    early_keys = finish_group(tq, partition_id, "early", n=2, value=21)
    tag_prompt_finished(tq, partition_id, "early")

    def sizes_now():
        d = store_root / OPLOG_DIR
        return {p.name: p.stat().st_size for p in d.glob("*.log")}

    pre_sizes = sizes_now()

    mutated = {}

    def in_window_mutation():
        # runs inside the tick, AFTER the offset pre-capture and BEFORE tq.save_checkpoint:
        # consume 'early' and write 'sneaky' (TQ reuses the freed indexes).
        consume_group(tq, partition_id, "early", n=2)
        finish_group(tq, partition_id, "sneaky", n=2, value=99)
        mutated["done"] = True
        return {"global_steps": 1}

    jw = JournalWriter(tmp_path / "journals", interval_s=0.0, ring=2, metadata_fn=in_window_mutation)
    info = jw.tick()
    assert info is not None and mutated.get("done")

    journal_dir = tmp_path / "journals" / "journal-000"
    recorded = json.loads((journal_dir / OPLOG_OFFSETS_FILE).read_text())
    assert recorded == pre_sizes, "journal offsets must be the sizes from BEFORE the tick body ran"
    # the in-window mutation appended to some log past the recorded offset
    assert any(size > recorded.get(name, 0) for name, size in sizes_now().items()), (
        "expected in-window records past the fence"
    )

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, journal_dir, partition_id)
    live = tq.kv_list().get(partition_id, {})
    assert not any(k.startswith("sneaky") for k in live), "in-snapshot-window writes must be fenced"
    assert report["fenced_objects"] >= 4, report  # sneaky's 2 trajectories x 2 fields
    assert not any(k in live for k in early_keys), "consumed-in-window samples must be dropped, not served"


# --------------------------------------------------------------------------- comment 2


def _write_controller_state(path) -> None:
    """The controller pickle a real ``tq.save_checkpoint`` leaves in a slot.

    Stubs that replace ``save_checkpoint`` to observe ordering still have to produce it:
    the tick reads it back to fingerprint the snapshot's schema against what
    reconciliation reads (``fsck.snapshot_schema_fingerprint_of``).
    """
    (Path(path) / "controller_state.pkl").write_bytes(pickle.dumps({"partitions": {}}))


def test_tick_bodies_are_serialized_never_overlapped(tq_session, tmp_path, monkeypatch):
    import transfer_queue as tq_mod

    entered = threading.Event()
    release = threading.Event()

    def slow_save_checkpoint(path, **kwargs):
        os.makedirs(path, exist_ok=True)
        # A stub standing in for save_checkpoint has to leave the artifact a real one
        # leaves: the tick reads the controller pickle back to fingerprint the snapshot's
        # schema, and a slot without one is not selectable anyway. Empty partitions are
        # enough, and they exercise the nothing-to-observe fallback.
        _write_controller_state(path)
        entered.set()
        assert release.wait(timeout=30)

    monkeypatch.setattr(tq_mod, "save_checkpoint", slow_save_checkpoint)
    jw = JournalWriter(tmp_path / "journals", interval_s=0.0, ring=3)

    results = []
    t = threading.Thread(target=lambda: results.append(jw.tick()))
    t.start()
    assert entered.wait(timeout=30)
    assert jw.tick() is None, "a tick requested while one runs must be skipped, not queued"
    assert jw.stats["skipped_busy"] == 1
    assert jw.maybe_tick() is None
    release.set()
    t.join(timeout=30)

    assert results and results[0] is not None
    assert jw.stats["ticks"] == 1
    dirs = [p for p in (tmp_path / "journals").iterdir() if p.is_dir()]
    assert len(dirs) == 1, f"overlapping ticks wrote {len(dirs)} snapshots"


def test_tick_captures_dataloader_state_before_the_tq_snapshot(tq_session, tmp_path, monkeypatch):
    """The tick must capture the loader state FIRST: that is the duplication-not-loss
    direction (loader-behind re-serves already-buffered prompts, benign), whereas the
    reverse ordering silently loses every prompt fetched in between. This ordering is
    NOT loss-free — the residual, documented window is a dispatch straddling the
    entire tick (fetched before the loader capture, registered after the TQ cut); see
    the JournalWriter docstring and the README guarantees."""
    import transfer_queue as tq_mod

    order = []

    def fake_save_checkpoint(path, **kwargs):
        order.append("tq_snapshot")
        os.makedirs(path, exist_ok=True)
        _write_controller_state(path)  # see slow_save_checkpoint above

    def dataloader_state():
        order.append("dataloader")
        return {"pos": 7}

    monkeypatch.setattr(tq_mod, "save_checkpoint", fake_save_checkpoint)
    jw = JournalWriter(tmp_path / "journals", interval_s=0.0, ring=2, dataloader_fn=dataloader_state)
    assert jw.tick() is not None
    assert order == ["dataloader", "tq_snapshot"], order
    saved = torch.load(tmp_path / "journals" / "journal-000" / DATALOADER_FILE, weights_only=False)
    assert saved == {"pos": 7}


# --------------------------------------------------------------------------- comment 3


def test_mismatched_journal_aborts_before_any_mutation(tq_session, store_root, partition_id, tmp_path):
    tq = tq_session
    finish_group(tq, partition_id, "bound", n=2, value=1)
    journal = tmp_path / "journal"
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1})

    # a DIFFERENT store with its own identity and contents
    foreign = tmp_path / "foreign_store"
    foreign.mkdir()
    (foreign / STORE_ID_FILE).write_text("f" * 32)
    (foreign / "0@rewards.bin").write_bytes(b"precious foreign bytes")
    live_before = dict(tq.kv_list().get(partition_id, {}))

    with pytest.raises(StoreMismatchError):
        fsck(tq, foreign, journal, partition_id)

    assert (foreign / "0@rewards.bin").read_bytes() == b"precious foreign bytes"
    assert sorted(p.name for p in foreign.iterdir()) == ["0@rewards.bin", STORE_ID_FILE]
    assert dict(tq.kv_list().get(partition_id, {})) == live_before, "metadata must be untouched on abort"

    # the identity escape hatch does NOT exempt the fence artifacts: a foreign store
    # cannot satisfy the journal's op-log delta, so the default mode still aborts ...
    from verl.experimental.durable_buffer import UnrecoverableJournalError

    with pytest.raises(UnrecoverableJournalError):
        fsck(tq, foreign, journal, partition_id, dry_run=True, force_store_mismatch=True)
    # ... and the EXPLICIT legacy mtime flow is the remaining (labeled) escape
    report = fsck(tq, foreign, journal, partition_id, mode="mtime", dry_run=True, force_store_mismatch=True)
    assert report["mode"] == "mtime"


# --------------------------------------------------------------------------- comment 4


def test_val_metadata_dropped_when_val_objects_vanish_during_train_restore(tq_session, tq_conf, store_root, tmp_path):
    tq = tq_session
    train_p, val_p = "train-dropx", "val-dropx"
    finish_group(tq, train_p, "tkeep", n=2, value=1)
    finish_group(tq, val_p, "vkeep", n=2, value=2)
    finish_group(tq, val_p, "vgone", n=2, value=3)
    journal = tmp_path / "journal"
    time.sleep(0.05)
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1, "fence_ts": time.time()})
    # consumed in ANOTHER partition after the snapshot
    tq.kv_clear(keys=["vgone_0_0", "vgone_1_0"], partition_id=val_p)

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, journal, train_p)

    live_val = tq.kv_list().get(val_p, {})
    assert not any(k.startswith("vgone") for k in live_val), (
        f"val metadata for consumed objects survived a train restore: {report}"
    )
    assert {"vkeep_0_0", "vkeep_1_0"} <= set(live_val), "untouched val samples must survive"
    assert {"tkeep_0_0", "tkeep_1_0"} <= set(tq.kv_list().get(train_p, {}))
    assert report["dropped_by_partition"].get(val_p) == 2, report


# --------------------------------------------------------------------------- comment 5


def test_multiple_outputs_from_one_session_do_not_fake_completeness(tq_session, store_root, partition_id):
    tq = tq_session
    # 4 successful outputs, ALL from session 0 -> NOT a complete group of 4 sessions
    submit_prompt(tq, partition_id, "fakegrp", status="running")
    tq.kv_batch_put(
        keys=[f"fakegrp_0_{i}" for i in range(4)],
        partition_id=partition_id,
        fields=make_fields(4, 5),
        tags=[{"status": "success", "global_steps": 1}] * 4,
    )
    # a genuine group: sessions 0..3, one output each
    submit_prompt(tq, partition_id, "realgrp", status="running")
    finish_group(tq, partition_id, "realgrp", n=4, value=6)

    report = promote_complete_groups(tq, partition_id, group_size=4, assume_uniform_group_size=True)
    assert "fakegrp" not in report["promoted"], report
    assert "fakegrp" in report["partial_left_for_reissue"], report
    assert "realgrp" in report["promoted"], report
    live = tq.kv_list().get(partition_id, {})
    assert live["fakegrp"]["status"] == "running"
    assert live["realgrp"]["status"] == "finished"


# --------------------------------------------------------------------------- comment 6


def _make_ckpt(tq, tmp_path, step: int):
    ckpt_dir = tmp_path / "ckpts"
    step_dir = ckpt_dir / f"global_step_{step}"
    step_dir.mkdir(parents=True)
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text(str(step))
    torch.save({"pos": f"step{step}"}, step_dir / "data.pt")
    tq.save_checkpoint(str(step_dir / "transfer_queue"), metadata={"global_steps": step})
    return ckpt_dir, step_dir


def _dir_snapshot(d):
    return sorted((p.name, p.stat().st_size) for p in d.rglob("*") if p.is_file())


def test_restore_tool_never_applies_an_older_journal(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    finish_group(tq, partition_id, "fresh", n=2, value=4)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)

    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 3})
    assert jw.tick() is not None  # an OLDER journal (step 3 < checkpoint step 5)

    ckpt_before = _dir_snapshot(step_dir)
    store_before = objects(store_root)
    tq.close()

    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "checkpoint_fresher", result
    assert result["freshness"]["journal_step"] == 3 and result["freshness"]["ckpt_step"] == 5
    assert _dir_snapshot(step_dir) == ckpt_before, "an older journal must not touch the checkpoint"
    assert objects(store_root) == store_before, "an older journal must not touch the store"
    assert not (step_dir / "transfer_queue.pre_reconcile").exists()

    # explicit override still available
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, force=True)
    assert result["status"] == "reconciled", result

    import transfer_queue as tq2

    tq2.init(tq_conf)  # leave the session fixture usable


def test_restore_tool_treats_unordered_journals_as_ambiguous(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=2)
    journal_root = tmp_path / "journals"
    JournalWriter(journal_root, interval_s=0.0, ring=2).tick()
    journal_dir = journal_root / "journal-000"
    # strip every freshness marker (no steps, no fence_ts anywhere) while KEEPING the
    # store identity + oplog record intact — identity-less journals are rejected
    # earlier, as store_mismatch, which is a different regression
    (journal_dir / "metadata.json").write_text(json.dumps({"storage_saved": True, "user_metadata": {}}))
    fence = json.loads((journal_dir / "fence.json").read_text())
    del fence["fence_ts"]
    (journal_dir / "fence.json").write_text(json.dumps(fence))

    ckpt_before = _dir_snapshot(step_dir)
    tq.close()
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "ordering_ambiguous", result
    assert _dir_snapshot(step_dir) == ckpt_before

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_restore_tool_prefers_a_strictly_newer_journal(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    finish_group(tq, partition_id, "afterckpt", n=2, value=8)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
    assert jw.tick() is not None
    tq.close()

    result = restore(
        store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, group_size=2
    )
    assert result["status"] == "reconciled", result
    assert result["source_step"] == 6
    assert (step_dir / "transfer_queue.pre_reconcile").exists(), "the original state must be kept as a backup"

    import transfer_queue as tq2

    tq2.init(tq_conf)
    tq2.load_checkpoint(str(step_dir / "transfer_queue"))
    live = tq2.kv_list().get(partition_id, {})
    assert {"afterckpt_0_0", "afterckpt_1_0"} <= set(live)


# --------------------------------------------------------------------------- comment 7


def test_restore_tool_publication_crash_is_recoverable(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    finish_group(tq, partition_id, "pubcrash", n=2, value=9)
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

    backup = step_dir / "transfer_queue.pre_reconcile"
    assert backup.exists(), "the original state must survive a crash mid-publication"
    assert not (step_dir / "transfer_queue").exists()
    # round 5: the staging dir carries the transaction id in its name
    assert list(step_dir.glob("transfer_queue.reconciled.*")), "the staged replacement must survive the crash"

    # a re-run heals the interrupted publication and completes
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "reconciled", result
    assert (step_dir / "transfer_queue" / "controller_state.pkl").exists()

    import transfer_queue as tq2

    tq2.init(tq_conf)
    tq2.load_checkpoint(str(step_dir / "transfer_queue"))
    assert {"pubcrash_0_0", "pubcrash_1_0"} <= set(tq2.kv_list().get(partition_id, {}))


# --------------------------------------------------------------------------- comment 8


def test_oplog_handle_closes_without_resourcewarning(tmp_path):
    """The op-log handle must be closed by a finalizer when the client is GC'd.

    Recording (rather than raising) is the reliable in-process check: a warning
    raised inside ``__del__`` is swallowed by the interpreter, so ``-W error`` would
    pass vacuously.
    """

    def cycle():
        client = DurableFSClient({"root": str(tmp_path / "rw_store"), "serializer": "tq"})
        client.put(["1@rewards"], [torch.tensor([1.0])])
        del client

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cycle()
        gc.collect()
    leaks = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert not leaks, f"op-log handle leaked: {[str(w.message) for w in leaks]}"

    # explicit close is idempotent
    client = DurableFSClient({"root": str(tmp_path / "rw_store"), "serializer": "tq"})
    client.put(["2@rewards"], [torch.tensor([2.0])])
    client.close()
    client.close()


# --------------------------------------------------------------------------- comment 9


def test_concurrent_same_key_publishes_use_unique_tmp_names(tmp_path):
    root = tmp_path / "store"
    a = DurableFSClient({"root": str(root), "serializer": "tq"})
    b = DurableFSClient({"root": str(root), "serializer": "tq"})
    path = a._path("3@rewards")
    tmp_a, tmp_b = a._tmp_path(path), b._tmp_path(path)
    assert tmp_a != tmp_b, "two clients writing the same key must never share a tmp file"
    assert tmp_a != a._tmp_path(path), "tmp names must be unique per call"

    # the round-1 failure interleaving: A stages its tmp, B completes a full put of the
    # same key, then A publishes. With a shared tmp name A's rename raised (or published
    # B's bytes); with unique tmps this is plain last-writer-wins.
    tmp_a.write_bytes(_dumps(torch.tensor([1.0]), "tq"))
    b.put(["3@rewards"], [torch.tensor([2.0])])
    os.replace(tmp_a, path)  # A's publish, after B's complete put

    reader = DurableFSClient({"root": str(root), "serializer": "tq"})
    assert torch.equal(reader.get(["3@rewards"])[0], torch.tensor([1.0]))
    assert not list(root.glob("*.tmp*")), "no tmp residue"
    for c in (a, b, reader):
        c.close()


# --------------------------------------------------------------------------- comment 10


HOSTILE_FIELDS = [
    "a/b",
    "../traversal",
    "a@b",
    "with space",
    "new\nline",
    ".hidden",
    "tab\there",
    "percent%41",
    # Unicode/vertical separators str.splitlines() splits on but literal-'\n' parsing
    # must NOT (legal names; a splitlines() parser corrupted their records):
    "nel\x85nel",
    "ls\u2028ls",
    "ps\u2029ps",
    "vt\x0bvt",
    "ff\x0cff",
]


def test_hostile_field_names_round_trip_and_never_escape_the_root(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    outside_before = sorted(p.name for p in tmp_path.iterdir())
    client = DurableFSClient({"root": str(root), "serializer": "tq"})

    keys = [f"7@{f}" for f in HOSTILE_FIELDS]
    values = [torch.tensor([float(i)]) for i in range(len(keys))]
    client.put(keys, values)

    assert sorted(p.name for p in tmp_path.iterdir()) == outside_before, "a hostile field name escaped the store root"
    created = [p for p in root.rglob("*") if p.is_file() and p.name.endswith(".bin")]
    assert len(created) == len(keys)
    for p in created:
        assert p.parent == root, f"object landed outside the flat store root: {p}"
        assert not p.name.startswith("."), "encoded names must not enter the dot namespace"
        assert "\n" not in p.name and "\t" not in p.name

    for key, value in zip(keys, values, strict=True):
        assert torch.equal(client.get([key])[0], value)

    # the op log (one line per record) must parse back to exactly these objects
    fake_journal = tmp_path / "fake_journal"
    fake_journal.mkdir()
    (fake_journal / OPLOG_OFFSETS_FILE).write_text("{}")
    written, deleted = _read_oplog_tail(root, fake_journal)
    parsed = {_parse_object_name(n) for n in written}
    assert parsed == {(7, f) for f in HOSTILE_FIELDS}, parsed
    assert not deleted

    # index-scoped delete must find every encoded object through the field registry
    assert client.clear_by_index([7]) == len(keys)
    assert not list(root.glob("*.bin"))
    client.close()
