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
"""Regressions for the round-6 CR verdict (round 7 of hardening).

Each section pins one verdict comment:

1.  store-INTERNAL symlinks are refused or fenced, never followed — a symlinked
    ``_oplog`` dir refuses writes (``StoreIntegrityError``) and fails recovery
    closed; a symlinked canonical object refuses ``get`` and is FENCED by fsck
    (metadata dropped, the link's target untouched); a symlinked op-log file
    aborts recovery BEFORE any controller/store mutation. Outside sentinels are
    proven byte-identical in every case (no external read served, no external
    write landed).
2.  containment does not stop at ``global_step_N`` — symlinked live/backup
    ``transfer_queue`` dirs, ``data.pt`` halves, marker/lock files are refused as
    a controlled ``invalid_checkpoint_layout`` result (stable no-plan schema)
    BEFORE any lock or mutation, in both selection paths (pointer + scan) and
    both modes; symlinked journal-ring slots are never selected; symlinked
    journal artifacts fail direct ``recover()`` and both restore modes closed.

The residual check-then-use TOCTOU is a documented limitation (the store lives on
trusted paths; the threat model is misconfiguration, not a racing local attacker).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from helpers import crash_and_restart, finish_group, indexes_of
from test_round5_on_cpu import assert_plan_unavailable
from test_round6_on_cpu import _make_ckpt, _tree_bytes

from verl.experimental.durable_buffer import StoreIntegrityError, UnrecoverableJournalError, recover
from verl.experimental.durable_buffer.fs_kv import FENCE_FILE, OPLOG_DIR, DurableFSClient, backend_config
from verl.experimental.durable_buffer.journal import JournalWriter, latest_journal
from verl.experimental.durable_buffer.restore_tool import RESTORE_LOCK_FILE, restore


@pytest.fixture
def private_store(tq_session, tq_conf, tmp_path):
    """A throwaway store on a FRESH TQ (these tests corrupt store internals, which
    must never poison the session store). Yields (tq module, store, conf); always
    hands the session TQ back."""
    import transfer_queue as tq

    tq_session.close()
    store = tmp_path / "store"
    conf = backend_config(store)
    tq.init(conf)
    try:
        yield tq, store, conf
    finally:
        try:
            tq.close()
        except Exception:  # noqa: BLE001 - fixture restoration must always run
            pass
        import transfer_queue as tq2

        tq2.init(tq_conf)


def _journaled_run(tq, tmp_path, partition_id, uid: str):
    """One finished group + a checkpoint + one journal tick (the standard corpus)."""
    finish_group(tq, partition_id, uid, n=2, value=3)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
    info = jw.tick()
    assert info is not None
    return ckpt_dir, step_dir, journal_root, Path(info["dir"])


# --------------------- comment 1: store-internal symlinks refused or fenced


def test_symlinked_oplog_dir_refuses_writes_and_fails_recovery_closed(private_store, partition_id, tmp_path):
    """A symlinked ``_oplog`` used to redirect every intent-log append — the only
    recovery fence source — outside the store, and recovery replayed the redirected
    logs as if they were the store's own. Writers must refuse (put and the offsets
    pre-capture), recovery must abort BEFORE any controller/store mutation, and the
    restore tool must surface the same refusal as a controlled result: the external
    target byte-identical throughout (nothing written through the link)."""
    tq, store, conf = private_store
    ckpt_dir, _step_dir, journal_root, journal = _journaled_run(tq, tmp_path, partition_id, "so")

    outside = tmp_path / "outside_oplog"
    (store / OPLOG_DIR).rename(outside)  # same bytes, now OUTSIDE the store...
    (store / OPLOG_DIR).symlink_to(outside)  # ...reached only through the link
    outside_before = _tree_bytes(outside)

    # recovery: fail closed before tq.load_checkpoint / any store mutation
    tq = crash_and_restart(conf)
    store_before = _tree_bytes(store)
    with pytest.raises(UnrecoverableJournalError, match="not a real"):
        recover(tq, store, journal, partition_id)
    assert tq.kv_list().get(partition_id, {}) == {}, "controller mutated by a refused recovery"
    assert _tree_bytes(store) == store_before, "store mutated by a refused recovery"
    tq.close()

    # writers: put refuses at the intent log (before publishing anything), and the
    # journal's offsets pre-capture refuses rather than fencing against foreign logs
    client = DurableFSClient({"root": str(store), "serializer": "tq"})
    with pytest.raises(StoreIntegrityError):
        client.put(["1@x"], [torch.zeros(2)])
    with pytest.raises(StoreIntegrityError):
        client.oplog_offsets()
    assert not (store / "1@x.bin").exists(), "a refused put must publish nothing"

    # restore tool: the same refusal as a CONTROLLED result, zero writes, both modes
    before_store, before_ckpt = _tree_bytes(store), _tree_bytes(ckpt_dir)
    for dry in (True, False):
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        assert result["status"] == "unrecoverable_journal", (dry, result)
        assert _tree_bytes(store) == before_store, f"dry={dry}: store mutated by a refused restore"
        assert _tree_bytes(ckpt_dir) == before_ckpt, f"dry={dry}: checkpoint mutated by a refused restore"
    assert _tree_bytes(outside) == outside_before, "something reached THROUGH the symlinked _oplog"


def test_symlinked_object_refuses_get_and_is_fenced_by_fsck(private_store, partition_id, tmp_path):
    """A symlink wearing a canonical object name used to (a) make ``get``
    deserialize bytes from OUTSIDE the store under a trusted key and (b) be counted
    by fsck as the valid expected object. ``get`` must refuse with a clear error
    (lstat before any read), and fsck must FENCE the entry — removing the link
    itself, never its target — and drop the now-missing sample's metadata."""
    tq, store, conf = private_store
    finish_group(tq, partition_id, "keep", n=2, value=1)
    finish_group(tq, partition_id, "vic", n=2, value=2)
    vic_idx = indexes_of(tq, partition_id, ["vic_0_0"])[0]  # pick the victim BY SAMPLE, not by listing order
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
    info = jw.tick()
    assert info is not None

    victim_name = f"{vic_idx}@response_ids.bin"  # a journal-expected object
    assert (store / victim_name).is_file(), "test corpus must contain the victim object"
    outside = tmp_path / "outside_object.bin"
    outside.write_bytes(b"FOREIGN BYTES: must never be served or deleted")
    (store / victim_name).unlink()
    (store / victim_name).symlink_to(outside)
    outside_before = outside.read_bytes()

    client = DurableFSClient({"root": str(store), "serializer": "tq"})
    with pytest.raises(StoreIntegrityError):
        client.get([victim_name[: -len(".bin")]])
    with pytest.raises(StoreIntegrityError):  # a symlinked destination is corruption, not an overwrite
        client.put([victim_name[: -len(".bin")]], [torch.zeros(2)])
    assert outside.read_bytes() == outside_before

    tq = crash_and_restart(conf)
    report = recover(tq, store, Path(info["dir"]), partition_id)
    assert report["fenced_objects"] >= 1, report
    assert not (store / victim_name).is_symlink(), "the fenced entry (the link itself) must be gone"
    assert outside.read_bytes() == outside_before, "fencing must remove the link, NEVER its target"
    live = tq.kv_list().get(partition_id, {})
    assert "keep_0_0" in live and "keep_1_0" in live, f"undamaged data must survive: {sorted(live)}"
    assert "vic_0_0" not in live, f"the sample the link impersonated must be dropped: {sorted(live)}"


def test_symlinked_oplog_file_fails_recovery_closed(private_store, partition_id, tmp_path):
    """A symlinked ``*.log`` used to be opened and replayed as the writer's own log.
    Recovery must abort BEFORE any controller/store mutation even when the linked
    content is byte-identical to the real log (linkness alone is disqualifying),
    and the restore tool must report it as a controlled result in both modes."""
    tq, store, conf = private_store
    ckpt_dir, _step_dir, journal_root, journal = _journaled_run(tq, tmp_path, partition_id, "sl")

    log_name = sorted(p.name for p in (store / OPLOG_DIR).glob("*.log"))[0]
    outside = tmp_path / "outside_log.log"
    (store / OPLOG_DIR / log_name).rename(outside)  # identical content, external location
    (store / OPLOG_DIR / log_name).symlink_to(outside)
    outside_before = outside.read_bytes()

    tq = crash_and_restart(conf)
    store_before = _tree_bytes(store)
    with pytest.raises(UnrecoverableJournalError, match="regular non-symlink"):
        recover(tq, store, journal, partition_id)
    assert tq.kv_list().get(partition_id, {}) == {}, "controller mutated by a refused recovery"
    assert _tree_bytes(store) == store_before, "store mutated by a refused recovery"
    tq.close()

    before_ckpt = _tree_bytes(ckpt_dir)
    for dry in (True, False):
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        assert result["status"] == "unrecoverable_journal", (dry, result)
        assert _tree_bytes(store) == store_before, f"dry={dry}: store mutated by a refused restore"
        assert _tree_bytes(ckpt_dir) == before_ckpt, f"dry={dry}: checkpoint mutated by a refused restore"
    assert outside.read_bytes() == outside_before, "the external log target must never be touched"


# ------------- comment 2: containment below global_step_N (checkpoint + journal)


def _plant_outside_tq(tmp_path: Path, name: str) -> Path:
    """A plausible TQ-snapshot dir OUTSIDE every root the tool may touch."""
    outside = tmp_path / name
    outside.mkdir(parents=True)
    (outside / "metadata.json").write_text(json.dumps({"user_metadata": {"global_steps": 1}}))
    (outside / "sentinel.bin").write_bytes(b"MUST SURVIVE EVERY RESTORE ATTEMPT")
    return outside


@pytest.mark.parametrize("selection", ["pointer", "scan"])
def test_symlinked_live_tq_dir_is_refused_before_any_lock(private_store, partition_id, tmp_path, selection):
    """The round-6 containment stopped at the step dir: its ``transfer_queue`` child
    was consumed unvalidated, so a symlink there sent freshness probes, heals and
    swaps through to an external tree. Both selection paths (pointer + fallback
    scan) must now yield a controlled ``invalid_checkpoint_layout`` (stable no-plan
    schema) BEFORE any lock or mutation, in both modes."""
    tq, store, conf = private_store
    ckpt_dir, step_dir, journal_root, _journal = _journaled_run(tq, tmp_path, partition_id, "lt")
    tq.close()
    if selection == "scan":
        (ckpt_dir / "latest_checkpointed_iteration.txt").unlink()

    outside = tmp_path / "outside_live_tq"
    (step_dir / "transfer_queue").rename(outside)  # real snapshot, external location
    (step_dir / "transfer_queue").symlink_to(outside)

    before_outside, before_ckpt, before_store = _tree_bytes(outside), _tree_bytes(ckpt_dir), _tree_bytes(store)
    for dry in (True, False):
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        assert_plan_unavailable(result, status="invalid_checkpoint_layout", dry_run=dry)
        assert result["checkpoint_step_dir"] == str(step_dir), (dry, result)
        assert "symlink" in result["detail"], (dry, result)
        assert _tree_bytes(outside) == before_outside, f"dry={dry}: the restore reached through the symlink"
        assert _tree_bytes(ckpt_dir) == before_ckpt, f"dry={dry}: checkpoint mutated (lock/marker?) despite refusal"
        assert _tree_bytes(store) == before_store, f"dry={dry}: store mutated despite refusal"
        assert not (step_dir / RESTORE_LOCK_FILE).exists(), f"dry={dry}: refusal must precede the lock"


def test_symlinked_backup_tq_dir_blocks_the_heal(private_store, partition_id, tmp_path):
    """Healing renames the ``*.pre_reconcile`` backup into place; a symlinked backup
    would install a redirect as the live snapshot. With the live dir missing (the
    exact heal trigger) and a symlinked backup, both modes must refuse with
    ``invalid_checkpoint_layout`` and leave everything byte-identical."""
    tq, store, conf = private_store
    ckpt_dir, step_dir, journal_root, _journal = _journaled_run(tq, tmp_path, partition_id, "hb")
    tq.close()

    outside = tmp_path / "outside_backup_tq"
    (step_dir / "transfer_queue").rename(outside)  # live dir gone: a heal would run...
    (step_dir / "transfer_queue.pre_reconcile").symlink_to(outside)  # ...from a symlinked backup

    before_outside, before_ckpt = _tree_bytes(outside), _tree_bytes(ckpt_dir)
    for dry in (True, False):
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        assert_plan_unavailable(result, status="invalid_checkpoint_layout", dry_run=dry)
        assert _tree_bytes(outside) == before_outside, f"dry={dry}: the heal reached through the symlink"
        assert _tree_bytes(ckpt_dir) == before_ckpt, f"dry={dry}: checkpoint mutated despite refusal"
        assert (step_dir / "transfer_queue.pre_reconcile").is_symlink(), f"dry={dry}: the hostile backup was consumed"
        assert not (step_dir / "transfer_queue").exists(), f"dry={dry}: something was renamed into place"


@pytest.mark.parametrize(
    "victim_name",
    ["data.pt", "data.pt.pre_reconcile", "reconcile.state", "reconcile.lock"],
)
def test_symlinked_step_dir_files_are_refused(private_store, partition_id, tmp_path, victim_name):
    """Every step-dir file the tool reads, copies onto or replaces must be a real
    regular file: a symlinked ``data.pt`` makes the pre-reconcile backup copy READ
    foreign bytes, a symlinked ``data.pt.pre_reconcile`` makes ``copy2`` WRITE
    through to the target, and symlinked marker/lock files redirect reads. Each is
    a controlled ``invalid_checkpoint_layout`` in both modes, external target
    byte-identical."""
    tq, store, conf = private_store
    ckpt_dir, step_dir, journal_root, _journal = _journaled_run(tq, tmp_path, partition_id, "sf")
    tq.close()

    outside = tmp_path / "outside_file.bin"
    outside.write_bytes(b"EXTERNAL FILE: must never be read as ours nor written through")
    victim = step_dir / victim_name
    if victim.exists():
        victim.unlink()
    victim.symlink_to(outside)
    outside_before = outside.read_bytes()

    before_ckpt = _tree_bytes(ckpt_dir)
    for dry in (True, False):
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        assert_plan_unavailable(result, status="invalid_checkpoint_layout", dry_run=dry)
        assert victim_name in result["detail"], (dry, result)
        assert outside.read_bytes() == outside_before, f"dry={dry}: the tool went through the {victim_name} symlink"
        assert _tree_bytes(ckpt_dir) == before_ckpt, f"dry={dry}: checkpoint mutated despite refusal"


def test_symlinked_journal_slot_is_never_selected(private_store, partition_id, tmp_path):
    """``latest_journal`` used to follow a symlinked ``journal-*`` dir, so a link to
    an external (or recycled) tree with the newest stamp would win selection. The
    hostile slot must be skipped — the real (older) slot is selected and applied —
    and the link's target stays byte-identical."""
    import shutil

    tq, store, conf = private_store
    ckpt_dir, _step_dir, journal_root, journal = _journaled_run(tq, tmp_path, partition_id, "js")
    tq.close()

    # an external journal-shaped tree with a STRICTLY newer stamp and step
    outside = tmp_path / "outside_journal"
    shutil.copytree(journal, outside)
    stamp = json.loads((outside / "journal_stamp.json").read_text())
    stamp["completed_at"] = stamp["completed_at"] + 10_000
    (outside / "journal_stamp.json").write_text(json.dumps(stamp))
    md = json.loads((outside / "metadata.json").read_text())
    md.setdefault("user_metadata", {})["global_steps"] = 99
    (outside / "metadata.json").write_text(json.dumps(md))
    hostile = journal_root / "journal-001"
    assert not hostile.exists(), "the ring fixture must leave slot 001 free"
    hostile.symlink_to(outside)

    assert latest_journal(journal_root) == journal, "the symlinked slot must never win selection"

    before_outside = _tree_bytes(outside)
    for dry in (True, False):
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        assert result["status"] == ("dry_run" if dry else "reconciled"), (dry, result)
        assert result["source_journal"] == str(journal), f"dry={dry}: the REAL slot must drive the restore: {result}"
        assert _tree_bytes(outside) == before_outside, f"dry={dry}: the restore consumed the hostile slot's target"


def test_symlinked_journal_artifact_fails_closed_everywhere(private_store, partition_id, tmp_path):
    """Recovery-critical artifact files inside a selected snapshot must be regular
    non-symlink files: a symlinked ``fence.json`` (byte-identical content — linkness
    alone disqualifies) aborts direct ``recover()`` BEFORE any controller or store
    mutation and surfaces as a controlled ``unrecoverable_journal`` in both restore
    modes, trees byte-identical."""
    tq, store, conf = private_store
    ckpt_dir, _step_dir, journal_root, journal = _journaled_run(tq, tmp_path, partition_id, "ja")

    outside = tmp_path / "outside_fence.json"
    (journal / FENCE_FILE).rename(outside)  # identical bytes, external location
    (journal / FENCE_FILE).symlink_to(outside)
    outside_before = outside.read_bytes()

    tq = crash_and_restart(conf)
    store_before = _tree_bytes(store)
    with pytest.raises(UnrecoverableJournalError, match="regular non-symlink"):
        recover(tq, store, journal, partition_id)
    assert tq.kv_list().get(partition_id, {}) == {}, "controller mutated by a refused recovery"
    assert _tree_bytes(store) == store_before, "store mutated by a refused recovery"
    tq.close()

    before_ckpt = _tree_bytes(ckpt_dir)
    for dry in (True, False):
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        assert result["status"] == "unrecoverable_journal", (dry, result)
        assert _tree_bytes(store) == store_before, f"dry={dry}: store mutated by a refused restore"
        assert _tree_bytes(ckpt_dir) == before_ckpt, f"dry={dry}: checkpoint mutated by a refused restore"
    assert outside.read_bytes() == outside_before, "the artifact's external target must never be touched"
