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
"""Regressions for the round-7 CR verdict (round 8 of hardening).

The last three symlink-gate readers, each proven with a VALID external sentinel
(bytes that would PASS every downstream check if the link were followed — the
dangerous acceptance, not just a garbage file):

1.  the ``_store_id`` lineage authority — ``read_store_id``, the ``_ensure_store_id``
    concurrent-create fallback, and ``_fields.d`` children are lstat-gated; a
    symlinked stamp holding the journal's EXACT identity is refused by every
    reader (bootstrap, client attach, direct ``recover()``, both restore modes)
    before any mutation, and never overridable.
2.  interrupted-transaction completion — the marker's recorded source path must be
    PROVEN to live under the allowed journal/checkpoint roots before any
    identity/fence/dataloader read; a symlink-swapped slot (digest-identical
    target!) or an escaping/traversal path yields an explicit ``discard_marker``
    plan entry, identical in dry and real mode, with the checkpoint's data intact.
3.  the remaining direct readers — a symlinked ``latest_checkpointed_iteration.txt``
    is ignored (scan wins), exported ``load_journal`` refuses symlinked
    controller/metadata artifacts, the fence probe treats a symlinked stamp as
    absent (identity checks fail closed), and the offline audit manifest reader
    returns "unusable".

The residual check-then-use TOCTOU is a documented limitation (trusted FSx/NFS
paths; the threat model is misconfiguration, not a racing local attacker).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from helpers import crash_and_restart, finish_group
from test_round6_on_cpu import _make_ckpt, _tree_bytes

from verl.experimental.durable_buffer import (
    StoreIntegrityError,
    StoreMismatchError,
    UnrecoverableJournalError,
    load_journal,
    recover,
)
from verl.experimental.durable_buffer import restore_tool as restore_tool_mod
from verl.experimental.durable_buffer.fs_kv import (
    FENCE_FILE,
    FIELDS_REGISTRY,
    MANIFEST_FILE,
    STORE_ID_FILE,
    DurableFSClient,
    backend_config,
    bootstrap_durable_fs,
    read_store_id,
)
from verl.experimental.durable_buffer.fsck import _read_fence_metadata, _read_manifest, check_store_identity
from verl.experimental.durable_buffer.journal import JournalWriter
from verl.experimental.durable_buffer.restore_tool import (
    LATEST_FILE,
    RECONCILE_STATE_FILE,
    _marker_source_disposition,
    find_ckpt_tq_dir,
    restore,
)


@pytest.fixture
def private_store(tq_session, tq_conf, tmp_path):
    """A throwaway store on a FRESH TQ (these tests corrupt store internals and
    journal/marker state, which must never poison the session store). Yields
    (tq module, store, conf); always hands the session TQ back."""
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


def _crash_hook(name):
    def crash():
        raise RuntimeError(f"killed at {name}")

    return crash


def _interrupted_txn(tq, store, tmp_path, partition_id, hook_name: str):
    """A REAL interrupted publication: corpus + ckpt(step 5) + journal(step 6), then
    a restore that crashes at the named hook, leaving the marker behind."""
    finish_group(tq, partition_id, "txn", n=2, value=9)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": "journal6"},
        metadata_fn=lambda: {"global_steps": 6},
    )
    info = jw.tick()
    assert info is not None
    tq.close()
    original = getattr(restore_tool_mod, hook_name)
    setattr(restore_tool_mod, hook_name, _crash_hook(hook_name))
    try:
        with pytest.raises(RuntimeError, match=hook_name):
            restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        setattr(restore_tool_mod, hook_name, original)
    return ckpt_dir, step_dir, journal_root, Path(info["dir"])


# --------------- comment 1: the _store_id lineage authority and _fields.d children


def test_symlinked_store_id_is_refused_by_every_identity_reader(tmp_path):
    """The sentinel is the store's OWN identity, moved outside and reached only
    through a link — if any reader followed it, every identity check would PASS.
    ``read_store_id``, the ``_ensure_store_id`` concurrent-create fallback (both
    bootstrap flavors) and the attach-only client must all refuse instead, without
    rewriting the stamp or touching the external file."""
    store = tmp_path / "store"
    DurableFSClient({"root": str(store), "serializer": "tq"})  # creates + stamps
    outside = tmp_path / "outside_id"
    (store / STORE_ID_FILE).rename(outside)  # the REAL id, now external
    (store / STORE_ID_FILE).symlink_to(outside)
    before = outside.read_bytes()

    with pytest.raises(StoreIntegrityError, match="store identity stamp"):
        read_store_id(store)
    with pytest.raises(StoreIntegrityError, match="store identity stamp"):
        DurableFSClient({"root": str(store), "serializer": "tq"})  # create=True: _ensure fallback
    with pytest.raises(StoreIntegrityError, match="store identity stamp"):
        DurableFSClient({"root": str(store), "serializer": "tq", "create": False})  # attach-only
    with pytest.raises(StoreIntegrityError, match="store identity stamp"):
        bootstrap_durable_fs(backend_config(store))  # create=True bootstrap
    with pytest.raises(StoreIntegrityError, match="store identity stamp"):
        bootstrap_durable_fs(backend_config(store, create=False))  # attach-only bootstrap

    assert outside.read_bytes() == before, "the external identity file must never be touched"
    assert (store / STORE_ID_FILE).is_symlink(), "a refused reader must not rewrite the stamp"


def test_symlinked_store_id_aborts_recovery_and_both_restore_modes(private_store, partition_id, tmp_path):
    """The scariest acceptance: the external sentinel holds the journal's EXACT
    store id, so following the link would pass the identity preflight and unlock
    destructive reconciliation. Direct ``recover()`` must abort with a
    never-overridable ``StoreMismatchError`` BEFORE any controller/store mutation,
    and both restore modes must report the controlled ``store_mismatch`` with
    byte-identical trees."""
    tq, store, conf = private_store
    ckpt_dir, _step_dir, journal_root, journal = _journaled_run(tq, tmp_path, partition_id, "sid")
    tq = crash_and_restart(conf)  # restart FIRST: init would (rightly) refuse the corrupted stamp

    real_id = json.loads((journal / FENCE_FILE).read_text())["store_id"]
    outside = tmp_path / "outside_store_id"
    (store / STORE_ID_FILE).rename(outside)
    (store / STORE_ID_FILE).symlink_to(outside)
    assert outside.read_text().strip() == real_id, "the sentinel must be the MATCHING identity"
    outside_before = outside.read_bytes()

    store_before = _tree_bytes(store)
    with pytest.raises(StoreMismatchError, match="cannot be trusted"):
        recover(tq, store, journal, partition_id)
    assert tq.kv_list().get(partition_id, {}) == {}, "controller mutated by a refused recovery"
    assert _tree_bytes(store) == store_before, "store mutated by a refused recovery"
    tq.close()

    before_ckpt = _tree_bytes(ckpt_dir)
    for dry in (True, False):
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        assert result["status"] == "store_mismatch", (dry, result)
        assert "cannot be trusted" in result["detail"], (dry, result)
        assert _tree_bytes(store) == store_before, f"dry={dry}: store mutated by a refused restore"
        assert _tree_bytes(ckpt_dir) == before_ckpt, f"dry={dry}: checkpoint mutated by a refused restore"
    # never overridable: --force-store-mismatch vouches for a KNOWN lineage, and a
    # redirected stamp has none to vouch for
    result = restore(
        store=store,
        journal_root=journal_root,
        ckpt_dir=ckpt_dir,
        partition=partition_id,
        dry_run=True,
        force_store_mismatch=True,
    )
    assert result["status"] == "store_mismatch", result
    assert outside.read_bytes() == outside_before, "something went THROUGH the identity symlink"


@pytest.mark.parametrize("child_kind", ["symlink", "directory"])
def test_non_regular_fields_registry_children_are_refused(tmp_path, child_kind):
    """``_known_fields`` used to count ANY ``_fields.d`` child as a field marker.
    A symlink or directory there proves tampering (and silently skipping could hide
    a REPLACED legitimate marker, leaking that field's objects past every clear),
    so the registry read — and the whole-sample delete built on it — must refuse
    with ``StoreIntegrityError`` before logging or unlinking anything."""
    store = tmp_path / "store"
    client = DurableFSClient({"root": str(store), "serializer": "tq"})
    client.put(["1@x"], [torch.zeros(2)])
    outside = tmp_path / "outside_marker"
    outside.write_bytes(b"EXTERNAL: must never be counted as a field marker")
    hostile = store / FIELDS_REGISTRY / "evil"
    if child_kind == "symlink":
        hostile.symlink_to(outside)
    else:
        hostile.mkdir()
    before = outside.read_bytes()

    fresh = DurableFSClient({"root": str(store), "serializer": "tq"})  # empty field cache
    with pytest.raises(StoreIntegrityError, match="field marker"):
        fresh._known_fields()
    with pytest.raises(StoreIntegrityError, match="field marker"):
        fresh.clear_by_index([1])
    assert (store / "1@x.bin").is_file(), "a refused clear must delete nothing"
    assert outside.read_bytes() == before, "the marker symlink's target must never be touched"


# ------------------ comment 2: marker-recorded source provenance before completion


def test_marker_source_disposition_matrix(tmp_path):
    """The provenance verdicts the completion path is built on: contained real dirs
    are ``ok``, a contained-but-absent slot is ``missing`` (recycled ring slot),
    and everything unprovable — escapes, traversals, the root itself, symlinked
    slots or PARENT components, sources with no allowed root — is ``invalid``."""
    ring = tmp_path / "ring"
    (ring / "journal-000").mkdir(parents=True)
    step = tmp_path / "ckpts" / "global_step_5"
    (step / "transfer_queue").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    assert _marker_source_disposition(str(ring / "journal-000"), ring, step) == ("ok", None)
    assert _marker_source_disposition(str(step / "transfer_queue"), ring, step) == ("ok", None)
    assert _marker_source_disposition(str(ring / "journal-001"), ring, step) == ("missing", None)
    assert _marker_source_disposition(str(outside), ring, step)[0] == "invalid"
    assert _marker_source_disposition(str(step / ".." / "x"), ring, step)[0] == "invalid"
    assert _marker_source_disposition(str(ring), ring, step)[0] == "invalid"  # the root itself
    (ring / "journal-002").symlink_to(outside)
    assert _marker_source_disposition(str(ring / "journal-002"), ring, step)[0] == "invalid"
    link_parent = ring / "linkdir"
    link_parent.symlink_to(outside)  # symlinked PARENT of the recorded source
    assert _marker_source_disposition(str(link_parent / "child"), ring, step)[0] == "invalid"
    assert _marker_source_disposition(str(ring / "journal-000"), None, step)[0] == "invalid"  # no ring root


@pytest.mark.parametrize(
    ("hook", "marker_state", "expected_data"),
    [
        ("_POST_TQ_PUBLISH_CRASH_HOOK", "BEGIN", {"pos": "step5"}),
        ("_PRE_DONE_CRASH_HOOK", "TQ_PUBLISHED", {"pos": "journal6"}),
    ],
)
def test_symlink_swapped_source_slot_declines_completion(
    private_store, partition_id, tmp_path, hook, marker_state, expected_data
):
    """Crash mid-publication, then swap the marker-recorded journal slot for a
    symlink to a DIGEST-IDENTICAL external copy: if completion followed the link,
    every check (identity digest, lineage) would pass and the BEGIN case would copy
    the external dataloader state into ``data.pt``. It must instead DECLINE with an
    explicit ``discard_marker`` plan entry (identical in dry and real mode),
    leaving ``data.pt`` and the external tree byte-identical."""
    tq, store, _conf = private_store
    ckpt_dir, step_dir, journal_root, slot = _interrupted_txn(tq, store, tmp_path, partition_id, hook)
    marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert marker["state"] == marker_state
    assert torch.load(step_dir / "data.pt", weights_only=False) == expected_data

    outside = tmp_path / "outside_slot"
    shutil.copytree(slot, outside)  # digest-identical: following WOULD complete
    shutil.rmtree(slot)
    slot.symlink_to(outside)
    outside_before = _tree_bytes(outside)

    dry = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert dry["status"] == "dry_run", dry
    plan_entry = dry["would_complete_transaction"]
    assert plan_entry["action"] == "discard_marker", plan_entry
    assert plan_entry.get("unprovable_source") is True, plan_entry
    assert "not a real (non-symlink) directory" in plan_entry["reason"], plan_entry
    assert torch.load(step_dir / "data.pt", weights_only=False) == expected_data, "dry run mutated data.pt"
    assert _tree_bytes(outside) == outside_before, "the dry run consumed the hostile slot's target"

    real = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert real["status"] == "reconciled", real
    assert real["discarded_transaction"] == plan_entry, "dry and real must decide identically"
    assert "completed_interrupted_transaction" not in real, "an unprovable source must never complete"
    assert torch.load(step_dir / "data.pt", weights_only=False) == expected_data, "checkpoint data.pt must stay intact"
    assert _tree_bytes(outside) == outside_before, "the real run consumed the hostile slot's target"
    final = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert final["state"] == "DONE" and final["txn"] != marker["txn"], "a FRESH transaction must own the end state"


@pytest.mark.parametrize("shape", ["absolute_outside", "traversal"])
def test_escaping_marker_source_declines_completion(private_store, partition_id, tmp_path, shape):
    """A marker whose recorded source ESCAPES the allowed roots (absolute external
    path, or a ``..`` traversal spelled under the step dir) — with a digest-identical
    tree waiting at the target, so following it would complete. The plan must record
    an explicit ``discard_marker`` (unprovable provenance), dry and real identically,
    and the checkpoint's ``data.pt`` must stay intact."""
    tq, store, _conf = private_store
    ckpt_dir, step_dir, journal_root, slot = _interrupted_txn(
        tq, store, tmp_path, partition_id, "_POST_TQ_PUBLISH_CRASH_HOOK"
    )
    if shape == "absolute_outside":
        outside = tmp_path / "outside_abs"
        hostile_source = str(outside)
    else:
        outside = ckpt_dir / "outside_copy"  # resolves from step_dir/../outside_copy
        hostile_source = str(step_dir / ".." / "outside_copy")
    shutil.copytree(slot, outside)  # digest-identical: following WOULD complete
    shutil.rmtree(slot)  # ring recycled: nothing legitimate left to reconcile from

    marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert marker["state"] == "BEGIN"
    marker["source_journal"] = hostile_source
    (step_dir / RECONCILE_STATE_FILE).write_text(json.dumps(marker))
    outside_before = _tree_bytes(outside)

    dry = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    plan_entry = dry["would_complete_transaction"]
    assert plan_entry["action"] == "discard_marker", plan_entry
    assert plan_entry.get("unprovable_source") is True, plan_entry
    assert "allowed root" in plan_entry["reason"], plan_entry

    real = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert real["status"] == "reconciled", real
    assert real["discarded_transaction"] == plan_entry, "dry and real must decide identically"
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "step5"}, "data.pt must stay intact"
    assert _tree_bytes(outside) == outside_before, "the escaping source's target was consumed"
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE"


def test_begin_crash_with_symlinked_slot_reads_nothing(private_store, partition_id, tmp_path):
    """BEGIN marker with NO visible publication (crash before the swap), slot
    swapped for a symlink: the discard already fires at the txn proof — and nothing
    may be read through the marker's source on the way out."""
    tq, store, _conf = private_store
    ckpt_dir, step_dir, journal_root, slot = _interrupted_txn(
        tq, store, tmp_path, partition_id, "_POST_BEGIN_CRASH_HOOK"
    )
    marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert marker["state"] == "BEGIN"

    outside = tmp_path / "outside_slot"
    shutil.copytree(slot, outside)
    shutil.rmtree(slot)
    slot.symlink_to(outside)
    outside_before = _tree_bytes(outside)

    for dry in (True, False):
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        entry = result["would_complete_transaction"] if dry else result.get("discarded_transaction")
        assert entry is not None and entry["action"] == "discard_marker", (dry, result)
        assert "cannot be proven to be THIS transaction's publication" in entry["reason"], (dry, entry)
        assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "step5"}, f"dry={dry}"
        assert _tree_bytes(outside) == outside_before, f"dry={dry}: the hostile slot's target was consumed"
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE"


# --------------------- comment 3: the remaining direct readers, path by path


def test_symlinked_latest_pointer_is_ignored_not_followed(tmp_path):
    """The pointer file steers which step dir the lock/heal/swap machinery operates
    on. A symlinked pointer (its external target naming a REAL, selectable step —
    following would pick ``global_step_3``) must be IGNORED per the documented
    fail-closed pointer semantics: the scan wins and selects the highest step."""
    ckpt_dir = tmp_path / "ckpts"
    for step in (3, 5):
        (ckpt_dir / f"global_step_{step}").mkdir(parents=True)
    outside = tmp_path / "outside_pointer.txt"
    outside.write_text("3")
    (ckpt_dir / LATEST_FILE).symlink_to(outside)

    found = find_ckpt_tq_dir(ckpt_dir)
    assert found is not None
    step_dir, _ckpt_tq = found
    assert step_dir.name == "global_step_5", "the scan, not the symlinked pointer, must drive selection"
    assert outside.read_text() == "3", "the pointer's external target must never be touched"


@pytest.mark.parametrize("artifact", ["controller_state.pkl", "metadata.json"])
def test_symlinked_journal_artifact_fails_exported_load_journal_closed(tq_session, partition_id, tmp_path, artifact):
    """``load_journal`` is exported and reachable OUTSIDE ``recover()``'s preflight,
    so its own reads must be gated: a symlinked controller pickle or metadata file
    (byte-identical content — linkness alone disqualifies) raises
    ``UnrecoverableJournalError`` instead of being read through."""
    tq = tq_session
    finish_group(tq, partition_id, "lj", n=2, value=4)
    journal = tmp_path / "journal"
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1})
    assert load_journal(journal, partition_id).live, "the corpus must load before corruption"

    outside = tmp_path / f"outside_{artifact}"
    (journal / artifact).rename(outside)  # identical bytes, external location
    (journal / artifact).symlink_to(outside)
    before = outside.read_bytes()
    with pytest.raises(UnrecoverableJournalError, match="regular non-symlink"):
        load_journal(journal, partition_id)
    assert outside.read_bytes() == before, "the artifact's external target must never be touched"


def test_symlinked_fence_stamp_never_donates_identity(tq_session, store_root, partition_id, tmp_path):
    """The fence probe feeds the journal side of the identity check. The sentinel is
    the REAL fence payload — its ``store_id`` MATCHES the store, so following the
    link would pass identity. The probe must treat the symlinked stamp as absent
    (``{}``) and the identity check must fail closed, force or no force."""
    tq = tq_session
    finish_group(tq, partition_id, "fs", n=2, value=4)
    journal = tmp_path / "journal"
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1})
    assert json.loads((journal / FENCE_FILE).read_text())["store_id"] == read_store_id(store_root)

    outside = tmp_path / "outside_fence.json"
    (journal / FENCE_FILE).rename(outside)
    (journal / FENCE_FILE).symlink_to(outside)
    before = outside.read_bytes()

    assert _read_fence_metadata(journal) == {}, "a symlinked stamp must read as absent"
    with pytest.raises(StoreMismatchError, match="no valid store identity"):
        check_store_identity(store_root, journal)
    with pytest.raises(StoreMismatchError):  # never overridable: no valid identity to vouch for
        check_store_identity(store_root, journal, force=True)
    assert outside.read_bytes() == before, "the stamp's external target must never be touched"


def test_symlinked_audit_manifest_is_unusable_not_read(tmp_path):
    """The offline audit reader must reject a symlinked manifest exactly like a
    malformed one (``None`` = unusable) — an audit must not report rows read from
    outside the snapshot dir."""
    store = tmp_path / "mstore"
    client = DurableFSClient({"root": str(store), "serializer": "tq"})
    client.put(["1@x"], [torch.zeros(2)])
    snap = tmp_path / "snap"
    client.write_manifest(snap, full=True)
    baseline = _read_manifest(snap)
    assert baseline is not None and "1@x.bin" in baseline, baseline

    outside = tmp_path / "outside_manifest.tsv"
    (snap / MANIFEST_FILE).rename(outside)  # identical rows, external location
    (snap / MANIFEST_FILE).symlink_to(outside)
    before = outside.read_bytes()
    assert _read_manifest(snap) is None, "a symlinked manifest must be unusable"
    assert outside.read_bytes() == before, "the manifest's external target must never be touched"
