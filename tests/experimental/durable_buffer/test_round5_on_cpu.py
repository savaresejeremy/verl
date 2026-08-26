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
"""Regressions for the round-4 CR verdict (round 5 of hardening).

Each section pins one verdict comment:

1.  op-log parsing fails CLOSED — strict UTF-8, and every record name is validated
    as a canonical encoded backend-object basename BEFORE any tail is returned: a
    traversal record (``P ../outside.bin``), an absolute-path record and an
    invalid-UTF-8 record each abort recovery with an outside-the-store sentinel
    file proven byte-identical (the old ``errors="replace"`` + prefix-only check
    let such a record reach ``(store_root / name).unlink()``);
2.  plan-then-execute restore — ALL validation happens in the read-only preflight
    and the SAME plan drives dry-run and real mode: (a) a publish crash with no
    journal reconciles from the healable backup (the old code dropped the plan as
    ``nothing_to_restore``), (b) invalid offsets with a healable backup abort with
    ZERO mutation in BOTH modes (the old code healed/completed first), (c)
    ambiguous freshness returns the identical plan in both modes, byte-identically;
3.  lineage binding — a stale marker + published dir left by a DELETED store must
    not complete against a store recreated at the same path: the marker and the
    published dir carry the store lineage, cross-lineage transactions are
    discarded WITH a report and the foreign ``data.pt`` is never adopted;
4.  exclusive per-checkpoint lock — a held lock fails a second restorer fast
    (``locked``, holder identity reported, nothing written, the lock is never
    broken: conservative manual stale-lock policy), and a deterministic
    two-restorer interleaving proves the mutating phase cannot interleave;
5.  force_store_mismatch narrowed — via the PUBLIC restore(): missing/empty store
    identities return a controlled ``store_mismatch`` even WITH the force flag
    (the old code logged "forced, proceeding" and then crashed with an uncaught
    FileNotFoundError from the attach-only tq.init), while two valid-but-different
    identities remain forceable;
6.  strict ``__rollout_n__`` parsing — promotion accepts exactly one positive
    integer scalar: a boolean, a fractional value or a multi-element field leaves
    the prompt running (``skipped_unverifiable_n``) even under the operator's
    uniform-n assertion (the old flatten+[0]+int() coerced all three).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

import pytest
import torch
from helpers import (
    crash_and_restart,
    finish_group,
    submit_prompt,
    tag_prompt_finished,
)

from verl.experimental.durable_buffer import (
    UnrecoverableJournalError,
    recover,
)
from verl.experimental.durable_buffer import restore_tool as restore_tool_mod
from verl.experimental.durable_buffer.fs_kv import OPLOG_DIR, OPLOG_OFFSETS_FILE, backend_config
from verl.experimental.durable_buffer.fsck import promote_complete_groups
from verl.experimental.durable_buffer.journal import JournalWriter
from verl.experimental.durable_buffer.restore_tool import (
    RECONCILE_STATE_FILE,
    RESTORE_LOCK_FILE,
    restore,
)


def _tree_bytes(root: Path) -> dict[str, str]:
    """Relative path -> content hash for every file under root (byte-identity check)."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _make_ckpt(tq, tmp_path, step: int):
    ckpt_dir = tmp_path / "ckpts"
    step_dir = ckpt_dir / f"global_step_{step}"
    step_dir.mkdir(parents=True)
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text(str(step))
    torch.save({"pos": f"step{step}"}, step_dir / "data.pt")
    tq.save_checkpoint(str(step_dir / "transfer_queue"), metadata={"global_steps": step})
    return ckpt_dir, step_dir


def _crash_hook(name):
    def crash():
        raise RuntimeError(f"killed at {name}")

    return crash


# --------------------------------------- comment 1: op-log parsing fails closed


@pytest.mark.parametrize("record_kind", ["traversal", "absolute", "invalid_utf8", "over_encoded", "unsupported_escape"])
def test_hostile_oplog_record_aborts_before_any_mutation(tq_session, tq_conf, tmp_path, partition_id, record_kind):
    """A post-snapshot op-log record that names anything but a canonical store-object
    basename must abort recovery BEFORE any tail is acted on. The old fail-open path
    (errors="replace" + a 'P '/'D ' prefix check only) sent ``P ../outside.bin``
    straight into ``(store_root / name).unlink()`` — a delete OUTSIDE the store.
    Round 6 adds COMPLETED-record spellings the escape grammar alone accepted but the
    writer never emits: an over-encoding (``1@%41.bin`` aliases ``1@A.bin``) and an
    escape for a character the writer stores raw (``%7F``) — both must abort
    pre-mutation via the decode/re-encode byte-equality check."""
    import transfer_queue as tq

    tq_session.close()
    store = tmp_path / "store"
    conf = backend_config(store)
    tq.init(conf)
    try:
        submit_prompt(tq, partition_id, "vic", status="running")
        finish_group(tq, partition_id, "vic", n=2, value=7)
        tag_prompt_finished(tq, partition_id, "vic")
        journal = tmp_path / "journal"
        tq.save_checkpoint(str(journal), metadata={"global_steps": 1, "fence_ts": time.time()})

        # the sentinel lives OUTSIDE the store root: exactly what a traversal or an
        # absolute-path record would have deleted
        sentinel = tmp_path / "outside.bin"
        sentinel.write_bytes(b"MUST SURVIVE THE RECOVERY ATTEMPT")
        log = next((store / OPLOG_DIR).glob("*.log"))
        if record_kind == "traversal":
            with open(log, "a", encoding="utf-8") as fh:  # past the snapshot offsets
                fh.write("P ../outside.bin\n")
            match = "non-canonical"
        elif record_kind == "absolute":
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"P {sentinel}\n")
            match = "non-canonical"
        elif record_kind == "over_encoded":
            with open(log, "a", encoding="utf-8") as fh:  # writer emits '1@A.bin', never '%41'
                fh.write("P 1@%41.bin\n")
            match = "non-canonical"
        elif record_kind == "unsupported_escape":
            with open(log, "a", encoding="utf-8") as fh:  # \x7f is stored RAW, never escaped
                fh.write("P 1@%7F.bin\n")
            match = "non-canonical"
        else:  # invalid UTF-8 bytes inside a completed record
            with open(log, "ab") as fh:
                fh.write(b"P \xff\xfe@x.bin\n")
            match = "invalid UTF-8"

        tq = crash_and_restart(conf)
        store_before = _tree_bytes(store)
        with pytest.raises(UnrecoverableJournalError, match=match):
            recover(tq, store, journal, partition_id)

        assert sentinel.read_bytes() == b"MUST SURVIVE THE RECOVERY ATTEMPT", (
            f"{record_kind}: the recovery attempt reached outside the store"
        )
        assert _tree_bytes(store) == store_before, f"{record_kind}: store mutated by a refused recovery"
        assert tq.kv_list().get(partition_id, {}) == {}, f"{record_kind}: controller mutated by a refused recovery"
    finally:
        tq.close()
        tq.init(tq_conf)  # leave the session fixture usable


def test_canonical_escape_names_are_still_accepted_and_fenced(tq_session, tq_conf, tmp_path, partition_id):
    """The strict re-encode equality must not over-reject: the canonical spellings
    the writer actually emits — ``%25`` (a literal '%'), ``%2F`` (a '/'), a leading
    ``%2E`` (a dot-prefixed field) and raw hostile Unicode (U+2028, which is LEGAL
    and stays unescaped) — replay, recover cleanly and are fenced as post-snapshot
    writes."""
    import transfer_queue as tq

    tq_session.close()
    store = tmp_path / "store"
    conf = backend_config(store)
    tq.init(conf)
    try:
        submit_prompt(tq, partition_id, "vic", status="running")
        finish_group(tq, partition_id, "vic", n=2, value=7)
        tag_prompt_finished(tq, partition_id, "vic")
        journal = tmp_path / "journal"
        tq.save_checkpoint(str(journal), metadata={"global_steps": 1, "fence_ts": time.time()})

        # post-snapshot writes under every canonical special spelling: exactly what
        # the writer emits for fields '%', '/', '.hidden' and 'a\u2028b'
        canonical = ["999@%25.bin", "999@%2F.bin", "999@%2Ehidden.bin", "999@a\u2028b.bin"]
        log = next((store / OPLOG_DIR).glob("*.log"))
        with open(log, "a", encoding="utf-8") as fh:
            fh.writelines(f"P {name}\n" for name in canonical)
        for name in canonical:
            (store / name).write_bytes(b"post-snapshot bytes")

        tq = crash_and_restart(conf)
        report = recover(tq, store, journal, partition_id)  # must NOT abort
        assert report["fenced_objects"] >= len(canonical), report
        for name in canonical:
            assert not (store / name).exists(), f"{name!r}: a canonical post-snapshot object must be fenced"
        # the snapshotted data survived the reconcile
        live = tq.kv_list().get(partition_id, {})
        assert "vic" in live, live
    finally:
        tq.close()
        tq.init(tq_conf)  # leave the session fixture usable


# ------------------------- comment 2: preflight plan == dry plan == real execution

# The FULL stable plan schema (round-6 comment 5): every restore() result must carry
# every one of these keys, and dry vs real must agree on ALL of them — not on a
# hand-picked subset (the round-5 view compared four keys and missed that execution
# mutated plan["source"] after the backup rename).
_PLAN_SCHEMA_KEYS = (
    "checkpoint_step_dir",
    "plan_available",
    "detail",
    "pending_transaction",
    "would_heal_publication",
    "would_complete_transaction",
    "source_selection",
    "source_journal",
    "freshness",
    "tq_protocol_siblings",
)


def _plan_view(result: dict) -> dict:
    missing = [k for k in _PLAN_SCHEMA_KEYS if k not in result]
    assert not missing, f"stable plan schema violated: {missing} absent from {sorted(result)}"
    return {k: result[k] for k in _PLAN_SCHEMA_KEYS}


# The plan-carrying subset of _PLAN_SCHEMA_KEYS (restore_tool._PLAN_FIELDS): for a
# status that never ran a preflight these must be explicitly null, not merely absent.
_PLAN_ONLY_KEYS = (
    "pending_transaction",
    "would_heal_publication",
    "would_complete_transaction",
    "source_selection",
    "source_journal",
    "freshness",
    "tq_protocol_siblings",
)


def assert_plan_unavailable(result: dict, *, status: str, dry_run: bool) -> None:
    """Pin the COMPLETE stable result schema for statuses that never ran a preflight
    (``no_checkpoint``, ``locked``, ``invalid_checkpoint_layout``): every base key is
    present (``status``/``dry_run`` correct, ``detail`` a non-empty string),
    ``plan_available`` is strict-bool ``False``, and every plan field is explicitly
    ``None`` — so silently dropping ``plan_available: false`` or nulled plan fields
    from the schema can no longer stay green (round-6 verdict comment 5)."""
    view = _plan_view(result)  # asserts every _PLAN_SCHEMA_KEYS member is present
    assert result["status"] == status, result
    assert result["dry_run"] is dry_run, result
    assert view["plan_available"] is False, f"plan_available must be strict-bool False: {result}"
    assert isinstance(view["detail"], str) and view["detail"], f"detail must explain the refusal: {result}"
    for key in _PLAN_ONLY_KEYS:
        assert view[key] is None, f"{key} must be explicitly null when no preflight ran: {result}"


def test_publish_crash_with_no_journal_reconciles_from_the_healable_backup(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    """Checkpoint-only run (no journal) whose previous publication crashed mid-swap:
    the old code computed its probe from the healable backup but then returned
    ``nothing_to_restore``, dropping the plan. Now: the dry run reports the COMPLETE
    plan (heal + marker discard + checkpoint_backup source + full fsck counts) with
    byte-identical trees, and the real run executes exactly that plan."""
    tq = tq_session
    submit_prompt(tq, partition_id, "con", status="running")
    finish_group(tq, partition_id, "con", n=2, value=1)
    tag_prompt_finished(tq, partition_id, "con")
    submit_prompt(tq, partition_id, "keep", status="running")
    finish_group(tq, partition_id, "keep", n=2, value=2)
    tag_prompt_finished(tq, partition_id, "keep")
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    # post-checkpoint churn for the fsck plan: consumption (DROP) + production (FENCE)
    tq.kv_clear(keys=["con", "con_0_0", "con_1_0"], partition_id=partition_id)
    finish_group(tq, partition_id, "post", n=2, value=9)
    tq.close()

    original = restore_tool_mod._PUBLISH_CRASH_HOOK
    restore_tool_mod._PUBLISH_CRASH_HOOK = _crash_hook("mid-swap")
    try:
        with pytest.raises(RuntimeError, match="mid-swap"):
            restore(store=store_root, journal_root=None, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._PUBLISH_CRASH_HOOK = original
    assert not (step_dir / "transfer_queue").exists() and (step_dir / "transfer_queue.pre_reconcile").exists()

    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)
    dry = restore(store=store_root, journal_root=None, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert dry["status"] == "dry_run", dry
    assert dry["dry_run"] is True
    assert dry["would_heal_publication"] is True
    assert dry["pending_transaction"]["state"] == "BEGIN"
    assert dry["would_complete_transaction"]["action"] == "discard_marker"
    assert dry["source_selection"] == "checkpoint_backup", dry
    # the crashed run's fsck already fenced 'post' (store mutations precede the
    # publish), so the remaining plan is the DROP side: con's metadata still lives in
    # the backup snapshot while its objects (2 trajectories x 2 fields + payload) are gone
    assert dry["fsck"]["missing_objects"] >= 4, dry["fsck"]
    assert dry["fsck"]["dropped_samples"] >= 2, dry["fsck"]
    assert _tree_bytes(store_root) == before_store, "dry run mutated the store"
    assert _tree_bytes(ckpt_dir) == before_ckpt, "dry run mutated the checkpoint"

    real = restore(store=store_root, journal_root=None, ckpt_dir=ckpt_dir, partition=partition_id)
    assert real["status"] == "reconciled", real
    assert real["dry_run"] is False
    assert real["healed_publication"] is True
    assert real["discarded_transaction"]["action"] == "discard_marker"
    assert "completed_interrupted_transaction" not in real
    assert _plan_view(real) == _plan_view(dry), "the real run must execute the dry run's exact plan"
    for key in ("fenced_objects", "missing_objects", "dropped_samples", "prompts_dropped"):
        assert dry["fsck"][key] == real["fsck"][key], (key, dry["fsck"], real["fsck"])
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE"
    assert (step_dir / "transfer_queue").exists()

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_invalid_offsets_with_healable_backup_heals_but_refuses_the_journal(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    """The round-4 fail-open: invalid offsets were discovered only INSIDE recover(),
    after the real mode had already healed the backup and completed/discarded the
    marker — a mutation followed by ``unrecoverable_journal``. The offsets/tail
    validation now lives in the read-only preflight, so a refused journal reconciles
    NOTHING and completes no transaction, and ``--dry-run`` still writes nothing at all.

    The refusal is scoped to the JOURNAL, though. ``heal_publication`` means the
    checkpoint has no live ``transfer_queue`` dir, only this tool's own backup of it,
    and restoring that backup is a rename that depends on no journal artifact. Blocking
    it left the checkpoint in the one state verl resumes from silently and wrongly (a
    missing TQ dir is skipped without a warning, so the run starts with an empty buffer
    and every durable trajectory is orphaned). The real mode therefore heals and reports
    ``healed_only``, with the store still byte-identical, the marker untouched and a
    non-zero exit."""
    tq = tq_session
    finish_group(tq, partition_id, "off5", n=2, value=4)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
    info = jw.tick()
    assert info is not None
    tq.close()

    original = restore_tool_mod._PUBLISH_CRASH_HOOK
    restore_tool_mod._PUBLISH_CRASH_HOOK = _crash_hook("mid-swap")
    try:
        with pytest.raises(RuntimeError, match="mid-swap"):
            restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._PUBLISH_CRASH_HOOK = original
    assert (step_dir / "transfer_queue.pre_reconcile").exists(), "the crash must leave a healable backup"
    assert not (step_dir / "transfer_queue").exists(), "and no live dir (that is what makes it healable)"

    # NOW the journal's offsets fence goes bad (e.g. torn/corrupted on the shared fs).
    # Both ring slots must be poisoned: the fallback would otherwise recover from the
    # older complete slot, which is a different test (round 10).
    for slot in sorted(journal_root.glob("journal-*")):
        (slot / OPLOG_OFFSETS_FILE).write_text(json.dumps({"w.log": -5}))

    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)
    dry = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert dry["status"] == "unrecoverable_journal", dry
    assert dry["dry_run"] is True
    # the plan is still reported: the pending repairs were NOT silently executed
    assert dry["would_heal_publication"] is True
    assert dry["pending_transaction"]["state"] == "BEGIN"
    assert dry["checkpoint_tq_present"] is False, "a dry run must not heal, so no live dir yet"
    assert restore_tool_mod._exit_ok(dry) is False, "a missing live TQ dir must exit non-zero"
    assert _tree_bytes(store_root) == before_store, "dry: store mutated by a refused restore"
    assert _tree_bytes(ckpt_dir) == before_ckpt, "dry: checkpoint mutated by a refused restore"

    real = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert real["status"] == "healed_only", real
    assert real["healed_publication"] is True
    assert real["refused"]["status"] == "unrecoverable_journal", real["refused"]
    assert real["checkpoint_tq_present"] is True, "the heal must leave a live transfer_queue dir"
    assert restore_tool_mod._exit_ok(real) is False, "healed_only still needs an operator: non-zero exit"
    assert (step_dir / "transfer_queue").is_dir(), "the interrupted swap must be healed"
    assert not (step_dir / "transfer_queue.pre_reconcile").exists(), "the backup was renamed back into place"
    assert "completed_interrupted_transaction" not in real, "a refused journal completes no transaction"
    assert "discarded_transaction" not in real
    assert _tree_bytes(store_root) == before_store, "real: a refused journal must not touch the store"
    assert (step_dir / RECONCILE_STATE_FILE).exists(), "the marker must survive a refused restore"
    # the plan view (what was FOUND) is identical in both modes; only the disposition
    # differs, because dry mode never mutates
    plan_keys = [k for k in _PLAN_SCHEMA_KEYS if k != "detail"]
    assert {k: dry[k] for k in plan_keys} == {k: real[k] for k in plan_keys}

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_ambiguous_freshness_returns_the_identical_plan_in_both_modes(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    tq = tq_session
    finish_group(tq, partition_id, "amb", n=2, value=3)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
    assert jw.tick() is not None
    tq.close()

    # strip every freshness signal from the checkpoint snapshot: ordering cannot be
    # established (steps missing on one side, no usable fence_ts pair)
    ckpt_tq = step_dir / "transfer_queue"
    (ckpt_tq / "metadata.json").write_text(json.dumps({"user_metadata": {}}))
    fence = json.loads((ckpt_tq / "fence.json").read_text())
    fence.pop("fence_ts", None)
    (ckpt_tq / "fence.json").write_text(json.dumps(fence))

    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)
    results = {}
    for dry in (True, False):
        results[dry] = restore(
            store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry
        )
        assert results[dry]["status"] == "ordering_ambiguous", (dry, results[dry])
        assert results[dry]["dry_run"] is dry
        assert "--force" in results[dry]["detail"]
        assert results[dry]["freshness"]["ckpt_step"] is None
        assert _tree_bytes(store_root) == before_store, f"dry={dry}: store mutated"
        assert _tree_bytes(ckpt_dir) == before_ckpt, f"dry={dry}: checkpoint mutated"
    assert _plan_view(results[True]) == _plan_view(results[False])

    import transfer_queue as tq2

    tq2.init(tq_conf)


# --------------------------------- comment 3: transaction bound to store lineage


def test_recreated_store_cannot_complete_a_deleted_lineages_transaction(tq_session, tq_conf, tmp_path, partition_id):
    """The round-4 hole: marker txn == live-dir txn proves the SWAP became visible,
    but both artifacts can outlive their store. A store deleted and recreated at the
    same path (new lineage) must NOT let the stale transaction complete — its
    ``data.pt`` half belongs to the dead lineage."""
    import transfer_queue as tq

    tq_session.close()
    store = tmp_path / "store"
    journal_root = tmp_path / "journals"
    try:
        # ---- lineage A: ckpt at step 5, fresher journal (step 6) with a loader position
        tq.init(backend_config(store))
        lineage_a = (store / "_store_id").read_text()
        submit_prompt(tq, partition_id, "a1", status="running")
        finish_group(tq, partition_id, "a1", n=2, value=6)
        tag_prompt_finished(tq, partition_id, "a1")
        ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
        jw = JournalWriter(
            journal_root,
            interval_s=0.0,
            ring=3,
            dataloader_fn=lambda: {"pos": "journal6-lineageA"},
            metadata_fn=lambda: {"global_steps": 6},
        )
        assert jw.tick() is not None  # slot 0: lineage A's journal, preserved below
        jw.stop()  # release the ring lease: one writer per root (jw2 takes it below)
        tq.close()

        # crash between the TQ swap and the data.pt half: marker BEGIN(txn T, lineage A),
        # live dir published and stamped (txn T, lineage A)
        original = restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK
        restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK = _crash_hook("post-publish")
        try:
            with pytest.raises(RuntimeError, match="post-publish"):
                restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
        finally:
            restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK = original
        marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
        assert marker["state"] == "BEGIN" and marker["store_id"] == lineage_a
        live_md = json.loads((step_dir / "transfer_queue" / "metadata.json").read_text())
        assert live_md["user_metadata"]["reconcile_txn"] == marker["txn"]
        assert live_md["user_metadata"]["reconcile_store_id"] == lineage_a

        # ---- the store is deleted and recreated at the SAME path: lineage B, with an
        # OLDER journal (step 5) in a different ring slot (slot 0 survives untouched,
        # so the marker's source digest still matches — the round-4 guards all pass)
        import shutil as _shutil

        _shutil.rmtree(store)
        tq.init(backend_config(store))
        lineage_b = (store / "_store_id").read_text()
        assert lineage_b != lineage_a
        finish_group(tq, partition_id, "b1", n=2, value=5)
        jw2 = JournalWriter(
            journal_root,
            interval_s=0.0,
            ring=3,
            dataloader_fn=lambda: {"pos": "journal5-lineageB"},
            metadata_fn=lambda: {"global_steps": 5},
        )
        jw2._slot = 1  # do not recycle lineage A's slot: keep its digest intact
        assert jw2.tick() is not None
        tq.close()

        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
        # the B journal (step 5) is older than the A-published live dir (step 6), so
        # after the cross-lineage discard nothing is applied — and above all, lineage
        # A's data.pt half must never have been adopted
        assert result["status"] == "checkpoint_fresher", result
        assert "completed_interrupted_transaction" not in result, "a cross-lineage transaction must not complete"
        assert result["discarded_transaction"]["cross_lineage"] is True, result
        assert lineage_a[:8] in result["discarded_transaction"]["reason"] or "lineage" in str(
            result["discarded_transaction"]["reason"]
        )
        assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "step5"}, (
            "the deleted lineage's loader position was adopted"
        )
        assert not (step_dir / RECONCILE_STATE_FILE).exists(), "the stale marker must be discarded"
        assert (step_dir / "transfer_queue.pre_reconcile").exists(), "backups must stay intact"
    finally:
        try:
            tq.close()
        except Exception:  # noqa: BLE001 - fixture restoration must always run
            pass
        tq.init(tq_conf)


# ------------------------------- comment 4: exclusive per-checkpoint restore lock


def _locked_run_fixture(tq, tmp_path, partition_id):
    finish_group(tq, partition_id, "lk", n=2, value=9)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": "journal6"},
        metadata_fn=lambda: {"global_steps": 6},
    )
    assert jw.tick() is not None
    return ckpt_dir, step_dir, journal_root


def test_held_lock_fails_fast_and_is_never_broken(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """Fail-fast + conservative stale-lock policy: a held (possibly stale) lock stops
    both modes with the holder's identity, nothing is written, and the tool NEVER
    removes the lock itself — the operator does, after verifying the holder is dead
    (exactly the documented manual procedure, executed here)."""
    tq = tq_session
    ckpt_dir, step_dir, journal_root = _locked_run_fixture(tq, tmp_path, partition_id)
    tq.close()

    holder = {"uuid": "dead-run", "host": "some-other-host", "pid": 12345, "created_ts": 0.0}
    lock = step_dir / RESTORE_LOCK_FILE
    lock.write_text(json.dumps(holder))
    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)

    for dry in (False, True):
        result = restore(
            store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry
        )
        assert_plan_unavailable(result, status="locked", dry_run=dry)  # complete base schema, nulled plan
        assert result["checkpoint_step_dir"] == str(step_dir), (dry, result)
        assert result["lock"] == holder, "the holder's identity must be reported"
        assert _tree_bytes(store_root) == before_store, f"dry={dry}: store mutated despite the lock"
        assert _tree_bytes(ckpt_dir) == before_ckpt, f"dry={dry}: checkpoint mutated despite the lock"
    assert lock.read_text() == json.dumps(holder), "the tool must never break a lock it does not own"

    # the operator verified host/pid is dead and removes the lock manually:
    lock.unlink()
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "reconciled", result
    assert not lock.exists(), "the acquirer must release its own lock"

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_two_concurrent_restorers_cannot_interleave(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """Deterministic two-restorer interleaving (events as barriers): restorer A is
    held INSIDE its mutating phase (post-BEGIN, lock held); restorer B attempts the
    same checkpoint and must fail fast with ``locked`` — it never reaches the marker
    / backup / swap / data.pt paths. A then completes untouched."""
    tq = tq_session
    ckpt_dir, step_dir, journal_root = _locked_run_fixture(tq, tmp_path, partition_id)
    tq.close()

    entered, release = threading.Event(), threading.Event()

    def barrier_hook():  # runs inside A's lock, right after the BEGIN marker
        entered.set()
        assert release.wait(timeout=60), "restorer B never finished its attempt"

    result_b: dict = {}

    def second_restorer():
        assert entered.wait(timeout=60), "restorer A never reached its mutating phase"
        result_b.update(restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id))
        release.set()

    thread = threading.Thread(target=second_restorer)
    original = restore_tool_mod._POST_BEGIN_CRASH_HOOK
    restore_tool_mod._POST_BEGIN_CRASH_HOOK = barrier_hook
    thread.start()
    try:
        result_a = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._POST_BEGIN_CRASH_HOOK = original
        release.set()  # never leave A hanging if B failed early
        thread.join(timeout=60)

    assert_plan_unavailable(result_b, status="locked", dry_run=False)  # complete base schema, nulled plan
    assert result_b["lock"]["pid"] == os.getpid(), "the lock content must identify the live holder"
    assert result_a["status"] == "reconciled", result_a
    # coherent final state: exactly A's transaction, fully done
    marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert marker["state"] == "DONE"
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "journal6"}
    assert not (step_dir / RESTORE_LOCK_FILE).exists()

    import transfer_queue as tq2

    tq2.init(tq_conf)


# ------------------------ comment 5: force_store_mismatch narrowed (public restore)


def test_force_never_overrides_missing_or_empty_identities(tq_session, tq_conf, tmp_path):
    """The round-4 crash: --force-store-mismatch on a store with a missing identity
    logged 'forced, proceeding' and then died with an uncaught FileNotFoundError from
    the attach-only tq.init. Sanctioned resolution: the override covers ONLY two
    valid-but-different identities; missing/empty identities return a controlled
    fail-closed result — with zero writes — even WITH the flag."""
    import transfer_queue as tq

    tq_session.close()
    partition = "fsm"
    store = tmp_path / "store"
    try:
        tq.init(backend_config(store))
        submit_prompt(tq, partition, "g", status="running")
        finish_group(tq, partition, "g", n=2, value=5)
        tag_prompt_finished(tq, partition, "g")
        ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=1)
        journal_root = tmp_path / "journals"
        jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 2})
        assert jw.tick() is not None
        tq.close()
        true_id = (store / "_store_id").read_text()

        for case, value in {"missing": None, "empty": ""}.items():
            if value is None:
                (store / "_store_id").unlink()
            else:
                (store / "_store_id").write_text(value)
            before_store, before_ckpt = _tree_bytes(store), _tree_bytes(ckpt_dir)
            for dry in (False, True):
                result = restore(
                    store=store,
                    journal_root=journal_root,
                    ckpt_dir=ckpt_dir,
                    partition=partition,
                    dry_run=dry,
                    force_store_mismatch=True,  # must NOT apply to {case}
                )
                assert result["status"] == "store_mismatch", (case, dry, result)
                assert "does not apply" in result["detail"], (case, result["detail"])
            assert _tree_bytes(store) == before_store, f"{case}: store mutated (incl. any stamping)"
            assert _tree_bytes(ckpt_dir) == before_ckpt, f"{case}: checkpoint mutated"

        # two VALID but DIFFERENT identities: the override applies (the operator
        # vouches for the lineage), and the reconcile completes without a traceback
        (store / "_store_id").write_text(uuid.uuid4().hex)
        result = restore(
            store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition, force_store_mismatch=True
        )
        assert result["status"] == "reconciled", result
        # ... while without the flag the same pair still fails closed
        (store / "_store_id").write_text(true_id)
    finally:
        try:
            tq.close()
        except Exception:  # noqa: BLE001 - fixture restoration must always run
            pass
        tq.init(tq_conf)


# ----------------------------- comment 6: strict __rollout_n__ scalar validation


def _submit_prompt_with_raw_rollout_n(tq, partition_id, uid, value):
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": "running", "global_steps": 1},
        fields={"prompt_ids": torch.arange(4).unsqueeze(0), "__rollout_n__": value},
    )


@pytest.mark.parametrize(
    ("bad_value", "sessions"),
    [
        pytest.param(torch.tensor([True]), 1, id="bool"),  # old: int(True)=1 -> promoted a 1-session group
        pytest.param(torch.tensor([2.5]), 2, id="fractional"),  # old: int(2.5)=2 -> promoted at a truncated count
        pytest.param(torch.tensor([[2, 3]]), 2, id="multi_element"),  # old: [0]=2 -> second element ignored
    ],
)
def test_malformed_persisted_rollout_n_fails_closed(tq_session, partition_id, bad_value, sessions):
    """The expected count sizes the completeness check, so it must be EXACTLY one
    positive integer scalar. A malformed persisted value leaves the prompt running
    (``skipped_unverifiable_n``) — even under the operator's uniform-n assertion,
    because a record that cannot be parsed is a record that cannot be trusted."""
    tq = tq_session
    _submit_prompt_with_raw_rollout_n(tq, partition_id, "malformed", bad_value)
    finish_group(tq, partition_id, "malformed", n=sessions, value=5)  # complete at the coerced size

    for assume in (False, True):
        report = promote_complete_groups(
            tq, partition_id, group_size=sessions, assume_uniform_group_size=assume, dry_run=False
        )
        assert "malformed" in report["skipped_unverifiable_n"], (assume, report)
        assert "malformed" not in report["promoted"], (assume, report)
        assert tq.kv_list()[partition_id]["malformed"]["status"] == "running", "fail closed = leave for re-issue"


def test_valid_integral_scalars_still_promote(tq_session, partition_id):
    """The strictness must not break legal shapes: a plain int tensor and an
    integral float tensor (a float dtype holding a whole number) are both one
    positive integer scalar."""
    tq = tq_session
    _submit_prompt_with_raw_rollout_n(tq, partition_id, "int_n", torch.tensor([2]))
    finish_group(tq, partition_id, "int_n", n=2, value=6)
    _submit_prompt_with_raw_rollout_n(tq, partition_id, "float_n", torch.tensor([2.0]))
    finish_group(tq, partition_id, "float_n", n=2, value=7)

    report = promote_complete_groups(tq, partition_id, group_size=4)  # no assertion needed
    assert "int_n" in report["promoted"], report
    assert "float_n" in report["promoted"], report
