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
"""Regressions for the round-5 CR verdict (round 6 of hardening).

Each section pins one verdict comment:

1.  checkpoint selection is CONTAINED in ``--ckpt-dir`` — a hostile
    ``latest_checkpointed_iteration.txt`` (path traversal in the pointer text) and a
    symlinked ``global_step_*`` candidate are both rejected BEFORE any lock or
    mutation: nothing (``reconcile.lock``, markers, backups, swaps) is ever
    created, renamed or deleted outside the checkpoint root, proven by
    outside-sentinel byte-identity in BOTH dry and real modes;
2.  ``fence.json`` fails CLOSED for the op-log mode — it must decode to a JSON
    object whose ``oplog`` is strict-bool ``True``: malformed JSON, non-object
    payloads (which used to crash ``.get`` later), a missing ``oplog`` field,
    ``null``, ``0`` and ``"false"`` each abort with a controlled
    ``UnrecoverableJournalError`` before any controller or store mutation;
3.  ``reconcile.state`` is fail-closed — an array marker used to crash the
    planner, an empty or unknown-state object used to be silently overwritten as
    DONE, and a same-lineage completion with a null ``source_identity`` used to
    bypass the digest verification entirely: every unprovable shape now becomes an
    EXPLICIT discard entry in the shared plan (identical in dry and real), and
    completion requires a valid source identity/digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from helpers import (
    crash_and_restart,
    finish_group,
    submit_prompt,
    tag_prompt_finished,
)
from test_round5_on_cpu import assert_plan_unavailable

from verl.experimental.durable_buffer import UnrecoverableJournalError, recover
from verl.experimental.durable_buffer import restore_tool as restore_tool_mod
from verl.experimental.durable_buffer.fs_kv import FENCE_FILE, backend_config
from verl.experimental.durable_buffer.journal import JournalWriter
from verl.experimental.durable_buffer.restore_tool import (
    RECONCILE_STATE_FILE,
    RESTORE_LOCK_FILE,
    find_ckpt_tq_dir,
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


# ------------------------- comment 1: checkpoint selection contained in --ckpt-dir


def _plant_victim(tmp_path: Path) -> Path:
    """A SIBLING directory outside --ckpt-dir, dressed up as a plausible step dir:
    exactly where the round-5 code created ``reconcile.lock`` from a hostile pointer."""
    victim = tmp_path / "victim_step"
    (victim / "transfer_queue").mkdir(parents=True)
    (victim / "transfer_queue" / "metadata.json").write_text(json.dumps({"user_metadata": {"global_steps": 1}}))
    torch.save({"pos": "victim"}, victim / "data.pt")
    (victim / "sentinel.bin").write_bytes(b"MUST SURVIVE EVERY RESTORE ATTEMPT")
    return victim


def test_hostile_pointer_text_cannot_escape_the_checkpoint_root(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    """``latest_checkpointed_iteration.txt`` content is untrusted: a traversal like
    ``5/../../victim_step`` used to compose a step dir OUTSIDE --ckpt-dir and create
    ``reconcile.lock`` in the sibling. The pointer must full-match a bare integer;
    anything else is ignored and the validated fallback scan picks the real step dir."""
    tq = tq_session
    finish_group(tq, partition_id, "c1a", n=2, value=3)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
    assert jw.tick() is not None
    tq.close()

    victim = _plant_victim(tmp_path)
    # the demonstrated round-5 escape: "global_step_" + pointer resolves into the sibling
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text("5/../../victim_step")
    assert (ckpt_dir / "global_step_5/../../victim_step/sentinel.bin").exists(), "attack path must be live"

    found = find_ckpt_tq_dir(ckpt_dir)
    assert found is not None and found[0] == step_dir, "selection must fall back to the contained step dir"

    victim_before = _tree_bytes(victim)
    for dry in (True, False):
        result = restore(
            store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry
        )
        assert result["status"] == ("dry_run" if dry else "reconciled"), (dry, result)
        assert result["checkpoint_step_dir"] == str(step_dir), "the contained dir must be the one restored"
        assert _tree_bytes(victim) == victim_before, f"dry={dry}: the restore reached outside --ckpt-dir"
        assert not (victim / RESTORE_LOCK_FILE).exists(), f"dry={dry}: lock created in the sibling directory"
        assert not (victim / RECONCILE_STATE_FILE).exists(), f"dry={dry}: marker created in the sibling directory"

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_symlinked_candidate_cannot_escape_the_checkpoint_root(tq_session, tq_conf, partition_id, tmp_path):
    """A ``global_step_*`` entry that is a symlink (or resolves outside the root)
    must be rejected: with no other candidate the tool reports ``no_checkpoint``
    instead of locking/mutating the symlink's target."""
    import transfer_queue as tq

    tq_session.close()
    store = tmp_path / "store"
    tq.init(backend_config(store))
    try:
        finish_group(tq, partition_id, "c1b", n=2, value=4)
        journal_root = tmp_path / "journals"
        jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
        assert jw.tick() is not None
    finally:
        tq.close()

    victim = _plant_victim(tmp_path)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "global_step_7").symlink_to(victim)  # hostile: newest candidate, points outside
    assert (ckpt_dir / "global_step_7" / "sentinel.bin").exists(), "attack path must be live"

    assert find_ckpt_tq_dir(ckpt_dir) is None, "a symlinked candidate must never be selected"

    victim_before = _tree_bytes(victim)
    for dry in (True, False):
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry)
        assert_plan_unavailable(result, status="no_checkpoint", dry_run=dry)  # complete base schema, nulled plan
        assert result["checkpoint_step_dir"] is None, (dry, result)
        assert _tree_bytes(victim) == victim_before, f"dry={dry}: the restore reached through the symlink"
        assert not (victim / RESTORE_LOCK_FILE).exists(), f"dry={dry}: lock created through the symlink"

    tq.init(tq_conf)


def test_hostile_pointer_naming_a_symlink_is_rejected_too(tq_session, tq_conf, partition_id, tmp_path):
    """Belt and braces: a NUMERIC pointer whose target dir is a symlink outside the
    root is rejected by the same containment check (the pointer being well-formed
    does not make its target trustworthy)."""
    import transfer_queue as tq

    tq_session.close()
    store = tmp_path / "store"
    tq.init(backend_config(store))
    try:
        finish_group(tq, partition_id, "c1c", n=2, value=5)
        journal_root = tmp_path / "journals"
        jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
        assert jw.tick() is not None
    finally:
        tq.close()

    victim = _plant_victim(tmp_path)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "global_step_9").symlink_to(victim)
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text("9")

    assert find_ckpt_tq_dir(ckpt_dir) is None

    victim_before = _tree_bytes(victim)
    result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert_plan_unavailable(result, status="no_checkpoint", dry_run=False)  # complete base schema, nulled plan
    assert _tree_bytes(victim) == victim_before, "the restore reached through the pointed-at symlink"
    assert not (victim / RESTORE_LOCK_FILE).exists()

    tq.init(tq_conf)


# ---------------------- comment 2: fence.json fails CLOSED for the op-log mode


@pytest.mark.parametrize(
    "shape",
    ["malformed_json", "non_object", "missing_field", "null", "zero", "string_false", "missing_file", "invalid_utf8"],
)
def test_untrustworthy_fence_stamp_aborts_before_any_mutation(tq_session, tq_conf, tmp_path, partition_id, shape):
    """``fence.json`` must decode to a JSON object with strict-bool ``oplog: true``.
    The round-5 check (``.get("oplog") is False``) accepted a missing field, null,
    0 and "false" (fail open) and crashed on non-object JSON. Every invalid shape
    must now abort with a controlled ``UnrecoverableJournalError`` BEFORE any
    controller or store mutation — via ``recover()`` and, as a controlled
    ``unrecoverable_journal`` result with byte-identical trees, via the restore
    tool in BOTH modes."""
    import transfer_queue as tq

    tq_session.close()
    store = tmp_path / "store"
    conf = backend_config(store)
    tq.init(conf)
    try:
        submit_prompt(tq, partition_id, "fs", status="running")
        finish_group(tq, partition_id, "fs", n=2, value=7)
        tag_prompt_finished(tq, partition_id, "fs")
        ckpt_dir, _step_dir = _make_ckpt(tq, tmp_path, step=1)
        journal_root = tmp_path / "journals"
        jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 2})
        info = jw.tick()
        assert info is not None

        fence_path = Path(info["dir"]) / FENCE_FILE
        fence = json.loads(fence_path.read_text())
        if shape == "malformed_json":
            fence_path.write_text("{not json")
        elif shape == "non_object":
            fence_path.write_text(json.dumps([1, 2, 3]))  # used to crash .get later
        elif shape == "missing_field":
            fence.pop("oplog")
            fence_path.write_text(json.dumps(fence))
        elif shape == "null":
            fence["oplog"] = None
            fence_path.write_text(json.dumps(fence))
        elif shape == "zero":
            fence["oplog"] = 0
            fence_path.write_text(json.dumps(fence))
        elif shape == "string_false":
            fence["oplog"] = "false"
            fence_path.write_text(json.dumps(fence))
        elif shape == "missing_file":
            fence_path.unlink()
        elif shape == "invalid_utf8":
            # read_text raises UnicodeDecodeError (not OSError, not json's ValueError):
            # it used to escape as a raw traceback instead of the fail-closed refusal
            fence_path.write_bytes(b"\xff\xfe{ not utf-8 }")

        tq = crash_and_restart(conf)
        store_before = _tree_bytes(store)
        with pytest.raises(UnrecoverableJournalError):
            recover(tq, store, Path(info["dir"]), partition_id)
        assert _tree_bytes(store) == store_before, f"{shape}: store mutated by a refused recovery"
        assert tq.kv_list().get(partition_id, {}) == {}, f"{shape}: controller mutated by a refused recovery"

        # the restore tool surfaces the same refusal as a CONTROLLED result (no
        # traceback, no writes of any kind) in both modes
        tq.close()
        before_store, before_ckpt = _tree_bytes(store), _tree_bytes(ckpt_dir)
        for dry in (True, False):
            result = restore(
                store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=dry
            )
            assert result["status"] == "unrecoverable_journal", (shape, dry, result)
            assert _tree_bytes(store) == before_store, f"{shape} dry={dry}: store mutated by a refused restore"
            assert _tree_bytes(ckpt_dir) == before_ckpt, f"{shape} dry={dry}: checkpoint mutated by a refused restore"
    finally:
        try:
            tq.close()
        except Exception:  # noqa: BLE001 - fixture restoration must always run
            pass
        tq.init(tq_conf)


# ------------------------------ comment 3: reconcile.state marker fails closed


def _crash_hook(name):
    def crash():
        raise RuntimeError(f"killed at {name}")

    return crash


@pytest.mark.parametrize(
    ("marker_text", "kind"),
    [
        pytest.param(json.dumps([1, 2, 3]), "array", id="array"),  # used to CRASH the planner
        pytest.param(json.dumps({}), "empty_object", id="empty_object"),  # used to be silently overwritten as DONE
        pytest.param(json.dumps({"state": "WEIRD", "txn": "x"}), "unknown_state", id="unknown_state"),  # ditto
        pytest.param(b"\xff\xfe not utf-8", "invalid_utf8", id="invalid_utf8"),  # used to LEAK UnicodeDecodeError
    ],
)
def test_garbage_marker_becomes_an_explicit_discard_in_both_modes(
    tq_session, tq_conf, store_root, partition_id, tmp_path, marker_text, kind
):
    """An unprovable ``reconcile.state`` must never crash the tool and never be
    silently overwritten: BOTH modes carry the identical EXPLICIT discard entry in
    the shared plan, the dry run writes nothing, and the real run reports the
    discard before proceeding."""
    tq = tq_session
    finish_group(tq, partition_id, "gm", n=2, value=8)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 6})
    assert jw.tick() is not None
    tq.close()

    if isinstance(marker_text, bytes):  # invalid UTF-8 cannot go through write_text
        (step_dir / RECONCILE_STATE_FILE).write_bytes(marker_text)
    else:
        (step_dir / RECONCILE_STATE_FILE).write_text(marker_text)

    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)
    dry = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert dry["status"] == "dry_run", (kind, dry)
    assert dry["pending_transaction"]["state"] == "CORRUPT", (kind, dry)
    assert dry["would_complete_transaction"]["action"] == "discard_marker", (
        f"{kind}: an unprovable marker must be an EXPLICIT discard entry, not a silent overwrite"
    )
    assert _tree_bytes(store_root) == before_store, f"{kind}: dry run mutated the store"
    assert _tree_bytes(ckpt_dir) == before_ckpt, f"{kind}: dry run mutated the checkpoint"

    real = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert real["status"] == "reconciled", (kind, real)
    assert real["discarded_transaction"]["action"] == "discard_marker", (kind, real)
    assert real["would_complete_transaction"] == dry["would_complete_transaction"], (
        f"{kind}: dry and real must share the identical discard entry"
    )
    assert "completed_interrupted_transaction" not in real, kind
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE", (
        "after the explicit discard the NEW transaction's DONE marker is expected"
    )

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_null_digest_same_lineage_completion_is_discarded(tq_session, tq_conf, tmp_path, partition_id):
    """Completion REQUIRES a valid source identity/digest. A marker whose txn AND
    lineage both match the live dir but whose ``source_identity`` is null used to
    complete while skipping the digest verification entirely — a recycled ring slot
    could then pair one tick's TQ state with another tick's data.pt. It must now be
    an explicit discard, and the foreign data.pt must never be adopted."""
    import transfer_queue as tq

    tq_session.close()
    store = tmp_path / "store"
    journal_root = tmp_path / "journals"
    tq.init(backend_config(store))
    try:
        submit_prompt(tq, partition_id, "nd", status="running")
        finish_group(tq, partition_id, "nd", n=2, value=6)
        tag_prompt_finished(tq, partition_id, "nd")
        ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=5)
        jw = JournalWriter(
            journal_root,
            interval_s=0.0,
            ring=2,
            dataloader_fn=lambda: {"pos": "journal6"},
            metadata_fn=lambda: {"global_steps": 6},
        )
        assert jw.tick() is not None
        tq.close()

        # crash between the TQ swap and the data.pt half: marker BEGIN with matching
        # txn + lineage + a VALID captured identity
        original = restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK
        restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK = _crash_hook("post-publish")
        try:
            with pytest.raises(RuntimeError, match="post-publish"):
                restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
        finally:
            restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK = original
        marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
        assert marker["state"] == "BEGIN" and marker["source_identity"] is not None

        # the round-5 bypass shape: same txn, same lineage, but a null identity
        marker["source_identity"] = None
        (step_dir / RECONCILE_STATE_FILE).write_text(json.dumps(marker))

        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
        assert "completed_interrupted_transaction" not in result, (
            "a null-digest transaction must never complete (digest verification bypass)"
        )
        assert result["discarded_transaction"]["action"] == "discard_marker", result
        assert "source identity" in result["discarded_transaction"]["reason"], result
        assert result["status"] == "checkpoint_fresher", result
        assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "step5"}, (
            "the unverifiable journal slot's data.pt was adopted"
        )
        assert not (step_dir / RECONCILE_STATE_FILE).exists(), "the unprovable marker must be discarded"
        assert (step_dir / "transfer_queue.pre_reconcile").exists(), "backups must stay intact"
    finally:
        try:
            tq.close()
        except Exception:  # noqa: BLE001 - fixture restoration must always run
            pass
        tq.init(tq_conf)
