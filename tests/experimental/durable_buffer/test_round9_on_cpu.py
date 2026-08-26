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
"""Regressions for the round-8 CR verdict (round 9 of hardening).

1.  Marker-source validation is UNCONDITIONAL: the round-8 code skipped provenance
    for a falsy ``source_journal`` (an empty string sailed on as "completable" and
    execution waved it through as benignly missing), and a malformed value — an
    embedded NUL — reached ``os.lstat`` and crashed with an uncaught ``ValueError``.
    Every pending source is now validated in BOTH planning and execution, and
    malformed shapes (empty, NUL, non-string, oversized) classify as
    ``invalid`` -> ``discard_marker`` with ``unprovable_source``, identically in
    dry and real mode, with the checkpoint's data byte-identical.

2.  Independent regressions for the two completion-time symlink guards that
    round-8 left covered only in combination: a SYMLINKED identity artifact
    (``journal_stamp.json`` pointing at a digest-identical external copy — reading
    through the link would "prove" the slot pairing) and a SYMLINKED
    ``dataloader_state.pt`` (``shutil.copy2`` follows links on read, so external
    bytes would be donated into the checkpoint's ``data.pt``). Each test goes red
    if exactly its guard is removed.
"""

from __future__ import annotations

import json
import shutil

import pytest
import torch
from test_round6_on_cpu import _tree_bytes
from test_round8_on_cpu import _interrupted_txn, private_store  # noqa: F401  (fixture)

from verl.experimental.durable_buffer.journal import DATALOADER_FILE, STAMP_FILE
from verl.experimental.durable_buffer.restore_tool import (
    RECONCILE_STATE_FILE,
    _marker_source_disposition,
    restore,
)

# --------- comment 1: unconditional marker-source validation, malformed shapes


def test_malformed_source_shapes_classify_invalid_without_syscalls(tmp_path):
    """Non-string, empty, NUL-carrying and oversized sources are not even plausible
    paths: they must classify ``invalid`` up front — no exception, no syscall — with
    the same unprovable-provenance verdict an escaping path gets. (The NUL shape
    used to reach ``os.lstat`` and crash with ``ValueError``; ``None`` would have
    crashed ``Path()`` with ``TypeError``.)"""
    ring = tmp_path / "ring"
    (ring / "journal-000").mkdir(parents=True)
    step = tmp_path / "ckpts" / "global_step_5"
    step.mkdir(parents=True)
    hostiles = (
        "",
        f"{ring / 'journal-000'}\x00evil",  # lexically contained: old code walked it into lstat
        None,
        123,
        str(ring / ("x" * 5000)),  # longer than any real journal/checkpoint path
    )
    for hostile in hostiles:
        disposition, problem = _marker_source_disposition(hostile, ring, step)
        assert disposition == "invalid", (hostile, disposition, problem)
        assert problem and "allowed root" in problem, (hostile, problem)


@pytest.mark.parametrize("shape", ["empty", "embedded_nul"])
def test_malformed_marker_source_declines_completion(private_store, partition_id, tmp_path, shape):  # noqa: F811
    """A pending marker whose recorded source is malformed — empty (the round-8 code
    SKIPPED validation entirely and treated it as completable) or NUL-carrying (it
    crashed ``os.lstat``) — must yield the explicit ``discard_marker`` /
    ``unprovable_source`` plan entry, identical in dry and real mode: no completion,
    no crash, checkpoint data byte-identical."""
    tq, store, _conf = private_store
    ckpt_dir, step_dir, journal_root, slot = _interrupted_txn(
        tq, store, tmp_path, partition_id, "_POST_TQ_PUBLISH_CRASH_HOOK"
    )
    marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert marker["state"] == "BEGIN"
    marker["source_journal"] = "" if shape == "empty" else f"{slot}\x00evil"
    (step_dir / RECONCILE_STATE_FILE).write_text(json.dumps(marker))
    shutil.rmtree(slot)  # ring recycled: any data.pt change would be the hostile marker's doing
    data_before = (step_dir / "data.pt").read_bytes()
    ckpt_before = _tree_bytes(ckpt_dir)

    dry = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert dry["status"] == "dry_run", dry
    plan_entry = dry["would_complete_transaction"]
    assert plan_entry["action"] == "discard_marker", plan_entry
    assert plan_entry.get("unprovable_source") is True, plan_entry
    assert ("empty" if shape == "empty" else "NUL") in plan_entry["reason"], plan_entry
    assert _tree_bytes(ckpt_dir) == ckpt_before, "the dry run must write nothing"

    real = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert real["status"] == "reconciled", real
    assert real["discarded_transaction"] == plan_entry, "dry and real must decide identically"
    assert "completed_interrupted_transaction" not in real, "a malformed source must never complete"
    assert (step_dir / "data.pt").read_bytes() == data_before, "checkpoint data.pt must stay byte-identical"
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "step5"}
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE"


# --------- comment 2: independent regressions for the completion-time symlink guards


def test_symlinked_identity_artifact_never_vouches_for_the_slot(private_store, partition_id, tmp_path):  # noqa: F811
    """Completion re-verifies the source slot against the digest captured at BEGIN by
    hashing the slot's stamp. Swap EXACTLY that artifact for a symlink to a
    byte-identical external copy: reading through the link would match the digest,
    "prove" the pairing, and copy the journal's dataloader state into ``data.pt``.
    The ``_snapshot_content_identity`` lstat gate must instead refuse to let a
    redirected artifact vouch for the slot — the data.pt half reports unavailable
    and the checkpoint's bytes stay put. (Remove that gate and this test goes red.)"""
    tq, store, _conf = private_store
    ckpt_dir, step_dir, journal_root, slot = _interrupted_txn(
        tq, store, tmp_path, partition_id, "_POST_TQ_PUBLISH_CRASH_HOOK"
    )
    marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert marker["state"] == "BEGIN"
    assert marker["source_identity"][0] == STAMP_FILE, "the digest under test must vouch via the stamp"
    data_before = (step_dir / "data.pt").read_bytes()

    outside = tmp_path / "outside_stamp.json"
    (slot / STAMP_FILE).rename(outside)  # byte-identical: following WOULD match the marker's digest
    (slot / STAMP_FILE).symlink_to(outside)
    outside_before = outside.read_bytes()

    dry = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert dry["status"] == "dry_run", dry
    assert dry["would_complete_transaction"]["action"] == "complete", (
        "txn/lineage/provenance/marker-identity are all genuinely fine here — the gate "
        "under test sits at the data.pt half itself"
    )
    assert (step_dir / "data.pt").read_bytes() == data_before, "the dry run mutated data.pt"

    real = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert real["status"] == "reconciled", real
    completed = real["completed_interrupted_transaction"]
    assert completed["dataloader_state"].startswith("unavailable"), completed
    assert "recycled or rewritten" in completed["dataloader_state"], completed
    assert (step_dir / "data.pt").read_bytes() == data_before, (
        "an identity read THROUGH a symlink vouched for the slot: the journal's "
        "dataloader state was copied into the checkpoint's data.pt"
    )
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "step5"}
    assert outside.read_bytes() == outside_before, "the stamp's external target must never be touched"
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE"


def test_symlinked_dataloader_state_never_donates_bytes(private_store, partition_id, tmp_path):  # noqa: F811
    """Identity and lineage check out for REAL (digest-matching, contained) slot
    artifacts — only ``dataloader_state.pt`` is a symlink to an external sentinel.
    The slot is half-recycled (controller pickle gone), so it is invisible to
    freshness selection and the preflight's own artifact validation never sees it:
    the ONLY read of that dataloader file is the completion path's, straight off
    the marker. ``shutil.copy2`` follows symlinks on read, so without the lstat
    gate at the copy the external payload would land in the checkpoint's
    ``data.pt``. The gate must keep the checkpoint's bytes and report the file as
    not regular/non-symlink. (Remove that gate and this test goes red.)"""
    tq, store, _conf = private_store
    ckpt_dir, step_dir, journal_root, slot = _interrupted_txn(
        tq, store, tmp_path, partition_id, "_POST_TQ_PUBLISH_CRASH_HOOK"
    )
    marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert marker["state"] == "BEGIN"
    data_before = (step_dir / "data.pt").read_bytes()

    outside = tmp_path / "outside_dataloader.pt"
    torch.save({"pos": "EXTERNAL"}, outside)  # a perfectly loadable payload: copy2 would donate it
    (slot / DATALOADER_FILE).unlink()
    (slot / DATALOADER_FILE).symlink_to(outside)
    (slot / "controller_state.pkl").unlink()  # half-recycled: latest_journal must not select it
    outside_before = outside.read_bytes()

    dry = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert dry["status"] == "dry_run", dry
    assert dry["source_selection"] == "checkpoint", "the half-recycled slot must not drive the reconcile"
    assert dry["would_complete_transaction"]["action"] == "complete", dry["would_complete_transaction"]
    assert (step_dir / "data.pt").read_bytes() == data_before, "the dry run mutated data.pt"

    real = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert real["status"] == "reconciled", real
    completed = real["completed_interrupted_transaction"]
    assert "not a regular non-symlink file" in completed["dataloader_state"], completed
    assert (step_dir / "data.pt").read_bytes() == data_before, (
        "copy2 followed the dataloader symlink: external bytes were donated into data.pt"
    )
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "step5"}
    assert outside.read_bytes() == outside_before, "the sentinel's bytes must never change"
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE"
