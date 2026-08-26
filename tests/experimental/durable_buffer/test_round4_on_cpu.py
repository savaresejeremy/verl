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
"""Regressions for the round-3 CR verdict (round 4 of hardening).

Each section pins one verdict comment:

1.  oplog=False removal — the full-manifest tick mode is GONE (config rejection at
    bootstrap and client), and a journal recorded with ``oplog: false`` — including
    one whose manifest records a same-name replacement written in the
    controller-save -> storage-hook gap as valid — is refused BEFORE any mutation,
    so the replacement can never be served as the snapshotted object;
2.  offsets fail closed — strict schema validation (str -> nonnegative int) with an
    ABORT for every corruption shape, each proven against a post-cut same-name
    rewrite; a journal tick whose offset pre-capture fails publishes NO complete
    slot (no stamp), so recovery lands on the previous good journal and fences the
    rewrite;
3.  restore_tool preflight + dry-run — identity is validated from plain file reads
    BEFORE healing/completion/tq.init (an unstamped store is never stamped, not even
    by ``tq.init``'s bootstrap), and ``--dry-run`` performs ZERO writes (byte-level
    check over the ENTIRE store and checkpoint trees) while reporting the complete
    fence/drop/orphan-prompt/promotion plan;
4.  transaction binding — the publication transaction carries a uuid stamped into
    both the marker and the published TQ dir, plus the source journal's content
    digest; completion happens only on proof (txn match + digest match), with fault
    injections before BEGIN, BEGIN->pre-publish over an older reconciled live dir,
    post-publish->pre-data.pt with ring-slot reuse, and post-data.pt->pre-DONE;
5.  fail-closed promotion — prompts whose expected rollout count is not durably
    present in their own TQ record are never promoted without the explicit
    ``assume_uniform_group_size`` operator assertion (exact-global-prefix,
    async-missing-field and sync/tag-only shapes);
6.  prompt-accounting characterization — one interleaving regression asserting the
    union of restored sources matches the documented duplicate-dispatch and skip
    windows (no false no-loss claim).
"""

from __future__ import annotations

import hashlib
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
    submit_prompt,
    tag_prompt_finished,
)
from omegaconf import OmegaConf

from verl.experimental.durable_buffer import (
    JournalPrecaptureError,
    UnrecoverableJournalError,
    fsck,
    recover,
)
from verl.experimental.durable_buffer import restore_tool as restore_tool_mod
from verl.experimental.durable_buffer.fs_kv import (
    FENCE_FILE,
    OPLOG_OFFSETS_FILE,
    DurableFSClient,
    backend_config,
    bootstrap_durable_fs,
)
from verl.experimental.durable_buffer.fsck import promote_complete_groups
from verl.experimental.durable_buffer.journal import JournalWriter, latest_journal
from verl.experimental.durable_buffer.restore_tool import RECONCILE_STATE_FILE, restore


def _storage_client(tq):
    return tq.get_client().storage_manager.storage_client


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


# ------------------------------------------------- comment 1: oplog=False removal


def test_oplog_false_mode_is_removed(tmp_path):
    """The full-manifest tick mode does not exist: both entry points reject it, and a
    rejected bootstrap must not even create the store directory."""
    with pytest.raises(ValueError, match="oplog"):
        DurableFSClient({"root": str(tmp_path / "s1"), "serializer": "tq", "oplog": False})
    conf = OmegaConf.create({"backend": {"DurableFS": {"root": str(tmp_path / "s2"), "oplog": False}}})
    with pytest.raises(ValueError, match="oplog"):
        bootstrap_durable_fs(conf)
    assert not (tmp_path / "s2").exists(), "a rejected bootstrap must not create the store"


def test_create_false_attaches_only_and_never_stamps(tmp_path):
    """The no-create mode (offline tools) must neither mkdir nor manufacture the
    ``_store_id`` identity that the fail-closed checks exist to demand."""
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(FileNotFoundError):
        DurableFSClient({"root": str(bare), "serializer": "tq", "create": False})
    assert not (bare / "_store_id").exists(), "create=False must never stamp"
    missing = tmp_path / "missing"
    conf = OmegaConf.create({"backend": {"DurableFS": {"root": str(missing), "create": False}}})
    with pytest.raises(FileNotFoundError):
        bootstrap_durable_fs(conf)
    assert not missing.exists(), "create=False must never mkdir"


def test_oplog_false_journal_with_gap_replacement_is_refused_before_any_mutation(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    """Deterministic fault injection into the controller-save -> storage-hook gap of
    the REMOVED mode: the controller is pickled, a same-name replacement lands, and
    only then are the manifest + fence written (recording the replacement as valid).
    Recovery must refuse the resulting ``oplog: false`` journal BEFORE
    ``tq.load_checkpoint`` — the replacement can never be served as the snapshotted
    object because no restored key ever exists to serve it."""
    tq = tq_session
    submit_prompt(tq, partition_id, "vic", status="running")
    keys = finish_group(tq, partition_id, "vic", n=2, value=11)
    tag_prompt_finished(tq, partition_id, "vic")
    idx = indexes_of(tq, partition_id, [keys[0]])[0]
    victim_key = f"{idx}@rewards"

    journal = tmp_path / "journal"
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1, "fence_ts": time.time()})  # controller cut

    # THE GAP: a same-name replacement (delete + rewrite of the same key -> same file
    # name, new inode and different bytes) lands after the controller pickle ...
    client = _storage_client(tq)
    client.clear([victim_key])
    client.put([victim_key], [torch.tensor([99.0])])
    # ... and only now does the (removed) mode write its storage artifacts, using the
    # REAL writer: the full manifest records the replacement's (inode, size) as valid
    # and the freshly captured offsets hide the in-gap records from any replay.
    client.write_manifest(journal, full=True)
    fence = json.loads((journal / FENCE_FILE).read_text())
    fence["oplog"] = False  # exactly what the removed mode recorded
    (journal / FENCE_FILE).write_text(json.dumps(fence))
    (journal / OPLOG_OFFSETS_FILE).unlink()  # the removed mode wrote no offsets fence

    tq = crash_and_restart(tq_conf)
    with pytest.raises(UnrecoverableJournalError, match="op log disabled"):
        recover(tq, store_root, journal, partition_id)
    # the refusal happened BEFORE load_checkpoint: no restored key exists that could
    # serve the replacement bytes under the snapshotted name
    assert tq.kv_list().get(partition_id, {}) == {}, "controller mutated by a refused recovery"
    # a direct fsck call (a caller that already loaded) refuses identically
    with pytest.raises(UnrecoverableJournalError):
        fsck(tq, store_root, journal, partition_id)
    # the replacement bytes still sit in the store, untouched and unreachable
    assert (store_root / f"{victim_key}.bin").exists()


# ------------------------------------------------- comment 2: offsets fail closed

_OFFSET_CORRUPTIONS = {
    "missing": None,
    "malformed_syntax": "{ not json",
    "wrong_type": json.dumps([1, 2, 3]),
    "nonnumeric": json.dumps({"w.log": "abc"}),
    "negative": json.dumps({"w.log": -5}),
    "boolean": json.dumps({"w.log": True}),
    "fractional": json.dumps({"w.log": 1.5}),
}


@pytest.mark.parametrize("case", sorted(_OFFSET_CORRUPTIONS))
def test_untrusted_offsets_abort_recovery(tq_session, tq_conf, store_root, partition_id, tmp_path, case):
    """Every corruption shape aborts BEFORE ``tq.load_checkpoint`` — never an
    incidental exception, never a silent fall-through to mtime — proven against a
    post-cut same-name rewrite that an offsets-less fence could not see."""
    tq = tq_session
    submit_prompt(tq, partition_id, "off", status="running")
    keys = finish_group(tq, partition_id, "off", n=2, value=11)
    tag_prompt_finished(tq, partition_id, "off")
    idx = indexes_of(tq, partition_id, [keys[0]])[0]
    victim_key = f"{idx}@rewards"

    journal = tmp_path / "journal"
    tq.save_checkpoint(str(journal), metadata={"global_steps": 1, "fence_ts": time.time()})

    # post-cut same-name rewrite: same file name, different bytes
    client = _storage_client(tq)
    client.clear([victim_key])
    client.put([victim_key], [torch.tensor([99.0])])

    payload = _OFFSET_CORRUPTIONS[case]
    offsets_path = journal / OPLOG_OFFSETS_FILE
    if payload is None:
        offsets_path.unlink()
    else:
        offsets_path.write_text(payload)

    tq = crash_and_restart(tq_conf)
    with pytest.raises(UnrecoverableJournalError):
        recover(tq, store_root, journal, partition_id)
    # fail closed = nothing restored: no key exists that could serve the replacement
    # bytes as the snapshotted object, and the store was not touched either
    assert tq.kv_list().get(partition_id, {}) == {}, f"{case}: controller mutated by a refused recovery"
    assert (store_root / f"{victim_key}.bin").exists(), f"{case}: store mutated by a refused recovery"


def test_precapture_failure_publishes_no_complete_slot(
    tq_session, tq_conf, store_root, partition_id, tmp_path, monkeypatch
):
    """A tick whose offset pre-capture fails must FAIL (no stamp): otherwise the slot
    would carry only the LATE (post-pickle) offsets and a post-cut same-name rewrite
    could evade the fence. Recovery then lands on the previous good journal, whose
    replay fences the rewrite — the replacement is never served."""
    tq = tq_session
    submit_prompt(tq, partition_id, "pc", status="running")
    keys = finish_group(tq, partition_id, "pc", n=2, value=11)
    tag_prompt_finished(tq, partition_id, "pc")
    idx = indexes_of(tq, partition_id, [keys[0]])[0]
    victim_key = f"{idx}@rewards"

    root = tmp_path / "journals"
    jw = JournalWriter(root, interval_s=0.0, ring=3, metadata_fn=lambda: {"global_steps": 1})
    info1 = jw.tick()
    assert info1 is not None
    good = Path(info1["dir"])

    def boom():
        raise OSError("stat of the op log failed")

    client = _storage_client(tq)
    monkeypatch.setattr(client, "oplog_offsets", boom)
    with pytest.raises(JournalPrecaptureError):
        # the pre-capture failure is the tick failure (surfaced via stats/logs in
        # production; tick() swallows it into a None return)
        from verl.experimental.durable_buffer.journal import _current_oplog_offsets

        _current_oplog_offsets()
    info2 = jw.tick()
    assert info2 is None, "a tick without a trustworthy pre-captured fence must fail"
    assert jw.stats["failures"] == 1
    monkeypatch.undo()

    # the failed slot published no stamp: the previous good journal stays newest
    assert latest_journal(root) == good
    stamps = list(root.glob("journal-*/journal_stamp.json"))
    assert len(stamps) == 1, f"a failed tick must not publish a stamped slot: {stamps}"

    # post-cut same-name rewrite, then recovery from the GOOD journal fences it
    client.clear([victim_key])
    client.put([victim_key], [torch.tensor([99.0])])
    tq = crash_and_restart(tq_conf)
    recover(tq, store_root, good, partition_id)
    live = tq.kv_list().get(partition_id, {})
    assert keys[0] not in live, "the rewritten sample must be dropped, not served"
    assert not (store_root / f"{victim_key}.bin").exists(), "the replacement must be fenced"
    # The sibling goes too, and so does the prompt: group atomicity (fsck._lost_group_siblings).
    # Keeping it would publish a finished 2-trajectory group holding one trajectory, which
    # nothing re-issues and nothing promotes. verl regenerates the group from the
    # dataloader position instead.
    assert keys[1] not in live, "a group that lost a member must not survive half-dropped"
    assert "pc" not in live, "the groupless prompt must go with it"
    assert not [k for k in live if k.startswith("pc")], sorted(live)


# ----------------------------- comment 3: restore_tool preflight + dry-run hygiene


def _build_isolated_run(tq_module, tmp_path, partition):
    """A dedicated store + trainer ckpt (step 1) + fresher journal (step 2)."""
    store = tmp_path / "idstore"
    conf = backend_config(store)
    tq_module.init(conf)
    submit_prompt(tq_module, partition, "g", status="running")
    finish_group(tq_module, partition, "g", n=2, value=5)
    tag_prompt_finished(tq_module, partition, "g")
    ckpt_dir, step_dir = _make_ckpt(tq_module, tmp_path, step=1)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": "journal2"},
        metadata_fn=lambda: {"global_steps": 2},
    )
    assert jw.tick() is not None
    jw.stop()  # release the ring lease: each phase below writes as its own driver
    tq_module.close()
    return store, ckpt_dir, step_dir, journal_root


def test_restore_identity_preflight_performs_zero_writes(tq_session, tq_conf, tmp_path):
    """Missing / empty / invalid / unequal store identities must abort through the
    PUBLIC restore() with ZERO writes of any kind — in particular, ``tq.init`` must
    never run against the store (its bootstrap would stamp an unstamped directory,
    manufacturing the identity the check exists to demand)."""
    import transfer_queue as tq

    tq_session.close()
    partition = "idp"
    store, ckpt_dir, step_dir, journal_root = _build_isolated_run(tq, tmp_path, partition)

    true_id = (store / "_store_id").read_text()
    cases = {"missing": None, "empty": "", "invalid": "not-a-uuid", "unequal": uuid.uuid4().hex}
    try:
        for case, value in cases.items():
            if value is None:
                (store / "_store_id").unlink()
            else:
                (store / "_store_id").write_text(value)
            before_store, before_ckpt = _tree_bytes(store), _tree_bytes(ckpt_dir)
            for dry in (False, True):
                result = restore(
                    store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition, dry_run=dry
                )
                assert result["status"] == "store_mismatch", (case, dry, result)
            assert _tree_bytes(store) == before_store, f"{case}: store bytes mutated (incl. any stamping)"
            assert _tree_bytes(ckpt_dir) == before_ckpt, f"{case}: checkpoint mutated"

        # with the true identity back, the same inputs reconcile
        (store / "_store_id").write_text(true_id)
        result = restore(store=store, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition)
        assert result["status"] == "reconciled", result
    finally:
        tq.init(tq_conf)  # leave the session fixture usable


def test_dry_run_is_zero_write_and_reports_the_complete_plan(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """Byte-level identity of the ENTIRE store and checkpoint trees across a dry run,
    while the dry report carries the complete fence/drop/orphan-prompt/promotion
    plan — equal to what the subsequent real run actually does."""
    tq = tq_session
    # material for every fsck action. 'con' is a TAG-ONLY prompt (no payload object):
    # after its trajectories are consumed post-tick, the restored prompt metadata has
    # no missing object of its own, so only the ORPHAN-PROMPT sweep can drop it —
    # the exact path the old dry-run under-reported.
    submit_prompt(tq, partition_id, "con", status="running", payload=False)
    finish_group(tq, partition_id, "con", n=2, value=1)
    tag_prompt_finished(tq, partition_id, "con")
    submit_prompt(tq, partition_id, "keep", status="running")
    finish_group(tq, partition_id, "keep", n=2, value=2)
    tag_prompt_finished(tq, partition_id, "keep")
    submit_prompt(tq, partition_id, "resc", status="running")  # complete but untagged -> PROMOTE
    finish_group(tq, partition_id, "resc", n=2, value=3)
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=1)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 2})
    assert jw.tick() is not None
    time.sleep(0.05)
    # post-journal churn: consumption (-> DROP + orphan prompt) and production (-> FENCE)
    tq.kv_clear(keys=["con", "con_0_0", "con_1_0"], partition_id=partition_id)
    finish_group(tq, partition_id, "post", n=2, value=9)
    tq.close()

    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)
    dry = restore(
        store=store_root,
        journal_root=journal_root,
        ckpt_dir=ckpt_dir,
        partition=partition_id,
        group_size=2,
        dry_run=True,
    )
    assert dry["status"] == "dry_run", dry
    assert _tree_bytes(store_root) == before_store, "dry run mutated the store (byte-level)"
    assert _tree_bytes(ckpt_dir) == before_ckpt, "dry run mutated the checkpoint (byte-level)"
    # the complete plan, not a truncated one:
    assert dry["fsck"]["fenced_objects"] >= 4, dry["fsck"]  # post's 2 trajectories x 2 fields
    assert dry["fsck"]["dropped_samples"] >= 2, dry["fsck"]  # con's 2 trajectories
    assert dry["fsck"]["prompts_dropped"] >= 1, dry["fsck"]  # con's orphaned prompt (was under-reported)
    assert dry["promotion"]["promoted"] == ["resc"], dry["promotion"]

    real = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id, group_size=2)
    assert real["status"] == "reconciled", real
    for key in ("fenced_objects", "missing_objects", "dropped_samples", "prompts_dropped", "dropped_by_partition"):
        assert dry["fsck"][key] == real["fsck"][key], (key, dry["fsck"], real["fsck"])
    assert dry["promotion"]["promoted"] == real["promotion"]["promoted"]

    import transfer_queue as tq2

    tq2.init(tq_conf)  # leave the session fixture usable


# ------------------------------------- comment 4: transaction binding fault matrix


def _txn_fixture(tq, tmp_path, partition_id, step=5, journal_step=6, loader="journal6"):
    ckpt_dir, step_dir = _make_ckpt(tq, tmp_path, step=step)
    finish_group(tq, partition_id, f"txn{journal_step}", n=2, value=9)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": loader},
        metadata_fn=lambda: {"global_steps": journal_step},
    )
    assert jw.tick() is not None
    jw.stop()  # release the ring lease: later phases write as their own driver incarnation
    return ckpt_dir, step_dir, journal_root


def _crash_hook(name):
    def crash():
        raise RuntimeError(f"killed at {name}")

    return crash


def test_txn_crash_before_begin_leaves_no_marker_and_converges(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    ckpt_dir, step_dir, journal_root = _txn_fixture(tq, tmp_path, partition_id)
    tq.close()
    ckpt_before = _tree_bytes(step_dir)

    original = restore_tool_mod._PRE_BEGIN_CRASH_HOOK
    restore_tool_mod._PRE_BEGIN_CRASH_HOOK = _crash_hook("pre-BEGIN")
    try:
        with pytest.raises(RuntimeError, match="pre-BEGIN"):
            restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._PRE_BEGIN_CRASH_HOOK = original

    assert not (step_dir / RECONCILE_STATE_FILE).exists(), "no marker may exist before BEGIN"
    assert _tree_bytes(step_dir) == ckpt_before, "the checkpoint must be untouched before BEGIN"

    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "reconciled", result
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE"
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "journal6"}

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_txn_begin_over_an_older_publication_is_discarded_not_completed(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    """A NEW transaction's BEGIN must never mistake an OLDER reconciled live dir for
    its own publication: the txn id in the live dir's metadata is the proof."""
    import transfer_queue as tq_mod

    tq = tq_session
    ckpt_dir, step_dir, journal_root = _txn_fixture(tq, tmp_path, partition_id, journal_step=6, loader="journal6")
    tq.close()
    # run 1 completes fully: live dir is reconciled (txn T1), data.pt = journal6
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "reconciled", result
    t1 = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["txn"]

    # newer work + a strictly fresher journal (step 7)
    tq_mod.init(tq_conf)
    finish_group(tq_mod, partition_id, "txn7", n=2, value=7)
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": "journal7"},
        metadata_fn=lambda: {"global_steps": 7},
    )
    assert jw.tick() is not None
    jw.stop()  # this driver incarnation is done with the ring
    tq_mod.close()

    # run 2 crashes after BEGIN (txn T2), before ANY publication step
    original = restore_tool_mod._POST_BEGIN_CRASH_HOOK
    restore_tool_mod._POST_BEGIN_CRASH_HOOK = _crash_hook("post-BEGIN")
    try:
        with pytest.raises(RuntimeError, match="post-BEGIN"):
            restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._POST_BEGIN_CRASH_HOOK = original
    marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert marker["state"] == "BEGIN" and marker["txn"] != t1
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "journal6"}, "nothing published yet"

    # run 3: the pairing cannot be proven (marker T2 vs live T1) -> marker discarded,
    # NO completion against the older publication, full fresh reconcile instead
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "reconciled", result
    assert "completed_interrupted_transaction" not in result, "an unprovable pairing must not be completed"
    final = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert final["state"] == "DONE" and final["txn"] not in (t1, marker["txn"])
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "journal7"}

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_txn_slot_reuse_cannot_pair_foreign_data_pt(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """Crash between the TQ swap and data.pt, then the journal ring recycles the
    source slot: completion must detect the digest mismatch and KEEP the checkpoint's
    data.pt rather than pairing one tick's TQ state with another tick's loader."""
    import transfer_queue as tq_mod

    tq = tq_session
    ckpt_dir, step_dir, journal_root = _txn_fixture(tq, tmp_path, partition_id, journal_step=6, loader="journal6")
    tq.close()

    original = restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK
    restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK = _crash_hook("post-publish")
    try:
        with pytest.raises(RuntimeError, match="post-publish"):
            restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK = original
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "step5"}, "data.pt half not yet applied"

    # ring-slot reuse: two more ticks (ring=2) recycle the ORIGINAL slot with a new
    # tick — same path, different stamp, different loader position
    tq_mod.init(tq_conf)
    finish_group(tq_mod, partition_id, "txn8", n=2, value=8)
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": "journal8"},
        metadata_fn=lambda: {"global_steps": 8},
    )
    jw._slot = 1  # continue the ring after slot 0 (the crashed run's source)
    assert jw.tick() is not None  # slot 1
    assert jw.tick() is not None  # slot 0 recycled
    jw.stop()
    tq_mod.close()

    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "reconciled", result
    completed = result["completed_interrupted_transaction"]
    assert completed["dataloader_state"].startswith("unavailable"), completed
    # the final state is coherent: the fresh reconcile (step 8) pairs with ITS loader
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "journal8"}
    assert json.loads((step_dir / RECONCILE_STATE_FILE).read_text())["state"] == "DONE"

    import transfer_queue as tq2

    tq2.init(tq_conf)


def test_txn_crash_before_done_completes_idempotently(tq_session, tq_conf, store_root, partition_id, tmp_path):
    tq = tq_session
    ckpt_dir, step_dir, journal_root = _txn_fixture(tq, tmp_path, partition_id, journal_step=6, loader="journal6")
    tq.close()

    original = restore_tool_mod._PRE_DONE_CRASH_HOOK
    restore_tool_mod._PRE_DONE_CRASH_HOOK = _crash_hook("pre-DONE")
    try:
        with pytest.raises(RuntimeError, match="pre-DONE"):
            restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._PRE_DONE_CRASH_HOOK = original
    marker = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert marker["state"] == "TQ_PUBLISHED"
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "journal6"}, "data.pt already applied"

    # the re-run proves the pairing (txn + digest) and finishes idempotently
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "checkpoint_fresher", result
    assert result["completed_interrupted_transaction"]["dataloader_state"] == "journal"
    assert torch.load(step_dir / "data.pt", weights_only=False) == {"pos": "journal6"}
    final = json.loads((step_dir / RECONCILE_STATE_FILE).read_text())
    assert final["state"] == "DONE" and final["txn"] == marker["txn"]

    # convergence: nothing changes on a further run
    snapshot = _tree_bytes(step_dir)
    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "checkpoint_fresher" and "completed_interrupted_transaction" not in result
    assert _tree_bytes(step_dir) == snapshot

    import transfer_queue as tq2

    tq2.init(tq_conf)


# --------------------------------------- comment 5: fail-closed group promotion


def test_exact_global_prefix_is_not_promoted_without_the_assertion(tq_session, partition_id):
    """The realistic async shape: the dispatched batch carried a batch-level
    ``__rollout_n__`` override (say 8) that verl does NOT persist, sessions 0..3
    landed before the crash, and the run-global rollout.n is 4. The record proves
    nothing, so without the operator assertion the prompt is left for re-issue."""
    tq = tq_session
    submit_prompt(tq, partition_id, "hidden", status="running")  # async payload, NO persisted override
    finish_group(tq, partition_id, "hidden", n=4, value=5)  # exactly sessions 0..group_size-1

    report = promote_complete_groups(tq, partition_id, group_size=4)
    assert "hidden" in report["skipped_no_durable_n"], report
    assert not report["promoted"], report
    assert tq.kv_list()[partition_id]["hidden"]["status"] == "running", "fail closed = leave for re-issue"

    # the explicit operator assertion (restore_tool --group-size) promotes at the
    # asserted size — documented as the operator's responsibility
    report = promote_complete_groups(tq, partition_id, group_size=4, assume_uniform_group_size=True)
    assert "hidden" in report["promoted"], report
    assert tq.kv_list()[partition_id]["hidden"]["status"] == "finished"


def test_sync_tag_only_prompt_needs_the_assertion(tq_session, partition_id):
    """Sync mode persists no prompt payload at all (tag-only record): the expected
    count is not durably knowable there either."""
    tq = tq_session
    submit_prompt(tq, partition_id, "tagonly", status="running", payload=False)
    finish_group(tq, partition_id, "tagonly", n=2, value=6)

    report = promote_complete_groups(tq, partition_id, group_size=2)
    assert "tagonly" in report["skipped_no_durable_n"], report
    assert tq.kv_list()[partition_id]["tagonly"]["status"] == "running"

    report = promote_complete_groups(tq, partition_id, group_size=2, assume_uniform_group_size=True)
    assert "tagonly" in report["promoted"], report
    assert tq.kv_list()[partition_id]["tagonly"]["status"] == "finished"


def test_persisted_rollout_n_promotes_without_any_assertion(tq_session, partition_id):
    """A per-sample override persisted in the prompt's own record IS a durable count:
    no assertion needed, and the global group_size is ignored in its favor."""
    tq = tq_session
    tq.kv_put(
        key="durable_n",
        partition_id=partition_id,
        tag={"is_prompt": True, "status": "running", "global_steps": 1},
        fields={"prompt_ids": torch.arange(4).unsqueeze(0), "__rollout_n__": torch.tensor([2])},
    )
    finish_group(tq, partition_id, "durable_n", n=2, value=7)

    report = promote_complete_groups(tq, partition_id, group_size=4)  # no assertion
    assert "durable_n" in report["promoted"], report
    assert report["trajectories_rescued"] == 2


# ------------------------- comment 6: prompt-accounting window characterization


def test_prompt_accounting_matches_the_documented_windows(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """Characterizes the documented contract — NOT a no-loss claim. Four dispatches
    interleave with one tick:

    * A: fetched + registered before the tick        -> restored, served-once
    * B: fetched after the loader capture, registered before the controller cut
         (the duplicate-dispatch window; in this tick geometry the pre-captured
         offsets fence B's group, converting the would-be duplicate into a
         regeneration)                                -> re-served, not duplicated
    * C: fetched before the loader capture, registered after the fence cut
         (the dispatch-spanning-both-cuts SKIP window) -> in NEITHER restored source
    * D: fetched + registered after the tick          -> fenced + re-served (fresh)

    The union of the restored sources must cover everything except exactly C, and
    nothing may be both buffered and re-served.
    """
    tq = tq_session
    fetched: list[str] = []

    def dispatch(uid, value):
        submit_prompt(tq, partition_id, uid, status="running")
        finish_group(tq, partition_id, uid, n=2, value=value)
        tag_prompt_finished(tq, partition_id, uid)

    fetched.append("A")
    dispatch("A", 1)
    fetched.append("C")  # C's registration is still in flight when the tick starts

    def loader_state_fn():
        state = {"pos": len(fetched)}  # captures A and C as served
        # B: fetched AFTER the loader capture, registered BEFORE the controller cut
        fetched.append("B")
        dispatch("B", 2)
        return state

    root = tmp_path / "journal_ring"
    jw = JournalWriter(
        root, interval_s=0.0, ring=2, dataloader_fn=loader_state_fn, metadata_fn=lambda: {"global_steps": 1}
    )
    info = jw.tick()
    assert info is not None
    jdir = Path(info["dir"])
    dispatch("C", 3)  # C's registration lands only AFTER the cut: spans BOTH cuts
    fetched.append("D")
    dispatch("D", 4)

    tq = crash_and_restart(tq_conf)
    recover(tq, store_root, jdir, partition_id)

    pos = torch.load(jdir / "dataloader_state.pt", weights_only=False)["pos"]
    epoch_order = ["A", "C", "B", "D"]
    assert fetched == epoch_order
    reserved = set(epoch_order[pos:])  # what a resumed loader re-serves this epoch
    live = tq.kv_list().get(partition_id, {})
    buffered = {k.rsplit("_", 2)[0] for k, tag in live.items() if not tag.get("is_prompt")}

    assert buffered == {"A"}, f"restored buffer: {sorted(live)}"
    assert reserved == {"B", "D"}, reserved
    # duplicate-dispatch window: bounded, and in this geometry converted to a
    # regeneration by the fence — nothing is both buffered and re-served
    assert buffered & reserved == set()
    # skip window: C is in NEITHER source — skipped for this epoch exactly as documented
    assert "C" not in (buffered | reserved)
    # and the union covers every dispatch except exactly C (no undocumented loss)
    assert buffered | reserved == {"A", "B", "D"}
