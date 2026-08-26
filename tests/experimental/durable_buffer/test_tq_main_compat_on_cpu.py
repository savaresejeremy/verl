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
"""Compatibility pins against TransferQueue@main and verl HEAD.

This package was written against an older TQ and an older verl. Each section here pins
one behavioural delta that the rest of the suite cannot see, because the rest of the
suite predates the change:

* **TQ#155 atomic checkpoint replacement.** ``tq.save_checkpoint`` now creates
  ``<dir>.tmp`` and ``<dir>.old`` SIBLINGS while it swaps a snapshot into place, and
  ``tq.load_checkpoint`` rolls forward from a surviving ``.old``. Three consequences are
  pinned: a ``.old`` inside the journal ring must never be selected as a slot, a ``.old``
  beside a checkpoint's ``transfer_queue`` dir must not steer the restore tool's source
  selection, and it must be gone before the tool's publication swap opens the very
  window that roll-forward reacts to.
* **TQ#141 ``mark_clearing``.** The controller zeroes production status before storage
  deletion, so a snapshot can be taken mid-clear. Pinned: what that leaves in the
  snapshot, and that fsck's existing DROP/FENCE rules already resolve it.
* **verl #6897 / #7373 buffer contract.** ``ReplayBufferAsync.sample`` carries a
  ``SkipManager.annotate_tq`` decorator, and ``separate_async`` now enters the buffer
  through ``wait_for_sampleable`` / ``get_sampleable_count`` as well as ``sample``.
  Pinned: the subclass carries the annotation, invokes it exactly ONCE per sample, and
  keeps journaling on the new entry points.
"""

from __future__ import annotations

import contextlib
import json
import logging
import pickle
import time

import pytest
import torch
from helpers import consume_group, finish_group, indexes_of, submit_prompt, tag_prompt_finished
from omegaconf import OmegaConf

from verl.experimental.durable_buffer import buffer
from verl.experimental.durable_buffer.buffer import (
    DurableReplayBuffer,
    _is_skip_wrapper,
    _peel_skip_wrapper,
)
from verl.experimental.durable_buffer.fsck import load_journal, recover
from verl.experimental.durable_buffer.journal import (
    JOURNAL_PREFIX,
    STAMP_FILE,
    JournalWriter,
    latest_journal,
)
from verl.experimental.durable_buffer.restore_tool import (
    TQ_PROTOCOL_SUFFIXES,
    _tq_protocol_siblings,
    restore,
)
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync
from verl.utils.skip import SkipManager

# --------------------------------------------------------------------------------------
# TQ#155: the journal ring must not select TQ's own protocol siblings
# --------------------------------------------------------------------------------------


def _fake_slot(root, name: str, completed_at: float | None, with_pickle: bool = True):
    """A journal-ring directory as the writer leaves it (stamp last, so no stamp = crashed)."""
    d = root / name
    d.mkdir(parents=True)
    if with_pickle:
        (d / "controller_state.pkl").write_bytes(pickle.dumps({"partitions": {}}))
    if completed_at is not None:
        (d / STAMP_FILE).write_text(json.dumps({"completed_at": completed_at, "fence_ts": completed_at}))
    return d


def test_latest_journal_ignores_tq_checkpoint_protocol_siblings(tmp_path):
    """A stamped ``journal-001.old`` must NOT win selection over an older ``journal-000``.

    TQ#155 renames the slot being replaced aside to ``<slot>.old`` inside the ring root
    (``transfer_queue/interface.py:1008-1052``). That directory is a COMPLETE previous
    snapshot: it has ``controller_state.pkl`` and this package's stamp, and its
    ``completed_at`` is newer than any surviving slot. Selecting on
    ``startswith("journal-")`` therefore picked it, which is how this regression was
    found. The content is genuine, so nothing is corrupted, but the returned path is
    outside the slot namespace that the ring's fence pairing and the restore tool's
    "slot recycled or rewritten since BEGIN" check reason about.

    ``restore_tool`` cannot pick a ``.old`` slot either, and that is by inheritance
    rather than by a second check: its preflight obtains the journal exclusively from
    ``latest_journal`` (``restore_tool._preflight``, ``journal = latest_journal(...)``),
    and so do fsck's offline entry points. Fixing selection here fixes every consumer,
    which is why this test asserts the invariant at ``latest_journal`` and
    ``test_restore_tool_ignores_a_stale_tq_old_sibling`` asserts the checkpoint-side
    analogue rather than duplicating this one.
    """
    ring = tmp_path / "ring"
    _fake_slot(ring, "journal-000", completed_at=100.0)
    _fake_slot(ring, "journal-001", completed_at=None)  # crashed mid-tick: pickle, no stamp
    _fake_slot(ring, "journal-001.old", completed_at=200.0)  # TQ#155 leftover, NEWEST
    _fake_slot(ring, "journal-001.tmp", completed_at=None)

    selected = latest_journal(ring)
    assert selected is not None
    assert selected.name == "journal-000", f"selected {selected.name!r}: a TQ protocol sibling is not a ring slot"


def test_latest_journal_ignores_a_stamped_tmp_sibling_too(tmp_path):
    """A ``.tmp`` is rejected by the grammar, not only by its missing stamp.

    On a clean save the stamp is written after ``tq.save_checkpoint`` returns, so a
    ``.tmp`` never has one. Relying on that would leave the grammar untested, and it is
    the grammar that has to hold: TQ writes the whole snapshot into ``.tmp`` before any
    rename, so a crash at the wrong instant can leave a fully-formed one.
    """
    ring = tmp_path / "ring"
    _fake_slot(ring, "journal-000", completed_at=100.0)
    _fake_slot(ring, "journal-000.tmp", completed_at=999.0)
    assert latest_journal(ring).name == "journal-000"


def test_latest_journal_accepts_the_exact_slot_grammar_only(tmp_path):
    """Names the writer can emit are selectable; nothing else wearing the prefix is."""
    ring = tmp_path / "ring"
    _fake_slot(ring, f"{JOURNAL_PREFIX}007", completed_at=10.0)
    for hostile in (
        f"{JOURNAL_PREFIX}001.old",
        f"{JOURNAL_PREFIX}001.tmp",
        f"{JOURNAL_PREFIX}abc",
        f"{JOURNAL_PREFIX}001-copy",
        f"{JOURNAL_PREFIX}001.bak",
        f"{JOURNAL_PREFIX}",
        # Unicode decimal digits: ``\d`` full-matches these, ``[0-9]`` does not, and the
        # writer composes slot names with ``{slot:03d}`` so it never emits them.
        f"{JOURNAL_PREFIX}\u0660\u0660\u0661",
        f"{JOURNAL_PREFIX}\uff10\uff10\uff12",
    ):
        _fake_slot(ring, hostile, completed_at=1000.0)
    assert latest_journal(ring).name == f"{JOURNAL_PREFIX}007"


def test_latest_journal_accepts_a_four_digit_slot(tmp_path):
    """``{slot:03d}`` zero-PADS, so a ring above 1000 emits four digits: still a slot."""
    ring = tmp_path / "ring"
    _fake_slot(ring, f"{JOURNAL_PREFIX}1024", completed_at=5.0)
    assert latest_journal(ring).name == f"{JOURNAL_PREFIX}1024"


def test_real_tick_survives_a_tq_protocol_sibling_in_the_ring(tq_session, partition_id, tmp_path):
    """End to end with a real TQ: an actual ``.old`` left in the ring is not selected.

    Constructed by driving ``tq.save_checkpoint`` at a ring path twice and interrupting
    its own cleanup, which is the only way to obtain a genuine TQ-written ``.old``.
    """
    tq = tq_session
    finish_group(tq, partition_id, "ring_probe", n=2, value=3)
    ring = tmp_path / "ring"
    jw = JournalWriter(ring, interval_s=0.0, ring=3, metadata_fn=lambda: {"global_steps": 1})
    first = jw.tick()
    assert first is not None
    slot0 = ring / f"{JOURNAL_PREFIX}000"

    # Re-save into slot 000 and keep the .old TQ would have deleted, exactly as a crash
    # between its rename-aside and its rmtree would leave it.
    old_sibling = ring / f"{JOURNAL_PREFIX}000.old"
    slot0.rename(old_sibling)
    time.sleep(0.02)
    second = jw.tick()  # writes slot 001
    assert second is not None
    # The .old carries the NEWEST stamp of anything in the ring.
    (old_sibling / STAMP_FILE).write_text(json.dumps({"completed_at": time.time() + 60, "fence_ts": 1.0}))

    selected = latest_journal(ring)
    assert selected.name == f"{JOURNAL_PREFIX}001", selected
    assert old_sibling.exists(), "the test's own fixture must still be there (selection only reads)"
    consume_group(tq, partition_id, "ring_probe", n=2)  # the store fixture is session-scoped


# --------------------------------------------------------------------------------------
# TQ#141 mark_clearing
# --------------------------------------------------------------------------------------


def test_mark_clearing_leaves_no_state_the_journal_parser_can_see(tq_session, partition_id, tmp_path):
    """Drive the REAL controller through ``mark_clearing`` and diff the snapshot.

    ``mark_clearing`` (``controller.py:1469-1502``) does exactly one thing: it zeroes
    ``production_status`` for the marked rows. There is no clearing-marked set in
    ``DataPartitionStatus`` for ``to_snapshot`` to copy, so a marked index stays in
    ``global_indexes``, ``keys_mapping``, ``revert_keys_mapping``,
    ``field_metadata[*].global_indexes`` and ``custom_meta``.

    That is what makes the state invisible to this package: ``load_journal`` reads only
    those five, never ``production_status`` or ``consumption_status``. This test pins the
    invisibility by asserting the parsed Journal is IDENTICAL before and after the mark,
    which is the property the fsck analysis rests on.
    """
    tq = tq_session
    submit_prompt(tq, partition_id, "mc", status="running")
    finish_group(tq, partition_id, "mc", n=2, value=11)
    tag_prompt_finished(tq, partition_id, "mc")
    keys = [f"mc_{s}_0" for s in range(2)]
    marked = indexes_of(tq, partition_id, keys)

    before_dir = tmp_path / "before"
    tq.save_checkpoint(str(before_dir), metadata={"global_steps": 1})
    before = load_journal(before_dir, partition=partition_id)

    # Step 1 of TQ's three-step clear (client.py:542-576), on its own.
    client = tq.get_client()
    meta = _FakeBatchMeta(global_indexes=marked, partition_ids=[partition_id] * len(marked))
    # The decorated coroutine, not its __wrapped__: @with_controller_socket is what
    # injects the DEALER socket, so unwrapping it would bypass the RPC entirely.
    client._run_coroutine(client._mark_clearing_in_controller(meta))

    after_dir = tmp_path / "after"
    tq.save_checkpoint(str(after_dir), metadata={"global_steps": 1})
    after = load_journal(after_dir, partition=partition_id)

    assert after.live == before.live, "mark_clearing must not remove indexes from global_indexes"
    assert after.expected == before.expected, "mark_clearing must not touch field metadata"
    assert after.view(partition_id).keys_mapping == before.view(partition_id).keys_mapping
    assert after.view(partition_id).revert == before.view(partition_id).revert
    assert after.tags == before.tags
    assert set(marked) <= after.live, "the marked indexes are still live metadata"

    # And the state IS observable in the raw pickle, so the assertion above is about what
    # load_journal reads rather than about the snapshot being unchanged.
    raw = pickle.loads((after_dir / "controller_state.pkl").read_bytes())
    status = raw["partitions"][partition_id].production_status
    assert status is not None
    assert int(status[marked, :].sum()) == 0, "production_status IS zeroed; the parser just never reads it"

    consume_group(tq, partition_id, "mc", n=2)


class _FakeBatchMeta:
    """The two attributes ``_mark_clearing_in_controller`` reads off a ``BatchMeta``."""

    def __init__(self, global_indexes, partition_ids):
        self.global_indexes = list(global_indexes)
        self.partition_ids = list(partition_ids)


def test_snapshot_taken_mid_clear_resolves_to_drop(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """A snapshot between TQ's storage delete and its metadata delete must DROP.

    This is the second half of the ``mark_clearing`` window and the one with a
    reachable, deterministic reproduction: mark, delete the objects, snapshot, then
    crash before the controller metadata is released. The snapshot expects
    ``(index, field)`` pairs whose objects are gone, which is fsck's clear-straddle
    direction, so the metadata must be dropped rather than restored into a controller
    that would serve keys whose reads can only fail.

    The first half (snapshot between the mark and the delete) is covered by
    ``test_mark_clearing_leaves_no_state_the_journal_parser_can_see``: the snapshot is
    self-consistent there, and it is the SAME snapshot content a pre-clear tick
    produces, so there is nothing further to assert about it.

    The no-FENCE half of the claim is asserted on the objects this test OWNS rather than
    on the report's store-wide ``fenced_objects`` count. The store fixture is
    session-scoped and shared, so the global count belongs to the whole suite, not to one
    test; scoping it also states the claim more exactly, which is that marking cannot
    cause collateral fencing of live data (it only ever deletes, so it cannot produce an
    object the snapshot fails to expect).
    """
    from helpers import crash_and_restart

    tq = tq_session
    submit_prompt(tq, partition_id, "straddle", status="running")
    finish_group(tq, partition_id, "straddle", n=2, value=12)
    tag_prompt_finished(tq, partition_id, "straddle")
    submit_prompt(tq, partition_id, "keeper", status="running")
    finish_group(tq, partition_id, "keeper", n=2, value=13)
    tag_prompt_finished(tq, partition_id, "keeper")

    keys = [f"straddle_{s}_0" for s in range(2)]
    marked = indexes_of(tq, partition_id, keys)
    keeper_keys = [f"keeper_{s}_0" for s in range(2)]
    keeper_objects = [
        store_root / f"{idx}@{field}.bin"
        for idx in indexes_of(tq, partition_id, keeper_keys)
        for field in ("response_ids", "rewards")
    ]
    assert all(p.exists() for p in keeper_objects), keeper_objects

    client = tq.get_client()
    meta = _FakeBatchMeta(global_indexes=marked, partition_ids=[partition_id] * len(marked))

    # Step 1: mark. Step 2: delete the storage bytes through the same store-client call
    # the manager's clear_data makes (``clear_by_index``; the manager would also call
    # ``clear`` for the batch's field-bearing keys first, which for these samples deletes
    # the same objects). Step 3 (metadata release) never happens: we snapshot and crash
    # instead, which is the straddle.
    client._run_coroutine(client._mark_clearing_in_controller(meta))
    store_client = client.storage_manager.storage_client
    store_client.clear_by_index(list(marked))

    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 7})
    assert jw.tick() is not None

    tq = crash_and_restart(tq_conf)
    report = recover(tq, store_root, latest_journal(journal_root), partition_id, group_size=2)

    assert report["dropped_samples"] >= len(keys), f"marked-and-deleted samples must DROP, got {report}"
    assert report["dropped_by_partition"].get(partition_id, 0) >= len(keys), report
    live = tq.kv_list().get(partition_id, {})
    assert not any(k.startswith("straddle") for k in live), f"stale metadata survived: {sorted(live)}"
    assert set(keeper_keys) <= set(live), "an untouched group must survive the reconcile"
    still_there = [p for p in keeper_objects if p.exists()]
    assert still_there == keeper_objects, (
        f"marking fenced live data it does not own: {sorted(set(keeper_objects) - set(still_there))}"
    )
    consume_group(tq, partition_id, "keeper", n=2)  # the store fixture is session-scoped


def test_snapshot_inside_the_unlink_loop_drops_the_whole_group(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """The third straddle state: a clear batch interrupted PART-WAY through its unlinks.

    ``DurableFSClient.clear``/``clear_by_index`` write one op-log ``D`` record for the
    whole batch and flush it before the first unlink, and the journal tick pre-captures
    the op-log offsets, so the entire batch sits BEHIND the fence: the tail replay
    reports nothing as ``deleted_after``. The already-unlinked objects DROP by the
    missing-object rule and the not-yet-unlinked ones are present AND expected, so on
    their own they would be KEPT — leaving a *finished* 2-trajectory group holding one
    trajectory, which nothing can ever complete (production ended, and a finished prompt
    is neither re-issued nor promoted).

    So the whole group must go, and its prompt with it: re-issue comes from the
    dataloader position, not from a half group nobody will finish.
    """
    from helpers import crash_and_restart

    tq = tq_session
    submit_prompt(tq, partition_id, "halfclear", status="running")
    finish_group(tq, partition_id, "halfclear", n=2, value=21)
    tag_prompt_finished(tq, partition_id, "halfclear")
    submit_prompt(tq, partition_id, "untouched", status="running")
    finish_group(tq, partition_id, "untouched", n=2, value=22)
    tag_prompt_finished(tq, partition_id, "untouched")

    traj_keys = [f"halfclear_{s}_0" for s in range(2)]
    marked = indexes_of(tq, partition_id, traj_keys)
    names = [f"{idx}@{field}.bin" for idx in marked for field in ("response_ids", "rewards")]
    assert all((store_root / n).exists() for n in names), names

    client = tq.get_client()
    meta = _FakeBatchMeta(global_indexes=marked, partition_ids=[partition_id] * len(marked))
    client._run_coroutine(client._mark_clearing_in_controller(meta))

    # The interrupted batch, reproduced through the store client's own intent log: ONE
    # flushed 'D' record covering every name (fs_kv._log_ops), then unlinks that die
    # after the first index. The surviving index's objects are what a KEEP verdict would
    # strand.
    store_client = client.storage_manager.storage_client
    store_client._log_ops("D", names)
    survivors = [n for n in names if not n.startswith(f"{marked[0]}@")]
    for name in names:
        if name not in survivors:
            (store_root / name).unlink()
    assert all((store_root / n).exists() for n in survivors), survivors

    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 11})
    assert jw.tick() is not None

    tq = crash_and_restart(tq_conf)
    # Dry run first: the closure is REPORTED but nothing is unlinked (pins the
    # dry_run guard inside _remove_closure_objects).
    dry = recover(tq, store_root, latest_journal(journal_root), partition_id, group_size=2, dry_run=True)
    assert dry["group_closure_objects"] == len(survivors), dry
    assert all((store_root / n).exists() for n in survivors), "a dry run must not unlink closure objects"
    report = recover(tq, store_root, latest_journal(journal_root), partition_id, group_size=2)

    # Nothing is replayed as deleted_after: the D record is behind the fence. That is the
    # premise of the whole state, so assert it rather than assume it.
    assert report["oplog_records_replayed"] == 0, report
    assert report["dropped_samples"] >= 2, f"the surviving sibling was kept: {report}"
    assert report["prompts_dropped"] >= 1, report

    live = tq.kv_list().get(partition_id, {})
    assert not any(k.startswith("halfclear") for k in live), (
        f"a half-lost group survived the reconcile: {sorted(k for k in live if k.startswith('halfclear'))}"
    )
    # The sibling's objects are residue once its metadata is gone: nothing refers to them,
    # and leaving them would strand bytes in the store until some later restore.
    assert not [n for n in survivors if (store_root / n).exists()], survivors
    assert report["group_closure_objects"] == len(survivors), report
    # An untouched group in the same partition is not collateral damage.
    assert {"untouched", "untouched_0_0", "untouched_1_0"} <= set(live), sorted(live)
    consume_group(tq, partition_id, "untouched", n=2)  # the store fixture is session-scoped


# --------------------------------------------------------------------------------------
# verl #6897: the rollout-skip annotation on the sample override
# --------------------------------------------------------------------------------------


def test_base_sample_is_wrapped_by_exactly_one_annotate_tq_wrapper():
    """Pins the assumption ``buffer._BASE_SAMPLE`` is built on.

    The subclass carries its own ``annotate_tq`` decorator and therefore calls the
    PEELED base, so that exactly one skip frame runs per sample. That peel is only
    correct while the base carries exactly one such wrapper and nothing else. If
    upstream restacks the decorators, this test fails and says so, instead of the peel
    silently no-op-ing and the skip hook running twice.
    """
    assert _is_skip_wrapper(ReplayBufferAsync.sample), (
        "ReplayBufferAsync.sample is no longer the annotate_tq wrapper; re-derive buffer._BASE_SAMPLE"
    )
    inner = ReplayBufferAsync.sample.__wrapped__
    assert not _is_skip_wrapper(inner), "unexpected extra annotate_tq frame under the first"
    assert not hasattr(inner, "__wrapped__"), (
        f"unexpected extra decorator under annotate_tq: {inner!r} still wraps something"
    )
    assert inner.__code__.co_name == "sample", inner.__code__.co_name
    assert _peel_skip_wrapper(ReplayBufferAsync.sample) is inner
    # And the peel refuses to unwrap anything that is not that wrapper.
    assert _peel_skip_wrapper(inner) is inner


def test_durable_sample_carries_the_annotation_equivalently_to_the_base():
    """The override is annotate_tq-wrapped, like the base, and is introspectable as such."""
    assert _is_skip_wrapper(DurableReplayBuffer.sample)
    assert DurableReplayBuffer.sample.__wrapped__.__qualname__ == "DurableReplayBuffer.sample"
    # Same wrapper factory on both, so a caller cannot tell them apart by shape.
    assert DurableReplayBuffer.sample.__code__ is ReplayBufferAsync.sample.__code__


class _RecordingSkip:
    """The duck-typed surface ``annotate_tq(phase="sample")`` uses off a skip instance."""

    support_online_step = False

    def __init__(self, steps):
        self.steps = steps
        self.saved: list = []

    def is_enabled(self):
        return True

    def should_save(self, step, partition_id="train"):
        return True

    def prepare_data(self, step, batch, global_steps):
        self.saved.append((step, global_steps))


@pytest.fixture
def recording_skip():
    """Install a recording rollout_tq skip instance for the duration of a test."""
    previous_instances = SkipManager.skip_instances
    previous_step = SkipManager.step
    skip = _RecordingSkip(steps=[5])
    SkipManager.skip_instances = dict(previous_instances)
    SkipManager.skip_instances["rollout_tq"] = skip
    SkipManager.set_step(5)
    try:
        yield skip
    finally:
        SkipManager.skip_instances = previous_instances
        SkipManager.step = previous_step


def _buffer_kwargs(**durable):
    return dict(
        trainer_mode="separate_async",
        trainer_config=OmegaConf.create({}),
        max_off_policy_threshold=10**6,
        max_off_policy_strategy="drop",
        sampler_kwargs=OmegaConf.create({"durable_buffer": durable}),
        poll_interval=0.05,
    )


def _stub_base_sample_internals(rb, monkeypatch, sentinel) -> list[str]:
    """Stub what ``ReplayBufferAsync.sample``'s BODY calls, on the instance.

    Deliberately NOT ``monkeypatch.setattr(buffer, "_BASE_SAMPLE", ...)``: that replaces
    the object under test, and every assertion about the peel then holds for whatever
    stub was installed instead. Patching the base's collaborators leaves the real
    function, the real peel and the real delegation in the call path, so
    ``_BASE_SAMPLE = ReplayBufferAsync.sample`` (peel removed) makes the tests fail.

    Returns the call log, which is how the tests show the base body ran at all.
    """
    calls: list[str] = []

    def wait_for_sampleable(global_steps, partition_id, batch_size):
        calls.append("wait_for_sampleable")
        return ["k0"], {}

    def select_prompt_uids(partition_id, sampleable_keys, batch_size):
        calls.append("_select_prompt_uids")
        return ["u0"], {}, {}

    def materialize_batch(partition_id, selected_prompt_uids, partition_snapshot):
        calls.append("_materialize_batch")
        return sentinel

    monkeypatch.setattr(rb, "wait_for_sampleable", wait_for_sampleable)
    monkeypatch.setattr(rb, "_select_prompt_uids", select_prompt_uids)
    monkeypatch.setattr(rb, "_materialize_batch", materialize_batch)
    return calls


def test_the_skip_hook_runs_exactly_once_per_sample(recording_skip, monkeypatch):
    """Two annotated frames in one call would dump the same mini-batch twice.

    ``RolloutSkip.prepare_data`` saves into the first missing inner sub-directory, so a
    doubled call writes one mini-batch into two of them and the
    ``parameter_sync_step > 1`` merge on load then reads a duplicate. The override
    therefore delegates to the PEELED base.
    """
    rb = DurableReplayBuffer(refill_fn=None, **_buffer_kwargs())
    sentinel = _FakeSampledBatch()
    calls = _stub_base_sample_internals(rb, monkeypatch, sentinel)

    # The peel is the thing under test, so state its result directly: what the override
    # calls must NOT itself be an annotate_tq wrapper, and it must be the function the
    # base's decorator wraps.
    assert not _is_skip_wrapper(buffer._BASE_SAMPLE), (
        "_BASE_SAMPLE is still an annotate_tq wrapper; the skip hook will fire twice per sample"
    )
    assert buffer._BASE_SAMPLE is ReplayBufferAsync.sample.__wrapped__

    batch, _metrics = rb.sample(5, "train", 2)
    assert batch is sentinel
    assert calls == ["wait_for_sampleable", "_select_prompt_uids", "_materialize_batch"], calls
    assert recording_skip.saved == [(5, 5)], f"skip hook ran {len(recording_skip.saved)} times, expected once"


def test_delegating_to_the_annotated_base_would_double_fire_the_hook(recording_skip):
    """Why the peel exists, demonstrated on the decorator itself rather than asserted.

    Same two-class shape as ``DurableReplayBuffer`` over ``ReplayBufferAsync``: annotate
    both, then delegate one way and the other. ``super().sample`` runs the hook twice
    because the base attribute IS the wrapper; the peeled call runs it once.
    """

    class _Base:
        @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
        def sample(self, global_steps, partition_id, batch_size):
            return _FakeSampledBatch(), {}

    class _SuperDelegating(_Base):
        @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
        def sample(self, global_steps, partition_id, batch_size):
            return super().sample(global_steps, partition_id, batch_size)

    _peeled_base = _peel_skip_wrapper(_Base.sample)

    class _PeelDelegating(_Base):
        @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
        def sample(self, global_steps, partition_id, batch_size):
            return _peeled_base(self, global_steps, partition_id, batch_size)

    _SuperDelegating().sample(5, "train", 2)
    assert len(recording_skip.saved) == 2, "super() delegation must be shown to double-fire, or this pin is vacuous"

    recording_skip.saved.clear()
    _PeelDelegating().sample(5, "train", 2)
    assert len(recording_skip.saved) == 1, "peeled delegation runs the hook exactly once"


class _Code310:
    """A code object as Python 3.10 exposes it: ``co_qualname`` arrived in 3.11."""

    def __init__(self, co_filename: str, co_name: str):
        self.co_filename = co_filename
        self.co_name = co_name


class _Func310:
    """Something with a ``__code__`` and nothing else ``_is_skip_wrapper`` may lean on."""

    def __init__(self, code):
        self.__code__ = code


def test_the_python_310_branch_identifies_the_wrapper_by_file_and_code_name():
    """verl supports 3.10 (``requires-python = ">=3.10,<3.13"``) and the suite runs 3.12.

    On 3.11+ the wrapper is identified by ``co_qualname``; on 3.10 that attribute does
    not exist and the fallback compares the defining FILE plus the code name. The
    fallback is therefore unreachable on this interpreter and would rot unnoticed, so it
    is exercised with duck-typed code objects that expose exactly what 3.10 does. Their
    field values are read off the real functions rather than written from memory.
    """
    wrapper_code = ReplayBufferAsync.sample.__code__  # the annotate_tq wrapper's own code
    inner_code = ReplayBufferAsync.sample.__wrapped__.__code__  # the method it wraps
    plain_code = ReplayBufferAsync.wait_for_sampleable.__code__  # never decorated
    assert not hasattr(_Code310("x", "y"), "co_qualname"), "the stand-in must not offer 3.11's attribute"
    assert (wrapper_code.co_filename, wrapper_code.co_name) == (buffer._SKIP_WRAPPER_FILE, "wrapper")

    def is_wrapper(code) -> bool:
        return buffer._is_skip_wrapper(_Func310(_Code310(code.co_filename, code.co_name)))

    assert is_wrapper(wrapper_code), "the 3.10 fallback must accept the base wrapper's shape"
    assert not is_wrapper(inner_code), "the function under the wrapper is not the wrapper"
    assert not is_wrapper(plain_code), "an undecorated method is not the wrapper"
    # Same file, different inner function: SkipManager.annotate's wrappers must not match.
    for name in ("sync_wrapper", "async_wrapper", "decorator"):
        assert not buffer._is_skip_wrapper(_Func310(_Code310(buffer._SKIP_WRAPPER_FILE, name))), name
    # And the 3.11+ branch still wins when co_qualname is present, whatever the file says.
    code_with_qualname = _Code310(buffer._SKIP_WRAPPER_FILE, "wrapper")
    code_with_qualname.co_qualname = "something.else"
    assert not buffer._is_skip_wrapper(_Func310(code_with_qualname))
    assert buffer._is_skip_wrapper(ReplayBufferAsync.sample), "the real 3.11+ path is unaffected"


def test_the_annotation_is_reached_through_the_override(recording_skip, monkeypatch):
    """The hook fires for the SUBCLASS, which is the behaviour #6897 asks for."""
    rb = DurableReplayBuffer(refill_fn=None, **_buffer_kwargs())
    sentinel = _FakeSampledBatch()
    calls = _stub_base_sample_internals(rb, monkeypatch, sentinel)
    assert not _is_skip_wrapper(buffer._BASE_SAMPLE)

    SkipManager.set_step(9)  # not in steps=[5]: the hook must stay out of the way
    batch, _metrics = rb.sample(9, "train", 2)
    assert batch is sentinel, "the override must return the base's batch on a non-skip step"
    assert recording_skip.saved == []
    SkipManager.set_step(5)
    rb.sample(5, "train", 2)
    assert recording_skip.saved == [(5, 5)]
    assert calls.count("_materialize_batch") == 2, calls


class _FakeSampledBatch:
    """Stands in for the ``KVBatchMeta`` ``sample`` returns (the hook reads one field)."""

    partition_id = "train"
    keys = ("k0",)


# --------------------------------------------------------------------------------------
# verl #7373: separate_async's two extra buffer entry points
# --------------------------------------------------------------------------------------


def test_durable_buffer_satisfies_the_separate_async_interface():
    """``trainer_separate_async`` raises unless BOTH methods are present on the buffer.

    They are satisfied by inheritance (``#7373`` extracted them out of
    ``ReplayBufferAsync.sample``), and inheritance is exactly why this needs pinning: a
    future override of ``sample`` alone, or a rename upstream, breaks the contract
    silently.
    """
    for name in ("wait_for_sampleable", "get_sampleable_count"):
        assert callable(getattr(DurableReplayBuffer, name, None)), f"separate_async requires {name}"
    rb = DurableReplayBuffer(refill_fn=None, **_buffer_kwargs())
    for name in ("wait_for_sampleable", "get_sampleable_count"):
        assert callable(getattr(rb, name))


def test_get_sampleable_count_drives_the_journal_poll_tick(tmp_path, monkeypatch):
    """The poll-tick backstop must fire through the NEW entry points too.

    ``get_sampleable_count`` and ``wait_for_sampleable`` both begin with
    ``_sync_metadata_from_transfer_queue`` (``replay_buffer.py:563``, ``:578``), which is
    where this subclass hooks ``maybe_tick``. Without that, a ``separate_async`` run in
    poll-tick mode could go a whole idle-lending window without journaling.
    """
    rb = DurableReplayBuffer(
        refill_fn=None,
        **_buffer_kwargs(journal_root=str(tmp_path / "j"), background=False, interval_s=0.0),
    )
    assert rb._journal is not None
    assert rb._journal._thread is None, "poll-tick mode must not start a thread"

    ticks: list[str] = []
    monkeypatch.setattr(rb._journal, "maybe_tick", lambda: ticks.append("tick") or {"dir": "stub"})
    # Neutralise the real TQ read: this test is about the tick, not about metadata.
    monkeypatch.setattr(ReplayBufferAsync, "_sync_metadata_from_transfer_queue", lambda self: None)

    assert rb.get_sampleable_count(1, "train") == 0
    assert ticks == ["tick"], "get_sampleable_count must reach the journal poll tick"

    rb._journal.stats["ticks"] = 1
    rb.get_sampleable_count(2, "train")
    assert len(ticks) == 2
    assert rb._journal_metrics, "a tick must publish journal metrics for the next sample()"


def test_get_sampleable_count_ticks_a_real_journal(tq_session, partition_id, tmp_path):
    """The same path with a real TQ and a real tick: a slot appears on disk."""
    tq = tq_session
    finish_group(tq, partition_id, "sc", n=2, value=21)
    tag_prompt_finished(tq, partition_id, "sc")
    root = tmp_path / "j"
    rb = DurableReplayBuffer(
        refill_fn=None,
        **_buffer_kwargs(journal_root=str(root), background=False, interval_s=0.0),
    )
    assert rb.get_sampleable_count(1, partition_id) >= 0
    assert latest_journal(root) is not None, "a real poll tick must publish a complete slot"
    assert rb._journal.stats["ticks"] == 1
    consume_group(tq, partition_id, "sc", n=2)


# --------------------------------------------------------------------------------------
# TQ#155: the checkpoint side, in the restore tool
# --------------------------------------------------------------------------------------


def _fake_checkpoint(tq, tmp_path, step: int, partition_id: str):
    ckpt_dir = tmp_path / "ckpts"
    step_dir = ckpt_dir / f"global_step_{step}"
    step_dir.mkdir(parents=True)
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text(str(step))
    torch.save({"pos": f"step{step}"}, step_dir / "data.pt")
    tq.save_checkpoint(str(step_dir / "transfer_queue"), metadata={"global_steps": step})
    return ckpt_dir, step_dir


def test_restore_tool_ignores_a_stale_tq_old_sibling(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """A stamped ``transfer_queue.old`` must not steer selection, and must not outlive the swap.

    Both halves matter and they fail differently. Selection: the sibling holds a
    complete, freshly-stamped snapshot of PRE-crash state, so if freshness comparison or
    source selection reached it, the tool would reconcile from state older than the
    journal it was given. Resurrection: ``tq.load_checkpoint`` renames
    ``transfer_queue.old`` back into place when ``transfer_queue`` is missing
    (``interface.py:1086-1092``), so a sibling left beside a published dir is a snapshot
    waiting to be rolled forward over the reconciled one.
    """
    from helpers import crash_and_restart

    tq = tq_session
    ckpt_dir, step_dir = _fake_checkpoint(tq, tmp_path, 3, partition_id)
    ckpt_tq = step_dir / "transfer_queue"

    # Work after the checkpoint, then a journal tick: the journal is the fresher source.
    for i, uid in enumerate(("sib1", "sib2")):
        submit_prompt(tq, partition_id, uid, status="running", global_steps=4)
        finish_group(tq, partition_id, uid, n=2, value=50 + i, global_steps=4)
        tag_prompt_finished(tq, partition_id, uid, global_steps=4)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": "step4"},
        metadata_fn=lambda: {"global_steps": 4},
    )
    assert jw.tick() is not None

    # A genuine TQ#155 leftover beside the live dir, carrying a NEWER step and fence_ts
    # than the journal, so a tool that looked at it would prefer it.
    old_sibling = step_dir / "transfer_queue.old"
    tq.save_checkpoint(str(old_sibling), metadata={"global_steps": 99, "fence_ts": time.time() + 600})
    tmp_sibling = step_dir / "transfer_queue.tmp"
    tmp_sibling.mkdir()
    (tmp_sibling / "controller_state.pkl").write_bytes(b"truncated garbage")
    assert {p.name for p in _tq_protocol_siblings(ckpt_tq)} == {"transfer_queue.old", "transfer_queue.tmp"}

    tq = crash_and_restart(tq_conf)
    tq.close()

    result = restore(
        store=store_root,
        journal_root=journal_root,
        ckpt_dir=ckpt_dir,
        partition=partition_id,
        group_size=2,
    )
    assert result["status"] == "reconciled", result
    # Selection came from the journal, not from the newer-looking sibling.
    assert result["source_selection"] == "journal", result
    assert result["source_step"] == 4, result
    assert result["freshness"]["ckpt_step"] == 3, (
        f"freshness must compare against the LIVE transfer_queue dir, not the .old sibling: {result['freshness']}"
    )
    # Both siblings were reported by the preflight and removed after the publication.
    assert sorted(result["tq_protocol_siblings"]) == sorted([str(old_sibling), str(tmp_sibling)]), result
    assert sorted(result["removed_tq_protocol_siblings"]) == sorted([str(old_sibling), str(tmp_sibling)]), result
    assert not old_sibling.exists(), "a stale .old beside a published dir is a resurrection hazard"
    assert not tmp_sibling.exists()
    assert (step_dir / "transfer_queue.pre_reconcile").exists(), "the tool's OWN rollback backup must survive"

    # Resurrection check: remove the published dir the way an operator or a stray script
    # would, and a stock verl resume must now find nothing rather than pre-reconcile state.
    import shutil

    import transfer_queue as tq2

    shutil.rmtree(ckpt_tq)
    tq2.init(tq_conf)
    with pytest.raises(FileNotFoundError):
        tq2.load_checkpoint(str(ckpt_tq))


def test_a_symlinked_tq_sibling_is_refused_before_any_lock(tq_session, store_root, partition_id, tmp_path):
    """A symlinked ``.old`` would let TQ's roll-forward rename read outside the checkpoint.

    ``load_checkpoint`` renames the sibling into place; renaming a symlink moves the
    link, so the subsequent read follows it wherever it points. That is refused at the
    layout gate, before the lock and before any mutation, like every other symlink in
    the step dir.
    """
    tq = tq_session
    ckpt_dir, step_dir = _fake_checkpoint(tq, tmp_path, 1, partition_id)
    outside = tmp_path / "outside_snapshot"
    outside.mkdir()
    (step_dir / "transfer_queue.old").symlink_to(outside, target_is_directory=True)

    result = restore(store=store_root, journal_root=None, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert result["status"] == "invalid_checkpoint_layout", result
    assert "transfer_queue.old" in result["detail"]
    assert "symlink" in result["detail"]
    assert result["plan_available"] is False


def test_the_preflight_reports_siblings_without_touching_them(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """A ``.old`` can be the ONLY copy of a snapshot after a crash mid-save.

    So the preflight reports it and stops there. Deleting or refusing on sight would
    turn the state TQ's roll-forward exists to repair into an operator ticket; the sweep
    happens only under the lock, with a complete reconciled snapshot already built and a
    live dir still in place.
    """
    tq = tq_session
    ckpt_dir, step_dir = _fake_checkpoint(tq, tmp_path, 2, partition_id)
    ckpt_tq = step_dir / "transfer_queue"
    # The crash-mid-save shape: live dir gone, .old holding everything.
    ckpt_tq.rename(step_dir / "transfer_queue.old")
    # ... and the crash took the controller with it, which is the only state the tool
    # reconciles in: a reachable controller is refused before the preflight runs at all.
    tq.close()

    result = restore(store=store_root, journal_root=None, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert result["tq_protocol_siblings"] == [str(step_dir / "transfer_queue.old")], result
    assert (step_dir / "transfer_queue.old").exists(), "dry-run must not remove the only surviving snapshot"
    assert "removed_tq_protocol_siblings" not in result, "the preflight never sweeps"

    import transfer_queue as tq2

    tq2.init(tq_conf)


def _crash_hook(label: str):
    def hook():
        raise RuntimeError(f"injected crash: {label}")

    return hook


def test_a_crash_inside_the_publication_swap_leaves_no_sibling_to_roll_forward(
    tq_session, tq_conf, store_root, partition_id, tmp_path, request
):
    """The swap's own window is why the sweep runs BEFORE the swap.

    ``_publish_reconciled_dir`` renames the live dir to ``.pre_reconcile`` and then
    renames the replacement in. Between those two renames ``transfer_queue`` does not
    exist, which is precisely the condition ``tq.load_checkpoint`` treats as an
    interrupted TQ save: it renames ``transfer_queue.old`` back into place
    (``interface.py:1088-1092``). Swept after the publication, the sibling is still there
    during that window, so a crash inside it leaves a checkpoint whose next resume
    resurrects PRE-reconcile buffer state over the tool's own backup.

    Swept before it, the crash leaves the healable state and nothing else: no ``.old``
    for TQ to roll forward, a ``.pre_reconcile`` backup for the tool to heal from, and a
    re-run that reconciles.
    """
    import shutil

    import transfer_queue as tq2

    from verl.experimental.durable_buffer import restore_tool as restore_tool_mod

    tq = tq_session
    ckpt_dir, step_dir = _fake_checkpoint(tq, tmp_path, 7, partition_id)
    ckpt_tq = step_dir / "transfer_queue"
    submit_prompt(tq, partition_id, "swap", status="running", global_steps=8)
    finish_group(tq, partition_id, "swap", n=2, value=61, global_steps=8)
    tag_prompt_finished(tq, partition_id, "swap", global_steps=8)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": "step8"},
        metadata_fn=lambda: {"global_steps": 8},
    )
    assert jw.tick() is not None

    # A stale TQ#155 leftover holding PRE-crash state, the resurrection candidate.
    old_sibling = step_dir / "transfer_queue.old"
    tq.save_checkpoint(str(old_sibling), metadata={"global_steps": 7})
    resurrection_marker = old_sibling / "RESURRECTED"
    resurrection_marker.write_text("pre-reconcile state")

    # From here the session TQ is closed and the test re-inits it several times. A
    # failed assertion in between would strand every later test in this file on
    # "TransferQueue is not initialized", so repair the session unconditionally.
    def _repair_session_tq():
        try:
            tq2.get_client()
        except Exception:  # noqa: BLE001 - any shape of "not initialized"
            tq2.init(tq_conf)

    request.addfinalizer(_repair_session_tq)
    tq.close()

    original = restore_tool_mod._PUBLISH_CRASH_HOOK
    restore_tool_mod._PUBLISH_CRASH_HOOK = _crash_hook("mid-swap")
    try:
        with pytest.raises(RuntimeError, match="mid-swap"):
            restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._PUBLISH_CRASH_HOOK = original

    assert not ckpt_tq.exists(), "the crash must land inside the swap window"
    assert (step_dir / "transfer_queue.pre_reconcile").exists(), "the tool's heal source must be there"
    assert not old_sibling.exists(), (
        "a .old surviving into the swap window is a snapshot tq.load_checkpoint will roll "
        "forward over the tool's backup"
    )

    # The stock resume path finds nothing to roll forward: the heal is the tool's.
    tq2.init(tq_conf)
    with pytest.raises(FileNotFoundError):
        tq2.load_checkpoint(str(ckpt_tq))
    assert not resurrection_marker.exists()
    tq2.close()

    real = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert real["status"] == "reconciled", real
    assert real["healed_publication"] is True, real
    assert ckpt_tq.exists()
    assert not _tq_protocol_siblings(ckpt_tq), "the re-run must not leave siblings either"

    shutil.rmtree(step_dir / "transfer_queue.pre_reconcile", ignore_errors=True)
    tq2.init(tq_conf)


def test_a_crash_after_the_publication_leaves_no_unswept_sibling(
    tq_session, tq_conf, store_root, partition_id, tmp_path
):
    """No swept-window gap: the completion path has nothing left to sweep.

    A crash between the swap and the ``TQ_PUBLISHED`` marker used to land in the gap
    between publishing and sweeping, and the completion path
    (``_execute_transaction_plan``) never swept, so the stale ``.old`` persisted for as
    long as the checkpoint did. With the sweep ahead of the swap there is no such
    ordering: by the time the live dir is replaced the siblings are already gone, and
    they stay gone across the completing re-run.
    """
    import transfer_queue as tq2

    from verl.experimental.durable_buffer import restore_tool as restore_tool_mod

    tq = tq_session
    ckpt_dir, step_dir = _fake_checkpoint(tq, tmp_path, 9, partition_id)
    ckpt_tq = step_dir / "transfer_queue"
    submit_prompt(tq, partition_id, "gap", status="running", global_steps=10)
    finish_group(tq, partition_id, "gap", n=2, value=71, global_steps=10)
    tag_prompt_finished(tq, partition_id, "gap", global_steps=10)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(
        journal_root,
        interval_s=0.0,
        ring=2,
        dataloader_fn=lambda: {"pos": "step10"},
        metadata_fn=lambda: {"global_steps": 10},
    )
    assert jw.tick() is not None
    old_sibling = step_dir / "transfer_queue.old"
    tq.save_checkpoint(str(old_sibling), metadata={"global_steps": 9})
    tmp_sibling = step_dir / "transfer_queue.tmp"
    tmp_sibling.mkdir()
    (tmp_sibling / "controller_state.pkl").write_bytes(b"interrupted build")
    tq.close()

    original = restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK
    restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK = _crash_hook("post-publish")
    try:
        with pytest.raises(RuntimeError, match="post-publish"):
            restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    finally:
        restore_tool_mod._POST_TQ_PUBLISH_CRASH_HOOK = original

    assert ckpt_tq.exists(), "the crash must land after the swap"
    assert not _tq_protocol_siblings(ckpt_tq), (
        "the publish->sweep gap is what left a stale sibling the completion path never removed"
    )

    real = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert real["status"] in ("reconciled", "checkpoint_fresher"), real
    assert not _tq_protocol_siblings(ckpt_tq), real

    tq2.init(tq_conf)


def test_the_sweep_gate_spares_a_load_bearing_old_sibling(tq_session, tq_conf, store_root, partition_id, tmp_path):
    """The ``ckpt_tq.exists()`` gate on the sweep call site is load-bearing: prove it.

    A crash between TQ's own two checkpoint renames leaves ``transfer_queue`` MISSING
    and ``transfer_queue.old`` holding the ONLY copy of that snapshot; TQ's roll-forward
    (``interface.py:1086-1092``) is the intended repair. A restore that runs in that
    state must not sweep the sibling on its way to publishing the reconciled dir: with
    the gate removed, the sweep composes the exact ``.old`` name from the absent live
    dir and deletes the only copy. Verified by mutation: dropping the gate makes this
    test fail on the sibling-survival assertion.
    """
    import shutil

    import transfer_queue as tq2

    tq = tq_session
    ckpt_dir, step_dir = _fake_checkpoint(tq, tmp_path, 3, partition_id)
    ckpt_tq = step_dir / "transfer_queue"
    submit_prompt(tq, partition_id, "gatekeeper", status="running", global_steps=4)
    finish_group(tq, partition_id, "gatekeeper", n=2, value=71, global_steps=4)
    tag_prompt_finished(tq, partition_id, "gatekeeper", global_steps=4)
    journal_root = tmp_path / "journals"
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 4})
    assert jw.tick() is not None

    # The mid-TQ-save crash state: live dir gone, .old is the only copy of the snapshot.
    old_sibling = step_dir / "transfer_queue.old"
    ckpt_tq.rename(old_sibling)
    only_copy_marker = old_sibling / "ONLY_COPY"
    only_copy_marker.write_text("the roll-forward repair depends on me")
    tq.close()

    result = restore(store=store_root, journal_root=journal_root, ckpt_dir=ckpt_dir, partition=partition_id)
    assert result["status"] == "reconciled", result
    assert result["removed_tq_protocol_siblings"] == [], (
        f"with the live dir absent the sweep must not run: {result['removed_tq_protocol_siblings']}"
    )
    assert old_sibling.is_dir(), "the sibling may be the only copy of a crashed TQ save; it must survive"
    assert only_copy_marker.read_text() == "the roll-forward repair depends on me"
    assert ckpt_tq.is_dir(), "the reconciled dir must still have been published"
    # The preflight still OBSERVED the sibling, it just refused to touch it.
    assert str(old_sibling) in (result["tq_protocol_siblings"] or []), result

    shutil.rmtree(old_sibling)
    tq2.init(tq_conf)


def test_registration_reaches_a_fresh_process_via_the_upstream_hook():
    """The zero-core-changes claim rests on verl's own VERL_USE_EXTERNAL_MODULES hook
    importing this package's registration module in EVERY process. Prove it the way a
    Ray worker would experience it: a fresh interpreter, the env var set, no explicit
    import of this package, and all three TransferQueue registries populated after
    nothing but ``import verl``.
    """
    import os
    import subprocess
    import sys

    probe = (
        "import verl\n"
        "from transfer_queue.storage.clients.base import StorageClientFactory\n"
        "from transfer_queue.storage.managers.base import StorageManagerFactory\n"
        "from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider\n"
        "assert any(k.lower() == 'durablefsclient' for k in StorageClientFactory._registry)\n"
        "assert any(k.lower() == 'durablefs' for k in StorageManagerFactory._registry)\n"
        "assert StorageBootstrapProvider.get_provider('DurableFS') is not None\n"
        "print('REGISTERED')\n"
    )
    env = dict(os.environ)
    env["VERL_USE_EXTERNAL_MODULES"] = "verl.experimental.durable_buffer.register"
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "REGISTERED" in done.stdout

    # And the negative control: without the env var, a fresh interpreter must NOT
    # have the backend (otherwise the test above proves nothing about the hook).
    env.pop("VERL_USE_EXTERNAL_MODULES")
    control = (
        "import verl\n"
        "from transfer_queue.storage.clients.base import StorageClientFactory\n"
        "assert not any(k.lower() == 'durablefsclient' for k in StorageClientFactory._registry)\n"
        "print('CLEAN')\n"
    )
    done = subprocess.run([sys.executable, "-c", control], capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "CLEAN" in done.stdout


def test_tq_protocol_suffixes_match_what_tq_actually_writes():
    """Pins the suffix list against TQ's own source rather than against a memory of it."""
    import inspect

    import transfer_queue as tq_mod

    src = inspect.getsource(tq_mod.save_checkpoint)
    for suffix in TQ_PROTOCOL_SUFFIXES:
        assert f'{{checkpoint_dir.name}}{suffix}"' in src, (
            f"tq.save_checkpoint no longer writes a {suffix} sibling; re-derive TQ_PROTOCOL_SUFFIXES"
        )
    load_src = inspect.getsource(tq_mod.load_checkpoint)
    assert 'f"{checkpoint_dir.name}.old"' in load_src, "tq.load_checkpoint no longer rolls forward from .old"


# --------------------------------------------------------------------------------------
# TQ@main shared ZMQ context: the manager's third constructor parameter
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def _captured_warnings(logger_name: str):
    """Collect this logger's records for the duration of the block."""

    class _Collector(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record):  # noqa: D102
            self.records.append(record)

    logger = logging.getLogger(logger_name)
    handler = _Collector()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_the_manager_accepts_the_zmq_context_the_factory_passes():
    """``StorageManagerFactory.create`` passes ``zmq_context`` and drops-with-a-warning
    any keyword the constructor cannot take (``storage/managers/base.py:483-507``).

    Accepting the parameter is what keeps this backend out of that path, so the claim is
    checked against TQ's own filter rather than against a reading of it: an upstream
    change to the filtering rule, or a signature drift here, fails this test.
    """
    import inspect

    from transfer_queue.storage.managers.base import StorageManagerFactory

    from verl.experimental.durable_buffer.fs_kv import BACKEND_NAME, DurableFSManager

    params = inspect.signature(DurableFSManager.__init__).parameters
    assert "zmq_context" in params, "the factory passes zmq_context; not accepting it warns on every create"
    assert params["zmq_context"].default is None, "it must stay optional for a TQ that never passes it"

    context = object()
    with _captured_warnings("transfer_queue") as records:
        kept = StorageManagerFactory._filter_supported_kwargs(DurableFSManager, {"zmq_context": context}, BACKEND_NAME)
    assert kept == {"zmq_context": context}, kept
    assert [r.getMessage() for r in records if "zmq_context" in r.getMessage()] == [], (
        "the installed TQ warned about zmq_context, so this signature no longer matches what it passes"
    )
    # The drop-and-warn behaviour the signature exists to avoid, shown on a keyword the
    # constructor really does not take.
    with _captured_warnings("transfer_queue") as records:
        dropped = StorageManagerFactory._filter_supported_kwargs(DurableFSManager, {"not_a_param": 1}, BACKEND_NAME)
    assert dropped == {}, dropped
    assert any("not_a_param" in r.getMessage() for r in records), [r.getMessage() for r in records]


def test_construction_still_works_against_a_tq_without_the_parameter(monkeypatch):
    """The legacy branch, driven by a parent that really cannot take the keyword.

    ``_PARENT_ACCEPTS_ZMQ_CONTEXT`` is a module global read inside ``__init__`` on every
    call (LOAD_GLOBAL, not a class-creation-time decision), so flipping it exercises the
    same code a pre-TQ#-shared-context runtime takes. The parent is replaced with a
    two-argument ``__init__`` for the same reason: a test that flips the flag while the
    real parent still accepts the keyword proves only that a branch was taken, not that
    the branch is the right one. Under that legacy parent, construction succeeds on the
    probe's False path, drops a caller's context with a warning, and would raise TypeError
    if the probe were wrong -- which is asserted rather than assumed.
    """
    from transfer_queue.storage.managers.base import KVStorageManager

    from verl.experimental.durable_buffer import fs_kv as fs_kv_mod

    calls: list[tuple] = []

    def legacy_parent_init(self, controller_info, config):
        """A ``KVStorageManager.__init__`` from before the shared-context work."""
        calls.append((controller_info, dict(config)))
        self.storage_client = None

    monkeypatch.setattr(KVStorageManager, "__init__", legacy_parent_init)
    monkeypatch.setattr(fs_kv_mod, "_PARENT_ACCEPTS_ZMQ_CONTEXT", False)

    manager = fs_kv_mod.DurableFSManager("controller-info", {"root": "/tmp/unused-by-the-fake-parent"})
    assert len(calls) == 1, calls
    assert calls[0][0] == "controller-info"
    assert calls[0][1]["client_name"] == fs_kv_mod.CLIENT_NAME, "the subclass still supplies its client_name"
    assert isinstance(manager, fs_kv_mod.DurableFSManager)

    with _captured_warnings("verl.experimental.durable_buffer.fs_kv") as records:
        fs_kv_mod.DurableFSManager("controller-info", {}, zmq_context=object())
    assert len(calls) == 2, calls
    assert any("zmq_context" in r.getMessage() for r in records), [r.getMessage() for r in records]

    # The probe is what keeps that call legal: forcing the modern path against this
    # legacy parent is the TypeError the probe exists to prevent.
    monkeypatch.setattr(fs_kv_mod, "_PARENT_ACCEPTS_ZMQ_CONTEXT", True)
    with pytest.raises(TypeError, match="zmq_context"):
        fs_kv_mod.DurableFSManager("controller-info", {}, zmq_context=None)


def test_the_probe_reports_the_installed_tq_truthfully():
    """The probe is derived from the parent signature, not from a pinned version floor."""
    import inspect

    from transfer_queue.storage.managers.base import KVStorageManager

    from verl.experimental.durable_buffer import fs_kv as fs_kv_mod

    assert fs_kv_mod._PARENT_ACCEPTS_ZMQ_CONTEXT == (
        "zmq_context" in inspect.signature(KVStorageManager.__init__).parameters
    )
    assert fs_kv_mod._PARENT_ACCEPTS_ZMQ_CONTEXT is True, (
        "the installed TransferQueue is expected to be TQ@main, whose KVStorageManager takes zmq_context"
    )
