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
"""``DurableReplayBuffer``: a ``ReplayBufferAsync`` that journals TQ metadata.

Two ways to get journal ticks; pick one.

1. **Background thread via ``register.start_journal(...)`` (recommended)** — spawns a
   ``JournalWriter`` in the driver process. Nothing about the buffer changes, so DAPO
   filtering / failure refill keep working exactly as configured.
2. **This subclass** — plugged in with
   ``trainer.v1.sampler.custom_sampler.{path,name}``. By default it ALSO runs the
   journal in a background thread (started on construction); ``background: false``
   switches to poll-tick mode, which ticks inside
   ``_sync_metadata_from_transfer_queue`` — i.e. only while the trainer is asking for
   a batch, measured ~once per training step, so the effective journal cadence is the
   step time, not ``interval_s`` (see the class docstring).

Custom-sampler caveat (verified against ``trainer_base._build_replay_buffer``): verl
passes ``filter_groups_metric`` / ``sync_refill_failed_groups`` (and the Sync-DAPO
batch-size knobs) only to NON-custom samplers — a custom sampler gets just the base
constructor args plus ``sampler_kwargs``, and the base class does NOT read those
features out of ``sampler_kwargs`` on its own. This subclass therefore accepts them
under ``sampler_kwargs.durable_buffer`` and forwards them into the base constructor
(before any base validation), so the documented re-supply path actually works; an
unknown key under ``durable_buffer`` raises instead of being silently dropped.

Config for the subclass path::

    trainer.v1.sampler.custom_sampler.path = verl/experimental/durable_buffer/buffer.py
    trainer.v1.sampler.custom_sampler.name = DurableReplayBuffer
    trainer.v1.sampler.sampler_kwargs.durable_buffer.journal_root = /shared/fs/runs/<run>/journal
    trainer.v1.sampler.sampler_kwargs.durable_buffer.interval_s = 2.0
    # optional re-supplied base features (dropped by verl for custom samplers):
    trainer.v1.sampler.sampler_kwargs.durable_buffer.filter_groups_metric = acc
    trainer.v1.sampler.sampler_kwargs.durable_buffer.sync_refill_failed_groups = false
"""

from __future__ import annotations

import logging

from verl.experimental.durable_buffer.journal import JournalWriter, JournalWriterLeaseError
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync
from verl.utils.skip import SkipManager

logger = logging.getLogger(__name__)

# Journal knobs owned by this subclass.
_JOURNAL_KEYS = {"journal_root", "interval_s", "ring", "background"}
# Base ReplayBuffer(Async) constructor kwargs that verl deliberately does not pass to
# custom samplers; re-suppliable through sampler_kwargs.durable_buffer.* (forwarded
# BEFORE super().__init__ so the base class's own validation applies to them).
_FORWARDED_BASE_KEYS = {
    "filter_groups_metric",
    "sync_refill_failed_groups",
    "train_batch_size",
    "gen_batch_size",
    "max_inflight_gen_batches",
}

#: ``__code__.co_qualname`` of the wrapper ``SkipManager.annotate_tq`` installs. Used to
#: identify that wrapper specifically: ``functools.wraps`` copies ``__name__`` and
#: ``__qualname__`` from the wrapped function, so those cannot distinguish a wrapper from
#: what it wraps, but ``__code__`` is the wrapper's own and is not copied.
_SKIP_WRAPPER_QUALNAME = "SkipManager.annotate_tq.<locals>.decorator.<locals>.wrapper"
#: Where that wrapper is defined, derived rather than hardcoded. Needed because
#: ``co_qualname`` only exists on Python 3.11+ and verl supports 3.10
#: (``pyproject.toml``: ``requires-python = ">=3.10,<3.13"``).
_SKIP_WRAPPER_FILE = SkipManager.annotate_tq.__func__.__code__.co_filename


def _is_skip_wrapper(func) -> bool:
    """Is ``func`` the wrapper ``SkipManager.annotate_tq`` installs?"""
    code = getattr(func, "__code__", None)
    if code is None:
        return False
    qualname = getattr(code, "co_qualname", None)
    if qualname is not None:  # Python 3.11+
        return qualname == _SKIP_WRAPPER_QUALNAME
    # Python 3.10: identify by defining file plus the wrapper's own code name. Within
    # skip_manager.py, "wrapper" belongs to annotate_tq alone; annotate's inner functions
    # are sync_wrapper and async_wrapper. Unreachable on 3.11+, so it is pinned by
    # ``test_the_python_310_branch_identifies_the_wrapper_by_file_and_code_name``, which
    # feeds it code objects shaped the way 3.10 exposes them.
    return code.co_filename == _SKIP_WRAPPER_FILE and code.co_name == "wrapper"


def _peel_skip_wrapper(func):
    """``func`` with an ``annotate_tq`` wrapper removed, or ``func`` unchanged.

    Peels at most ONE hop, and only when that hop is provably the ``annotate_tq``
    wrapper: anything else in the chain is left in place rather than unwrapped blindly,
    so a decorator upstream may add later cannot be silently discarded here.
    """
    inner = getattr(func, "__wrapped__", None)
    if inner is not None and _is_skip_wrapper(func):
        return inner
    return func


#: ``ReplayBufferAsync.sample`` WITHOUT its skip wrapper.
#:
#: ``sample`` is annotated at HEAD (``replay_buffer.py:575``:
#: ``@SkipManager.annotate_tq(role="rollout_tq", phase="sample")``), and this subclass
#: carries the same annotation on its override so a static check of the class, and a
#: future refactor that stops delegating, both still see rollout-skip support. Two
#: annotated frames in one call would run the skip hook twice, which is NOT merely
#: wasteful: ``RolloutSkip.prepare_data`` writes the batch into the first missing inner
#: sub-directory (``verl/utils/skip/rollout_skip.py:299-320``,
#: ``_find_first_missing_inner``), so a doubled call dumps one mini-batch into two
#: sub-directories and the ``parameter_sync_step > 1`` merge on load then reads a
#: duplicate. Calling the peeled base keeps it at exactly one.
#:
#: If upstream changes the decorator stack on ``ReplayBufferAsync.sample``, the peel
#: silently stops matching and both frames run again, so the assumption is pinned by a
#: test that fails loudly instead
#: (``tests/experimental/durable_buffer/test_tq_main_compat_on_cpu.py``:
#: ``test_base_sample_is_wrapped_by_exactly_one_annotate_tq_wrapper``).
_BASE_SAMPLE = _peel_skip_wrapper(ReplayBufferAsync.sample)


class DurableReplayBuffer(ReplayBufferAsync):
    """Async replay buffer that journals TQ metadata continuously.

    The journal runs in a **background thread** (started on construction), because the
    sampler's poll loop only spins while the trainer is asking for a batch — ticking
    from the poll loop measured one tick per training step (~26 s apart on the 0.5B
    cluster run), which reopens exactly the loss window the journal exists to close.
    The ``maybe_tick`` in ``_sync_metadata_from_transfer_queue`` stays as a backstop
    for a dead thread; ``JournalWriter`` serializes whole ticks on a non-blocking
    lock, so the two can never overlap (the poll-side call is skipped, not queued,
    while a background tick runs). TQ's client is thread-safe for this because its sync
    methods marshal onto the client's own background event loop; see the
    ``JournalWriter`` docstring for the derivation against TQ@main and for the one
    constraint it leaves (an unreachable controller parks a tick indefinitely).
    ``durable_buffer.background: false`` opts into pure poll-tick mode —
    supported, but its effective cadence IS the training-step time regardless of
    ``interval_s``; use it only when a background thread is unacceptable.

    Extras wired with zero verl-core changes (registration rides verl's external-modules hook):
      * dataloader state — discovered lazily through ``refill_fn.__self__`` (the bound
        trainer), journaled with each tick so a resume's prompt accounting shrinks to
        the two bounded windows around a single tick (duplicate-dispatch and skip —
        NOT exact; see the ``JournalWriter`` docstring and the README's guarantees)
        instead of re-serving everything since the last trainer checkpoint;
      * ``global_steps`` — recorded from ``sample()`` so the journal metadata carries
        the training step it was taken under.
    """

    def __init__(self, *args, **kwargs):
        # Extract our config from sampler_kwargs BEFORE super().__init__: re-supplied
        # base features (see module docstring) must reach the base constructor, where
        # its own validation (e.g. filter_groups_metric requires refill_fn) applies.
        sk = kwargs.get("sampler_kwargs")
        if sk is None and len(args) >= 5:  # positional call: (trainer_mode, trainer_config,
            sk = args[4]  # max_off_policy_threshold, max_off_policy_strategy, sampler_kwargs)
        cfg = dict(sk.get("durable_buffer", {}) or {}) if sk is not None else {}
        unknown = set(cfg) - _JOURNAL_KEYS - _FORWARDED_BASE_KEYS
        if unknown:
            raise ValueError(
                f"sampler_kwargs.durable_buffer has unknown keys {sorted(unknown)}; "
                f"journal keys: {sorted(_JOURNAL_KEYS)}, re-suppliable base kwargs: "
                f"{sorted(_FORWARDED_BASE_KEYS)}"
            )
        for name in sorted(_FORWARDED_BASE_KEYS & set(cfg)):
            value = cfg.pop(name)
            if name in kwargs and kwargs[name] != value:
                raise ValueError(
                    f"{name} passed both as a constructor kwarg ({kwargs[name]!r}) and via "
                    f"sampler_kwargs.durable_buffer ({value!r}); supply it once"
                )
            kwargs.setdefault(name, value)
        super().__init__(*args, **kwargs)
        self._journal: JournalWriter | None = None
        self._journal_metrics: dict = {}
        self._last_global_steps = 0
        self._dataloader_wired = False
        self._unwired_reported = False
        root = cfg.get("journal_root")
        if root:
            try:
                self._journal = JournalWriter(
                    root=root,
                    interval_s=float(cfg.get("interval_s", 2.0)),
                    ring=int(cfg.get("ring", 3)),
                    metadata_fn=lambda: {"global_steps": self._last_global_steps},
                )
            except JournalWriterLeaseError as exc:
                # A ring root has one writer, and a driver hard-killed inside the stale
                # window still holds a lease whose heartbeat looks fresh — so the
                # crash-restart path this package exists to serve is exactly where
                # construction can be refused. Journaling degrades, the run does not:
                # the rule tick() follows everywhere else. Watch
                # durable_buffer/journaling_enabled.
                logger.warning(
                    "DurableReplayBuffer: JOURNALING IS OFF — the journal ring lease on %s could "
                    "not be taken: %s. Training continues, but a restore will fall back to the "
                    "trainer checkpoint's own TransferQueue snapshot (re-serving every prompt "
                    "dispatched since it was written). After a hard kill the previous driver's "
                    "lease ages out and journaling resumes on the next start; if two drivers share "
                    "this root, give each its own",
                    root,
                    exc,
                )
            else:
                self._wire_dataloader()
                if bool(cfg.get("background", True)):
                    self._journal.start()
                    logger.info(
                        "DurableReplayBuffer: background journal -> %s every %ss", root, cfg.get("interval_s", 2.0)
                    )
                else:
                    logger.info(
                        "DurableReplayBuffer: poll-tick journal -> %s (ticks only while the trainer samples; "
                        "effective cadence is the training-step time, not interval_s)",
                        root,
                    )
        else:
            logger.warning("DurableReplayBuffer: no sampler_kwargs.durable_buffer.journal_root -> journaling OFF")

    def _wire_dataloader(self) -> None:
        """Best-effort: journal the trainer's StatefulDataLoader state with each tick."""
        if self._dataloader_wired or self._journal is None:
            return
        trainer = getattr(self.refill_fn, "__self__", None)
        dataloader = getattr(trainer, "train_dataloader", None)
        if dataloader is not None and hasattr(dataloader, "state_dict"):
            self._journal.dataloader_fn = dataloader.state_dict
            self._dataloader_wired = True
            logger.info("DurableReplayBuffer: journaling dataloader state with each tick")

    def _report_unwired_dataloader(self) -> None:
        """Say so, ONCE, when a journal is running without the loader position.

        The discovery above reads ``refill_fn.__self__`` and then the trainer's
        ``train_dataloader``, so an upstream change — ``refill_fn`` becoming a partial
        or a closure, the attribute moving, the buffer being built with a helper's
        bound method — switches loader journaling off without breaking anything else.
        The symptom is not an error: slots stop carrying ``dataloader_state.pt``, the
        restore falls back to the checkpoint's ``data.pt``, and the run silently
        re-serves every prompt dispatched since the last trainer checkpoint. That is
        discovered after the crash, when the duplicate generation has been paid for, so
        it is reported at the moment it breaks instead — here and, for dashboards, as
        ``durable_buffer/dataloader_journaled`` in the metrics.
        """
        if self._dataloader_wired or self._journal is None or self._unwired_reported:
            return
        self._unwired_reported = True
        logger.warning(
            "DurableReplayBuffer: journaling to %s but the trainer's dataloader could not be "
            "discovered through refill_fn.__self__.train_dataloader, so journal slots carry NO "
            "loader position and a restore will fall back to the checkpoint's data.pt "
            "(re-serving every prompt dispatched since the last trainer checkpoint). Call "
            "attach_dataloader(trainer.train_dataloader) explicitly; watch "
            "durable_buffer/dataloader_journaled",
            self._journal.root,
        )

    def attach_dataloader(self, dataloader) -> None:
        """Explicit alternative to the ``refill_fn`` discovery."""
        if self._journal is not None and dataloader is not None:
            self._journal.dataloader_fn = dataloader.state_dict
            self._dataloader_wired = True

    def _sync_metadata_from_transfer_queue(self):
        out = super()._sync_metadata_from_transfer_queue()
        if self._journal is not None:
            self._wire_dataloader()  # trainer builds the dataloader after the buffer
            self._report_unwired_dataloader()  # ... and says so if it is still missing
            info = self._journal.maybe_tick()  # no-op between intervals; backstop if thread died
            if info is not None or self._journal.stats["ticks"]:
                self._journal_metrics = self._journal.summary()
        return out

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        self._last_global_steps = max(self._last_global_steps, int(global_steps))
        # _BASE_SAMPLE, not super().sample: the base is itself annotate_tq-decorated, so
        # delegating to it would run the skip hook TWICE for one logical sample (see
        # _BASE_SAMPLE's comment for why that is not merely wasteful).
        batch, metrics = _BASE_SAMPLE(self, global_steps, partition_id, batch_size)
        metrics = dict(metrics or {})
        if self._journal_metrics:
            metrics.update({f"durable_buffer/{k}": v for k, v in self._journal_metrics.items()})
        # Always last, so it cannot disagree with the writer's own summary: 0 means a
        # journal_root was configured (or not) and nothing is journaling, which is the
        # only visible symptom of a refused ring lease until the next crash.
        metrics["durable_buffer/journaling_enabled"] = self._journal is not None
        return batch, metrics
