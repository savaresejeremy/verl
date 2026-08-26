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
"""Registration helpers.

Every process that builds a TransferQueue client rebuilds its storage manager *by name*
from the controller-stored config, so the backend module has to be imported in the
driver AND in every rollout worker. Importing this module registers everything::

    import verl.experimental.durable_buffer.register  # noqa: F401

The supported wiring is verl's existing external-modules hook, honored by
``verl/__init__.py`` (so it fires before any verl submodule import, in every
process that imports verl; documented in ``docs/extend_guide.rst``)::

    export VERL_USE_EXTERNAL_MODULES=verl.experimental.durable_buffer.register
    # forwarded into Ray workers via
    #   +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=...

Note the hook imports strictly: a mistyped module name fails ``import verl``
loudly rather than warning. The ``verl.plugins`` entry-point group is the
zero-env-var alternative for a backend distributed as its own package.

Do NOT wire this through Ray's job-level ``worker_process_setup_hook``: in a
multi-GPU cluster run it broke per-actor GPU assignment (every FSDP rank was placed on
the same CUDA device). ``setup_worker`` below is kept only as an explicit hook target for
environments where the env-var path is unavailable.
"""

from __future__ import annotations

import inspect
import logging
import weakref
from typing import Callable

from . import fs_kv  # noqa: F401  (import side effect: registers provider/client/manager)
from .journal import JournalWriter, JournalWriterLeaseError

logger = logging.getLogger(__name__)

_JOURNAL: JournalWriter | None = None


def setup_worker() -> None:
    """Explicit worker hook (prefer VERL_USE_EXTERNAL_MODULES; see module docstring)."""
    from . import fs_kv as _fs  # noqa: F401

    logger.debug("durable_buffer: backend registered in worker process")


def _trainer_step_metadata_fn(trainer) -> Callable[[], dict] | None:
    """A ``metadata_fn`` reporting ``trainer.global_steps``, or ``None``.

    Held by weak reference: the journal is process-wide and outlives individual calls,
    and it must not be the reason the trainer stays alive. A dead referent reports no
    step rather than raising, so a late tick cannot kill journaling.
    """
    if trainer is None:
        return None
    ref = weakref.ref(trainer)

    def metadata_fn() -> dict:
        obj = ref()
        step = getattr(obj, "global_steps", None) if obj is not None else None
        return {"global_steps": int(step)} if isinstance(step, int) and not isinstance(step, bool) else {}

    return metadata_fn


def _discover_trainer(depth: int = 2):
    """The caller's ``self`` when it looks like a verl trainer, else ``None``.

    ``start_journal`` is called from trainer code, so the trainer is normally the
    caller's ``self``. Recording its step matters because the alternatives are worse:
    slot ranking and the journal-versus-checkpoint freshness comparison both prefer
    ``global_steps`` and fall back to a wall-clock timestamp, and with the recommended
    wiring passing no ``metadata_fn`` that fallback was the NORMAL case — the
    recommended path was the weakest one.

    Deliberately narrow: only the immediate caller's frame is inspected, and the
    candidate is accepted only when it carries both an integer ``global_steps`` and a
    ``train_dataloader``, which together identify a verl trainer and nothing else. Any
    miss just leaves the step unrecorded (with a warning), and passing ``trainer=`` or
    ``metadata_fn=`` explicitly skips discovery entirely.
    """
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            frame = frame.f_back if frame is not None else None
        candidate = frame.f_locals.get("self") if frame is not None else None
        if candidate is None or not hasattr(candidate, "train_dataloader"):
            return None
        step = getattr(candidate, "global_steps", None)
        return candidate if isinstance(step, int) and not isinstance(step, bool) else None
    finally:
        del frame


def start_journal(
    root: str, interval_s: float = 2.0, dataloader=None, metadata_fn=None, trainer=None
) -> JournalWriter | None:
    """Start (or return) the process-wide background journal writer.

    ``metadata_fn`` overrides everything. Otherwise the trainer's step is recorded:
    from ``trainer`` when given, else from the calling trainer if it can be identified
    (:func:`_discover_trainer`). A run whose journal records no step falls back to
    comparing wall-clock timestamps, so failing to find one is reported.

    Returns ``None`` when the ring lease is held by another writer. A ring root has one
    writer, and a driver hard-killed less than ``max(30s, interval_s * 5)`` ago still
    holds a lease whose heartbeat looks fresh, so the restart this package exists to
    serve is exactly when construction can be refused. Journaling is the thing that
    degrades there, never the run: the same rule ``tick`` follows. The ``None`` and the
    warning are the whole signal from this wiring; the metrics key
    ``durable_buffer/journaling_enabled`` comes from :class:`DurableReplayBuffer`, which
    is the only part of this package verl merges metrics from.
    """
    global _JOURNAL
    if _JOURNAL is None:
        if metadata_fn is None:
            metadata_fn = _trainer_step_metadata_fn(trainer if trainer is not None else _discover_trainer())
            if metadata_fn is None:
                logger.warning(
                    "durable_buffer: start_journal found no trainer to read global_steps from, so "
                    "journal slots will carry no training step; slot ranking and the "
                    "journal-versus-checkpoint freshness comparison then fall back to wall-clock "
                    "timestamps. Pass trainer=<the trainer> or metadata_fn=lambda: "
                    "{'global_steps': ...} to record the step"
                )
        try:
            _JOURNAL = JournalWriter(
                root=root,
                interval_s=interval_s,
                dataloader_fn=(dataloader.state_dict if dataloader is not None else None),
                metadata_fn=metadata_fn,
            )
        except JournalWriterLeaseError as exc:
            logger.warning(
                "durable_buffer: JOURNALING IS OFF for this process — the journal ring lease on %s "
                "could not be taken: %s. A restore will fall back to the trainer checkpoint's own "
                "TransferQueue snapshot (re-serving every prompt dispatched since it was written). "
                "If this is a restart after a hard kill, the previous driver's lease is stale and "
                "journaling resumes on the next start once it ages out; if two drivers share this "
                "root, give each its own",
                root,
                exc,
            )
            return None
        _JOURNAL.start()
    return _JOURNAL


def stop_journal() -> None:
    global _JOURNAL
    if _JOURNAL is not None:
        _JOURNAL.stop()
        _JOURNAL = None


def journal() -> JournalWriter | None:
    return _JOURNAL
