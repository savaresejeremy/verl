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
"""Crash-durable replay buffer for verl's trainer-V1 async data plane.

verl V1 stores every rollout trajectory in TransferQueue (TQ) as it is produced, but
TQ's shipped storage backends are all process-lifetime (in-memory Ray actors or a
remote cache), so a trainer crash loses every finished-but-untrained trajectory. This
package makes the buffer survive process death with **zero verl-core changes**
(registration rides verl's existing ``VERL_USE_EXTERNAL_MODULES`` hook; everything
else is config plus a registered backend):

    transfer_queue:
      backend:
        storage_backend: DurableFS          # this package's backend
        DurableFS:
          root: /shared/fs/runs/<run>/buffer           # shared filesystem (FSx Lustre, NFS, ...)

Pieces
------
* ``fs_kv``   — the storage backend (bootstrap provider + KV client + manager). Writes
  one object per ``(sample, field)`` as it is produced, serialized with TQ's own
  ``serial_utils`` codec, published atomically (tmp + rename), intent-logged in a
  per-writer op log (the default fence source).
* ``journal``  — periodic metadata snapshots: the controller pickle plus the op-log
  offset fence that ``write_manifest`` produces (``tq.save_checkpoint`` with its
  default ``include_storage=True``, which for this backend writes fence artifacts,
  never a data copy). The trajectory-loss window is the journal cadence, not the
  trainer checkpoint interval. Optionally journals the ``StatefulDataLoader`` state,
  shrinking a restart's prompt accounting to two bounded windows around one tick
  (duplicate-dispatch and skip — see the README's guarantees).
* ``fsck``     — restore-time reconciliation between a restored journal and the store:
  fence objects whose *write identity* changed (index reuse!) via op-log replay —
  the ONLY recovery-grade fence source; untrusted artifacts abort fail-closed
  (``mode="mtime"`` exists as an explicit legacy flow, and the full ``(inode,
  size)`` manifest is an offline audit artifact recovery never consumes) — drop
  samples whose values were consumed after the snapshot, and rescue
  complete-but-untagged groups whose expected size is durably known (or
  operator-asserted).
* ``buffer``   — optional ``DurableReplayBuffer`` (a ``ReplayBufferAsync`` subclass)
  that runs the journal alongside the sampler (background thread by default,
  poll-tick mode opt-in).
* ``register`` — import side effects; wire it into every process with
  ``VERL_USE_EXTERNAL_MODULES=verl.experimental.durable_buffer.register`` (verl's
  existing external-modules hook in ``verl/__init__.py``).
* ``restore_tool`` — offline pre-reconciliation so a completely stock verl resume
  picks up the seconds-old journal instead of the checkpoint-old buffer state.

See ``README.md`` for the recovery contract, the failure modes it closes, and the
measured costs.
"""

from .fs_kv import DurableFSClient, DurableFSManager, StoreIntegrityError, bootstrap_durable_fs  # noqa: F401
from .fsck import (  # noqa: F401
    StoreMismatchError,
    UnrecoverableJournalError,
    check_journal_recoverable,
    fsck,
    load_journal,
    promote_complete_groups,
    recover,
)
from .journal import JournalPrecaptureError, JournalWriter, latest_journal  # noqa: F401

__all__ = [
    "DurableFSClient",
    "DurableFSManager",
    "JournalPrecaptureError",
    "JournalWriter",
    "StoreIntegrityError",
    "StoreMismatchError",
    "UnrecoverableJournalError",
    "bootstrap_durable_fs",
    "check_journal_recoverable",
    "fsck",
    "latest_journal",
    "load_journal",
    "promote_complete_groups",
    "recover",
]
