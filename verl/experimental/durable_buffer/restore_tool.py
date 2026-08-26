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
"""Offline pre-reconciliation so an *unmodified* verl resume picks up the journal.

verl's resume path loads the TransferQueue snapshot stored inside the trainer
checkpoint (``global_step_N/transfer_queue``) and then re-issues in-flight prompts.
That snapshot is ``trainer.save_freq`` old; the journal is seconds old. Nothing in
verl knows about the journal — and nothing has to. Run this tool between the crash
and the restart::

    python -m verl.experimental.durable_buffer.restore_tool \
        --store   /shared/fs/runs/<run>/buffer \
        --journal /shared/fs/runs/<run>/journal \
        --ckpt-dir /shared/fs/runs/<run>/ckpts          # trainer.default_local_dir

The tool runs in two phases:

**PHASE 0 — store exclusivity.** Reconciliation deletes store objects and rewrites
controller metadata, so it is only correct against a buffer whose writers and
controller are gone. Before anything else the tool probes for a reachable
``TransferQueueController`` actor and refuses with status ``live_controller_present``
if it finds one — in ``--dry-run`` too, which is not a read-only mode as far as a live
controller is concerned: ``tq.init`` ignores its ``create=False`` config whenever a
controller is reachable (``interface.py`` returns at ``if _init_from_existing()``
before the config is used) and ``recover(..., dry_run=True)`` still calls
``tq.load_checkpoint``, so a "preview" against a live job would replace that job's
buffer index. Actor names are scoped to a GCS, so WHERE the probe runs decides whether
it can see anything: it runs in the caller's Ray when there is one, else in a temporary
attachment to EVERY Ray instance this host can discover, one at a time, refusing if any
of them holds the controller. Every instance and not one: Ray's own resolution
(``RAY_ADDRESS``, then ``/tmp/ray/ray_current_cluster``, then a local ``gcs_server``)
returns a single address and its last step picks an arbitrary local ``gcs_server``, so on
a host running a second Ray the probe would ask the wrong GCS and find nothing there,
which reads as a confident absence rather than as doubt. The enumeration takes the union
of all three sources instead, and each attachment is detached again before the reconcile.
Discovery names POINTERS and not live clusters: the first two of those outlive the
cluster they describe, so a crash leaves an address nothing answers at. That case is
decided by one TCP connection (2 s) and counts as no cluster, because ``ray.init`` takes
820 s to fail on a dead address and then reports doubt, which would refuse the restore the
operator is running BECAUSE of the crash. The reconcile itself then runs in a PRIVATE Ray
instance of the tool's own, rather than letting ``ray.get_actor``'s auto-init join
whatever ``RAY_ADDRESS`` names, so its scratch TransferQueue can never bind a cluster
controller. What the refusal does NOT cover is a run on a cluster this host cannot
discover (a head node elsewhere with no pointer here, or a raylet this user cannot read
in ``ps``) or cannot reach: an address that does not answer a TCP connection within
``CLUSTER_CONNECT_TIMEOUT_S`` reads as no cluster, so the refusal does not apply and the
tool proceeds, whether the cluster behind that address is dead or merely unreachable from
here. Nor does it cover a writer publishing into the store from elsewhere: there is no
store-level writer lease, so exclusivity of the STORE is assumed and only the controller
half is checked. ``--allow-live-controller`` overrides the refusal (DANGEROUS).

**PHASE 1 — read-only PREFLIGHT.** From PLAIN FILE READS (no TQ, no writes of any
kind — not even its own healing/transaction-completion, and no ``tq.init``, whose
bootstrap would otherwise stamp an unstamped store), it validates EVERYTHING and
computes the complete plan:

1. store identity: the driving snapshot (a journal ring slot, else the checkpoint's
   live TQ snapshot, else the healable ``*.pre_reconcile`` backup of it) must
   positively match ``--store``. Missing/empty/invalid identities on either side abort
   like a mismatch and are NEVER overridable; ``--force-store-mismatch`` covers ONLY
   two valid-but-different identities (identity is the store's logical lineage — a
   copied store keeps it — and this is separate from the freshness ``--force``);
2. recovery-grade fence artifacts: an ``oplog: false`` snapshot is refused, a slot
   assembled from two different journal ticks is refused, a controller-state schema
   this code does not read is refused, the offsets fence is schema-validated, and the
   op-log tail is fully parsed and name-validated — all BEFORE anything reconciles, so
   an untrusted artifact can never be discovered after a mutation. Validation runs per
   ring slot, newest first: a poisoned newest slot falls back to the next-older
   complete slot and the result reports which slot was used and why the newer ones were
   refused. ``--journal-slot`` pins the choice to one slot instead;
3. pending-transaction state: whether an interrupted publication must be healed
   (``transfer_queue`` restored from its backup) and whether the ``reconcile.state``
   marker can be completed (proof: transaction id AND store lineage match AND the
   marker's recorded source path is provably a real directory under the allowed
   journal/checkpoint roots — see below) or must be discarded;
4. freshness: the selected journal slot is used only when *strictly fresher*
   (``global_steps``, then ``fence_ts``) than the checkpoint's snapshot — judged
   against the post-heal view. Otherwise the journal is not applied and the tool
   reports ``checkpoint_fresher``/``ordering_ambiguous`` (``--force`` overrides).

The resulting plan is IMMUTABLE and shared verbatim by both modes: ``--dry-run``
prints it and exits (ZERO writes of any kind — byte-identical store and checkpoint
trees), the real mode executes exactly it (post-rename path fixups live in an
execution-local copy; the report never drifts from the preflight view). Every
result — every status, both modes — carries ONE stable schema: ``status``,
``dry_run``, ``checkpoint_step_dir``, ``plan_available``, ``checkpoint_tq_present``
and the full plan (pending transaction, would-heal, completion decision, source
selection, source journal, ring-slot request/candidates/refusals, freshness evidence,
detail); statuses that never ran a preflight (``no_checkpoint``, ``locked``) mark the
plan explicitly unavailable instead of omitting fields. The real mode additionally
reports what was DONE (``healed_publication``,
``completed_interrupted_transaction``, ``discarded_transaction``).
``checkpoint_tq_present: false`` forces a non-zero exit whatever the status says,
because verl's resume skips a missing TransferQueue snapshot silently.

**PHASE 2 — guarded EXECUTE.** The entire mutating phase runs under an EXCLUSIVE
per-checkpoint lockfile (``reconcile.lock`` in the step dir, ``O_CREAT|O_EXCL``,
content: uuid + host + pid + timestamp). A second restorer fails fast with status
``locked`` and the holder's identity; it never interleaves markers, backups, TQ
swaps or ``data.pt`` halves with the holder. STALE-LOCK POLICY (conservative, by
design): the tool NEVER breaks a lock. After a hard kill the file survives; the
operator verifies the recorded host/pid is dead and removes the file manually.
Auto-breaking on pid-liveness would be wrong on a shared filesystem (pid reuse,
other hosts). The preflight itself runs under the lock too, so the plan cannot go
stale between computation and execution. Under the lock, the real mode:

1. heals an interrupted swap (backup renamed back into place). This is NOT gated on
   journal-artifact trust: the checkpoint has no live ``transfer_queue`` at all in that
   state, putting it back is a rename of a backup this tool created, and refusing it
   because a journal slot is corrupt would leave verl to resume with an empty buffer and
   no warning. A refused source therefore yields status ``healed_only`` — healed,
   nothing reconciled, non-zero exit — rather than a refusal that also declines the
   repair;
2. completes or discards the pending transaction — completion only on PROOF: the
   live TQ dir's transaction id matches the marker's (the swap became visible), the
   marker's recorded source path provably lives under the allowed
   journal/checkpoint roots (every component a real non-symlink directory — no
   identity/fence/dataloader artifact is read through an unproven path, and the
   artifacts themselves are lstat-validated at use), the marker's recorded source
   digest still matches the journal slot (a recycled ring slot must not pair one
   tick's TQ state with another tick's ``data.pt``), AND the transaction's store
   lineage matches the CURRENT store (a marker + published dir left behind by a
   deleted store must not complete against a recreated store of a different
   lineage — cross-lineage and unprovable-source transactions are discarded with a
   report);
3. brings up a scratch TransferQueue against the EXISTING store (``create=False``:
   attach-only, never mkdir/stamp), loads the source snapshot, runs ``fsck``
   (fence/drop, reusing the preflight's validated op-log tail) and — only under the
   ``--group-size`` operator assertion — the group-completeness rescue;
4. publishes the reconciled state into the checkpoint inside a transaction
   (``reconcile.state``: BEGIN -> TQ_PUBLISHED -> DONE, same-dir tmp + rename),
   BOUND to its artifacts: a uuid transaction id and the CURRENT store lineage id
   are stamped into the marker AND the published TQ dir's metadata, and the marker
   records the source journal's content digest. The staging dir carries the
   transaction id in its name, ownership is re-validated after the swap and before
   the ``data.pt`` half + DONE, and the original dirs are kept as
   ``*.pre_reconcile`` backups (a pre-existing TQ backup from an earlier reconcile
   is replaced by the current live dir; the first ``data.pt`` backup is kept);
5. sweeps TQ's own checkpoint-protocol leftovers around the live dir
   (``transfer_queue.old`` / ``transfer_queue.tmp``, TQ#155) between building the
   reconciled snapshot and publishing it, while the live dir still exists. Left in
   place, a ``.old`` is what ``tq.load_checkpoint`` rolls forward from whenever the
   live dir is missing (``interface.py:1086-1092``) — including the moment inside the
   publication swap when it is missing by design, so the sibling has to be gone before
   the swap starts or a crash there resumes the job from PRE-reconcile buffer state.
   With the live dir absent the siblings are never touched, because after a crash
   mid-save the ``.old`` can be the only copy of the snapshot; the preflight only
   REPORTS them (``tq_protocol_siblings``). A symlinked sibling is refused outright
   with ``invalid_checkpoint_layout``, before any lock.

The store-side fsck mutations (fencing post-journal objects, dropping consumed
samples) happen before the metadata publication and are idempotent: re-running the
tool against the same journal converges to the same state.

The training job itself needs only the durable backend config; resume behaviour,
re-issue, staleness and refill all stay verl-stock.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import stat
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

LATEST_FILE = "latest_checkpointed_iteration.txt"
PRE_RECONCILE_SUFFIX = ".pre_reconcile"
RECONCILE_STATE_FILE = "reconcile.state"
RESTORE_LOCK_FILE = "reconcile.lock"
_RECONCILED_BY = "durable_buffer.restore_tool"

#: Suffixes ``tq.save_checkpoint`` gives the SIBLINGS of a checkpoint dir while it swaps
#: a new snapshot into place (TQ#155, ``transfer_queue/interface.py:1008-1052``): the new
#: state is built in ``<dir>.tmp``, any existing ``<dir>`` is renamed aside to
#: ``<dir>.old``, the tmp is renamed into place and the ``.old`` is deleted. Both are
#: transient on a clean save and both can survive a crash. They matter here because
#: ``tq.load_checkpoint`` ROLLS FORWARD from ``<dir>.old`` when ``<dir>`` is missing
#: (``interface.py:1086-1092``).
TQ_PROTOCOL_SUFFIXES = (".old", ".tmp")


def _tq_protocol_siblings(ckpt_tq: Path) -> list[Path]:
    """Existing ``transfer_queue.old`` / ``transfer_queue.tmp`` siblings of the live dir."""
    return [p for p in (ckpt_tq.parent / (ckpt_tq.name + sfx) for sfx in TQ_PROTOCOL_SUFFIXES) if p.exists()]


# Test seams (fault-injection points for the publication transaction, in order):
#   _PRE_BEGIN_CRASH_HOOK        before the BEGIN marker is written
#   _POST_BEGIN_CRASH_HOOK       after BEGIN, before the reconciled snapshot is built/published
#   _PUBLISH_CRASH_HOOK          inside the swap: live dir moved to backup, replacement not yet in
#   _POST_TQ_PUBLISH_CRASH_HOOK  after the TQ dir swap, before the data.pt half
#   _PRE_DONE_CRASH_HOOK         after the data.pt half, before the DONE marker
_PRE_BEGIN_CRASH_HOOK = lambda: None  # noqa: E731
_POST_BEGIN_CRASH_HOOK = lambda: None  # noqa: E731
_PUBLISH_CRASH_HOOK = lambda: None  # noqa: E731
_POST_TQ_PUBLISH_CRASH_HOOK = lambda: None  # noqa: E731
_PRE_DONE_CRASH_HOOK = lambda: None  # noqa: E731


#: TransferQueue hardcodes its controller's actor name AND Ray namespace
#: (``ray.get_actor("TransferQueueController", namespace="transfer_queue")``,
#: ``transfer_queue/interface.py``), on both the attach and the create path.
TQ_CONTROLLER_ACTOR = "TransferQueueController"
TQ_CONTROLLER_NAMESPACE = "transfer_queue"
#: Ray namespace for the Ray instance this tool starts for itself (see
#: :func:`_ensure_isolated_ray`).
TOOL_RAY_NAMESPACE = "durable_buffer.restore_tool"
#: How long the liveness probe waits for the controller to answer before calling it
#: unresponsive (which counts as PRESENT: a wedged controller is exactly the
#: partial-failure case this refusal exists for).
CONTROLLER_PROBE_TIMEOUT_S = 5.0
#: How long the pre-check waits for a TCP connection to a DISCOVERED cluster address
#: before calling it dead (see :func:`_address_answers`). Small on purpose: it runs
#: against an address Ray's discovery just named, so anything alive answers at once.
CLUSTER_CONNECT_TIMEOUT_S = 2.0
#: GCS connect attempts left to ``ray.init`` while the tool attaches to a discovered
#: cluster (Ray's default is 20 and each attempt costs ~41 s). One retry, because the
#: pre-check has already shown that something accepts connections at that address.
ATTACH_GCS_CONNECT_ATTEMPTS = 2

#: ``controller_probe["outcome"]`` in every :func:`restore` result: WHERE the exclusivity
#: probe ran, or why it did not run. The refusal detail branches on these, so they are
#: constants rather than prose.
PROBE_IN_CALLER_RAY = "probed the Ray this process already had"
PROBE_IN_DISCOVERED_CLUSTER = "probed every discoverable cluster that answered"
PROBE_NO_CLUSTER = "no cluster address discovered, so no controller can be reachable"
PROBE_ADDRESS_IS_DEAD = "every discovered address answers nothing, so they are stale pointers"
PROBE_DISCOVERY_FAILED = "Ray's own cluster discovery failed"
PROBE_CLUSTER_UNJOINABLE = "a discovered cluster could not be joined"
#: The two ``reachable`` values that mean "the probe SAW it": ``True``, or an actor whose
#: handle was obtained and did not answer in time. Everything else is doubt, refused under
#: the same fail-closed rule but given its own refusal text, because "stop the job" is the
#: wrong instruction for a job that may not exist (:func:`_live_controller_detail`).
_REACHABLE_UNRESPONSIVE = "unresponsive"
#: Prefix of every ``reachable`` value that means "could not decide"
#: (:func:`_probe_controller`, :func:`_live_controller`).
_REACHABLE_UNKNOWN = "unknown"


def _ensure_isolated_ray() -> bool:
    """Give the tool a Ray of its own when it has none. Returns whether one was started.

    ``tq.init`` begins with ``if _init_from_existing(): return conf`` and
    ``_init_from_existing`` calls ``ray.get_actor``, so the ``create=False`` config this
    tool passes is IGNORED whenever a controller is reachable and the tool's handle
    becomes the running job's controller. Ray also auto-initializes on ``ray.get_actor``
    (``ray/_private/auto_init_hook.py``), so on a node with ``RAY_ADDRESS`` set the
    first TQ call joins the production cluster — inside ``--dry-run`` too, whose whole
    promise is that it writes nothing.

    TQ pins its namespace, so a namespace of our own cannot be handed to it: the
    isolation has to be the Ray INSTANCE. ``ray.init(address="local")`` starts a private
    one even when ``RAY_ADDRESS`` is set (that variable only overrides ``None`` and
    ``"auto"``), so the scratch controller the tool brings up cannot bind the job's, and
    Ray is joined deliberately rather than as a side effect.

    Runs AFTER the exclusivity probe, never before it: actor names are per-GCS, so a
    ``get_actor`` inside this private instance holds nothing by construction and would
    report absence whatever the cluster is doing. :func:`_live_controller` therefore
    probes the caller's Ray, or a temporary attachment to the discoverable cluster,
    while this function is what the reconcile itself runs inside. An already-initialized
    Ray is left alone: the caller owns it, and the probe has already run inside it.
    """
    import ray

    if ray.is_initialized():
        return False
    ray.init(address="local", namespace=TOOL_RAY_NAMESPACE, include_dashboard=False, log_to_driver=False)
    logger.info(
        "restore_tool: started a private Ray instance (namespace %s) so the scratch "
        "TransferQueue cannot attach to a cluster controller",
        TOOL_RAY_NAMESPACE,
    )
    return True


class _ClusterDiscoveryFailed(RuntimeError):
    """Ray's own address discovery broke, so cluster presence cannot be decided."""


def _discoverable_cluster_addresses() -> tuple[str, ...]:
    """EVERY ``host:port`` a Ray cluster this process could join is reachable at, deduped
    and in Ray's own preference order. Empty when nothing is discoverable. Raises
    :class:`_ClusterDiscoveryFailed` when the enumeration itself cannot be trusted.

    The three sources are Ray's, not ours: ``RAY_ADDRESS``, then the address of the last
    ``ray start`` in ``/tmp/ray/ray_current_cluster``, then every local ``gcs_server``
    process (``services.find_gcs_addresses``, which is how a cluster brought up by a
    driver rather than by ``ray start`` is found). What is ours is taking the UNION
    instead of the first hit. ``canonicalize_bootstrap_address("auto")`` returns ONE
    address, and its third step picks an arbitrary local ``gcs_server``, so on a host
    running a second Ray instance it can name a cluster that is not the run's; the
    controller is then absent from the GCS the probe asks, which reads as a confident
    absence rather than as doubt and lets the reconcile delete objects under a live job.
    Probing all of them costs one extra bounded attach per instance and removes that case
    (:func:`_live_controller` refuses if ANY of them holds the controller).

    ``RAY_ADDRESS=auto`` is SKIPPED rather than resolved, like ``local``. It is not an
    address but a request to run this same resolution: canonicalizing it re-enters
    ``get_ray_address_from_environment``, which reads the address file and the local
    ``gcs_server`` scan (the two sources this function already reads) and raises
    ``ConnectionError`` when neither has anything. Resolving it would therefore turn
    ``auto`` exported on a head node whose cluster is gone, which is the state this tool
    exists for, into a discovery failure, and doubt is refused. Nothing is lost by
    skipping: whatever ``auto`` would have named is a member of those two sources, and
    both are enumerated on their own account.

    Discovery only LOOKS (it opens no connection), so the common nothing-is-running case
    costs a file read and a process scan and cannot block on a dead GCS. What it cannot
    see is an instance with no pointer on this host and no local ``gcs_server``: a head
    node elsewhere, or a raylet this user cannot read in ``ps`` (Ray's scan swallows
    ``psutil.AccessDenied``). That residual is reachability from this host, which is the
    documented boundary of the refusal.
    """
    try:
        from ray._private import services

        # find_gcs_addresses caches a non-empty scan for the process lifetime, and a
        # stale hit would be the one direction that matters: an instance started since
        # the cache was filled would be missing from the enumeration.
        cached = getattr(services, "_cached_find_gcs_addresses", None)
        if cached is not None and hasattr(cached, "cache_clear"):
            cached.cache_clear()
        raw = [
            os.environ.get("RAY_ADDRESS") or None,
            services.find_bootstrap_address(None) or None,
            *sorted(services.find_gcs_addresses()),
        ]
        found: list[str] = []
        for candidate in raw:
            # "local" names an instance to START, and "auto" is a REQUEST to run this same
            # resolution, not an address: canonicalizing it re-enters
            # get_ray_address_from_environment, which reads the two sources below this line
            # and raises ConnectionError when neither has anything. Skipping it loses
            # nothing (its answer is always one of those sources, which are enumerated
            # directly) and keeps RAY_ADDRESS=auto on a host whose cluster is gone as an
            # empty enumeration rather than doubt, which is the crash path itself.
            if not candidate or candidate in ("local", "auto"):
                continue
            address = services.canonicalize_bootstrap_address(candidate)
            if address and address not in found:
                found.append(address)
        return tuple(found)
    except Exception as exc:  # noqa: BLE001 - discovery moved or broke: prove nothing
        raise _ClusterDiscoveryFailed(f"{type(exc).__name__}: {exc}") from exc


def _address_answers(address: str) -> bool | None:
    """Whether anything accepts a TCP connection at ``host:port`` within
    :data:`CLUSTER_CONNECT_TIMEOUT_S`. ``None`` when the address cannot be parsed.

    Discovery reports a POINTER, not a cluster. ``/tmp/ray/ray_current_cluster`` is
    written by ``ray start`` and removed by nothing (in Ray 2.56.1 the only writers are
    ``scripts.py`` and ``cluster_utils.py``; there is no removal call site), and an
    exported ``RAY_ADDRESS`` outlives the head it names. So the state this tool exists
    for, a cluster that died, is exactly the state in which discovery hands back an
    address nothing answers at, and ``ray.init`` is the wrong thing to learn that from:
    it retries the GCS connect ``ray_constants.NUM_REDIS_GET_RETRIES`` (20) times at
    about 41 s each, so a dead pointer costs 820 s (measured) before a verdict of
    "unknown". One refused connection costs microseconds and says the same thing.

    A dead address means no live cluster HERE, which is the same answer as no address at
    all: there is nothing to hold a controller, so the caller proceeds to isolation
    rather than refusing. The reverse is not implied, because something listening is not
    necessarily a healthy GCS, which is why the attach that follows is bounded too
    (:func:`_bounded_gcs_connect`).

    Ray canonicalizes to ``ip:port``, so the parse is a right split. A hostname would
    put its DNS lookup outside this timeout (``getaddrinfo`` is not bounded by it), and
    an address this cannot parse is not judged at all: ``None`` leaves the decision to
    the bounded attach instead of guessing.
    """
    host, _, port = address.rpartition(":")
    if not host or not port.isdigit():
        return None
    try:
        with socket.create_connection((host.strip("[]"), int(port)), timeout=CLUSTER_CONNECT_TIMEOUT_S):
            return True
    except OSError:
        # Refused, unreachable, timed out, or unresolvable: nothing is serving there.
        return False


@contextmanager
def _bounded_gcs_connect():
    """Cap ``ray.init``'s GCS connect retries for the duration of one attach.

    ``Node._init_gcs_client`` loops ``ray_constants.NUM_REDIS_GET_RETRIES`` times
    (``node.py:746``), reading the module attribute at call time, and each attempt
    against an address that is not a live GCS costs about 41 s: 5 s connect, 5 s
    observability pubsub, then a 30 s "waiting for GCS to become available". Twenty of
    those is the 820 s stall. Lowering the count is the only safe way to shorten it:
    Ray's C++ timeouts (``RAY_gcs_rpc_server_reconnect_timeout_s`` and friends) make the
    core ABORT the process instead ("The program will terminate", verified), which would
    leave the operator with no report at all.

    Cheap insurance rather than the main defence: :func:`_address_answers` already
    rejects an address nothing listens on. This bounds the rarer case of a port that
    accepts connections without being a healthy GCS (a reused port, a tunnel, a wedged
    head), where the verdict is still doubt but arrives in ~82 s instead of 820 s. Ray
    renaming the constant costs the bound, not the run.
    """
    try:
        from ray._private import ray_constants

        original = ray_constants.NUM_REDIS_GET_RETRIES
    except (ImportError, AttributeError):  # pragma: no cover - Ray moved the constant
        yield
        return
    ray_constants.NUM_REDIS_GET_RETRIES = ATTACH_GCS_CONNECT_ATTEMPTS
    try:
        yield
    finally:
        ray_constants.NUM_REDIS_GET_RETRIES = original


def _probe_controller() -> dict | None:
    """The controller probe itself, inside whatever Ray this process is attached to.

    FAIL CLOSED on doubt: a named actor that does not answer within
    :data:`CONTROLLER_PROBE_TIMEOUT_S`, or that cannot be probed at all, counts as
    present. Only a genuinely absent name, or a named actor that reports itself dead,
    counts as absent — which is the state after the total crash this tool is for.
    """
    import ray

    try:
        handle = ray.get_actor(TQ_CONTROLLER_ACTOR, namespace=TQ_CONTROLLER_NAMESPACE)
    except ValueError:
        return None  # no such actor: the job's controller is gone (the intended path)
    except Exception as exc:  # noqa: BLE001 - cannot prove absence -> assume presence
        return {
            "actor": TQ_CONTROLLER_ACTOR,
            "namespace": TQ_CONTROLLER_NAMESPACE,
            "reachable": f"{_REACHABLE_UNKNOWN} ({exc})",
        }
    found = {"actor": TQ_CONTROLLER_ACTOR, "namespace": TQ_CONTROLLER_NAMESPACE}
    probe = getattr(handle, "get_config", None)
    if probe is None:  # pragma: no cover - TQ renamed its config accessor
        found["reachable"] = f"{_REACHABLE_UNKNOWN} (the controller exposes no get_config)"
        return found
    try:
        ray.get(probe.remote(), timeout=CONTROLLER_PROBE_TIMEOUT_S)
    except ray.exceptions.RayActorError:
        # The name has not been reaped yet but the actor is dead (ray.kill is what
        # tq.close() does). Nothing to protect.
        return None
    except Exception as exc:  # noqa: BLE001 - alive but not answering: still a live job
        found["reachable"] = f"{_REACHABLE_UNRESPONSIVE} ({type(exc).__name__}: {exc})"
        return found
    found["reachable"] = True
    return found


def _presence_is_a_sighting(found: dict) -> bool:
    """Whether the probe SAW the controller, as opposed to failing to rule it out.

    ``reachable is True`` and ``unresponsive (...)`` are sightings: a handle to the named
    actor was obtained, so a job owns that buffer. Everything else, including a shape this
    does not recognise, is treated as doubt, which is refused just the same but never
    described to the operator as a controller that answered.
    """
    reachable = found.get("reachable")
    return reachable is True or str(reachable).startswith(_REACHABLE_UNRESPONSIVE)


def _live_controller(probe_record: dict | None = None) -> dict | None:
    """Identity of a REACHABLE TransferQueue controller, or ``None`` when there is none.

    Reconciliation deletes store objects and rewrites controller metadata, so it is
    only correct against a buffer whose writers and controller are gone. The tool's
    per-checkpoint lock serializes two restorers; it says nothing about a live run. And
    the failure is not hypothetical: with a controller reachable, ``tq.init`` attaches
    to it (see :func:`_ensure_isolated_ray`) and ``recover(..., dry_run=True)`` still
    calls ``tq.load_checkpoint``, so a preview run against a live job replaces its
    controller's metadata with a snapshot and destroys the buffer index it was only
    supposed to describe.

    WHERE the probe runs is the whole difficulty, because actor names are scoped to a
    GCS. Two paths:

    * an initialized Ray belongs to the caller (the in-process ``recover`` path), and it
      is the Ray that caller's TransferQueue lives in, so the probe runs in it directly;
    * with no Ray at all (every CLI invocation), the tool enumerates EVERY Ray instance
      it can discover locally (:func:`_discoverable_cluster_addresses`), attaches to each
      in turn, probes it and detaches, and refuses if ANY of them holds the controller.
      One address is not enough: Ray's own resolution returns a single one and its last
      step picks an arbitrary local ``gcs_server``, so a host running a second Ray would
      be asked about the wrong GCS and report a confident absence. Probing after
      :func:`_ensure_isolated_ray` instead would be pointless by construction: the
      private instance the tool starts for itself holds no actors, so the answer would be
      "absent" whatever the cluster is doing.

    Discovery finding an address does NOT mean a cluster is there. It reads
    ``RAY_ADDRESS`` and a file ``ray start`` leaves behind, both of which outlive the
    cluster they describe, so the post-crash state this tool runs in is exactly when the
    address is stale. A dead pointer is therefore pre-checked with one TCP connection
    (:func:`_address_answers`) and treated as no cluster at all, and the attach that
    follows a successful connection is bounded (:func:`_bounded_gcs_connect`). Without
    that, ``ray.init`` spends 820 s (measured) failing to join a cluster that does not
    exist and then reports doubt, which fails closed into a refusal: a fourteen-minute
    stall and a wrong verdict on the one path that matters.

    ``probe_record``, when given, is filled in with what the probe did: its ``outcome``
    (one of the ``PROBE_*`` constants), the list of addresses discovery named and, per
    address, whether anything answered there. :func:`restore` carries it into every
    result as ``controller_probe``, because "no controller found" and "no cluster found"
    are different facts and an operator reading a report has no other way to tell them
    apart.

    What it still does not see: a run whose cluster this host cannot discover or reach.
    A head node elsewhere with no pointer here leaves nothing to enumerate, and an
    address that does not answer within :data:`CLUSTER_CONNECT_TIMEOUT_S` reads as a
    stale pointer whether the cluster behind it is dead or merely unreachable. Store
    exclusivity is not otherwise checked (there is no store-level writer lease), so this
    refusal is the controller half of that precondition and not the whole of it; the
    README says so.
    """
    import ray

    record = probe_record if probe_record is not None else {}
    if ray.is_initialized():
        record["outcome"] = PROBE_IN_CALLER_RAY
        return _probe_controller()
    try:
        addresses = _discoverable_cluster_addresses()
    except _ClusterDiscoveryFailed as exc:
        record["outcome"] = PROBE_DISCOVERY_FAILED
        return {
            "actor": TQ_CONTROLLER_ACTOR,
            "namespace": TQ_CONTROLLER_NAMESPACE,
            "reachable": f"{_REACHABLE_UNKNOWN} (Ray cluster discovery failed: {exc})",
        }
    record["discovered"] = list(addresses)
    record["answered"] = {}
    if not addresses:
        record["outcome"] = PROBE_NO_CLUSTER
        return None  # no cluster to hold a controller
    dead: list[str] = []
    unjoinable: list[str] = []
    probed: list[str] = []
    for address in addresses:
        answered = _address_answers(address)
        record["answered"][address] = answered
        if answered is False:
            dead.append(address)
            continue
        try:
            with _bounded_gcs_connect():
                ray.init(address=address, namespace=TOOL_RAY_NAMESPACE, include_dashboard=False, log_to_driver=False)
        except Exception as exc:  # noqa: BLE001 - a cluster we cannot join, we cannot clear
            unjoinable.append(f"{address} ({type(exc).__name__}: {exc})")
            continue
        try:
            found = _probe_controller()
        finally:
            # Detach: the reconcile runs in a private instance of the tool's own
            # (_ensure_isolated_ray), never in a cluster it just probed. Detaching here
            # also frees this process to attach to the next candidate.
            ray.shutdown()
        probed.append(address)
        if found is not None:
            # A sighting anywhere is the whole answer: one instance holding the
            # controller is one live run whose objects this reconcile would delete.
            record["outcome"] = PROBE_IN_DISCOVERED_CLUSTER
            found["cluster"] = address
            return found
    if unjoinable:
        # Doubt outranks the instances that came back clean: an unjoined cluster could be
        # holding the controller and there is no way to tell from here.
        record["outcome"] = PROBE_CLUSTER_UNJOINABLE
        return {
            "actor": TQ_CONTROLLER_ACTOR,
            "namespace": TQ_CONTROLLER_NAMESPACE,
            "reachable": (
                f"{_REACHABLE_UNKNOWN} (cannot join the discoverable Ray "
                f"cluster{'s' if len(unjoinable) > 1 else ''} at {', '.join(unjoinable)})"
            ),
        }
    if probed:
        record["outcome"] = PROBE_IN_DISCOVERED_CLUSTER
        return None
    record["outcome"] = PROBE_ADDRESS_IS_DEAD
    logger.warning(
        "restore_tool: Ray's discovery names %s at %s but nothing accepted a TCP connection there "
        "within %.0fs. Treating it as no live cluster and proceeding. That is the right call for a "
        "pointer left behind by a dead cluster (a stale RAY_ADDRESS, or /tmp/ray/ray_current_cluster "
        "after a crash), and the wrong one for a cluster that is up but unreachable from this host "
        "within the timeout; TCP reachability from this host is the only evidence consulted. If the "
        "run's cluster may be alive elsewhere, stop before real mode",
        "a cluster" if len(dead) == 1 else "clusters",
        ", ".join(dead),
        CLUSTER_CONNECT_TIMEOUT_S,
    )
    return None


class _RestoreLockHeld(RuntimeError):
    """Another restorer holds this checkpoint's mutation lock (fail fast)."""

    def __init__(self, path: Path, holder: dict | None):
        self.path = path
        self.holder = holder
        super().__init__(
            f"another restore holds the exclusive checkpoint lock {path} "
            f"(holder: {holder}); refusing to run two mutating restores against one "
            "checkpoint. If the holder crashed hard (the lock records its uuid/host/"
            "pid/timestamp), verify that process is dead and remove the lock file "
            "manually — the tool never breaks a lock itself (conservative stale-lock "
            "policy: auto-breaking would be wrong on shared filesystems)."
        )


def _read_lock_holder(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@contextmanager
def _exclusive_checkpoint_lock(step_dir: Path):
    """EXCLUSIVE per-checkpoint mutation lock: ``O_CREAT|O_EXCL`` on a lockfile whose
    content identifies the holder (uuid + host + pid + timestamp).

    Fail-fast: if the file already exists, :class:`_RestoreLockHeld` is raised — the
    caller reports it as a controlled ``locked`` result. The lock is removed only by
    its acquirer (including on exceptions); a hard kill leaves it behind ON PURPOSE
    (see the module docstring's stale-lock policy).
    """
    path = step_dir / RESTORE_LOCK_FILE
    payload = {
        "uuid": uuid.uuid4().hex,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "tool": _RECONCILED_BY,
        "created_ts": time.time(),
    }
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise _RestoreLockHeld(path, _read_lock_holder(path)) from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload))
        yield payload
    finally:
        path.unlink(missing_ok=True)


# Checkpoint selection consumes two UNTRUSTED filesystem inputs — the pointer file's
# text and the directory names under --ckpt-dir — and its result is where the
# exclusive lock, markers, backups and swaps happen. Both are therefore validated
# fail-closed BEFORE any lock or mutation: the pointer must be a bare integer (a
# hostile pointer like "5/../../victim" would otherwise compose a step dir OUTSIDE
# --ckpt-dir and create reconcile.lock in a sibling directory), candidate names must
# full-match the writer's grammar, and the chosen dir must be a real directory (not a
# symlink) whose resolved path is strictly inside the resolved checkpoint root.
# Containment does NOT stop at the step dir: _checkpoint_layout_problem then applies
# the same lstat discipline INSIDE it (live/backup transfer_queue dirs, data.pt and
# its backup, marker and lock files, the snapshot-probe files within the TQ dirs) —
# see restore().
_STEP_POINTER_RE = re.compile(r"[0-9]+")
_STEP_DIR_NAME_RE = re.compile(r"global_step_[0-9]+")


def _contained_step_dir(ckpt_root_resolved: Path, candidate: Path) -> Path | None:
    """``candidate`` if it is a non-symlink directory strictly inside the resolved
    checkpoint root, else ``None`` (containment check for untrusted names)."""
    try:
        if candidate.is_symlink() or not candidate.is_dir():
            return None
        resolved = candidate.resolve()
    except OSError:  # pragma: no cover - unreadable/looping paths: reject
        return None
    if resolved.parent != ckpt_root_resolved or resolved.name != candidate.name:
        return None
    return candidate


def _layout_entry_problem(path: Path, root_resolved: Path, want_dir: bool) -> str | None:
    """``None`` when ``path`` is absent or a real (non-symlink) node of the wanted
    kind resolving under ``root_resolved``; else a human-readable problem."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:  # pragma: no cover - uninspectable entry
        return f"{path} cannot be inspected ({exc})"
    if stat.S_ISLNK(st.st_mode):
        return f"{path} is a symlink"
    if want_dir and not stat.S_ISDIR(st.st_mode):
        return f"{path} is not a directory (mode {stat.filemode(st.st_mode)})"
    if not want_dir and not stat.S_ISREG(st.st_mode):
        return f"{path} is not a regular file (mode {stat.filemode(st.st_mode)})"
    try:
        resolved = path.resolve()
    except OSError as exc:  # pragma: no cover - unresolvable entry
        return f"{path} cannot be resolved ({exc})"
    if not resolved.is_relative_to(root_resolved):
        return f"{path} resolves to {resolved}, outside {root_resolved}"
    return None


def _checkpoint_layout_problem(step_dir: Path, ckpt_tq: Path) -> str | None:
    """Fail-closed layout validation of the SELECTED checkpoint, run BEFORE any lock
    or mutation (round-7 comment 2: containment must not stop at ``global_step_N``).

    Every path the tool later reads through, renames, copies onto or unlinks inside
    the step dir must be exactly what verl/this tool creates — real directories
    (live/backup ``transfer_queue``, and TQ's own ``.old``/``.tmp`` protocol siblings
    of the live dir) and regular files (``data.pt`` + its backup,
    ``reconcile.state``, ``reconcile.lock``, and the snapshot-probe files inside the
    TQ dirs) — all resolving under the resolved step dir. A symlink anywhere here
    redirects a heal's rename target, a ``copy2`` backup write or a freshness/stamp
    read outside the checkpoint. Returns a problem description (the caller reports
    a controlled ``invalid_checkpoint_layout`` result) or ``None``.

    The ``.old``/``.tmp`` siblings are validated for the same reason as everything
    else, and they carry one extra hazard of their own: ``tq.load_checkpoint`` renames
    ``transfer_queue.old`` back into place when ``transfer_queue`` is missing
    (``interface.py:1086-1092``). Renaming a SYMLINK moves the link, so a symlinked
    ``.old`` would let a later load read a controller snapshot from outside the
    checkpoint entirely. Refusing here is the only point at which that is cheap to
    stop; a real-directory sibling is benign and is swept later, just before the
    publication swap (see :func:`_sweep_tq_protocol_siblings`).
    """
    try:
        step_resolved = step_dir.resolve()
    except OSError as exc:  # pragma: no cover - selection already resolved it once
        return f"{step_dir} cannot be resolved ({exc})"
    backup = ckpt_tq.parent / (ckpt_tq.name + PRE_RECONCILE_SUFFIX)
    entries: list[tuple[Path, bool]] = [
        (ckpt_tq, True),
        (backup, True),
        (step_dir / "data.pt", False),
        (step_dir / ("data.pt" + PRE_RECONCILE_SUFFIX), False),
        (step_dir / RECONCILE_STATE_FILE, False),
        (step_dir / RESTORE_LOCK_FILE, False),
    ]
    entries += [(ckpt_tq.parent / (ckpt_tq.name + sfx), True) for sfx in TQ_PROTOCOL_SUFFIXES]
    # Snapshot-probe files inside each existing TQ dir: freshness (_snapshot_step /
    # _snapshot_fence_ts), the reconcile stamp reader and check_journal_recoverable
    # all read these with following opens.
    for tq_dir in (ckpt_tq, backup):
        if tq_dir.is_dir() and not tq_dir.is_symlink():
            entries += [(tq_dir / name, False) for name in ("metadata.json", "fence.json", "oplog_offsets.json")]
    for path, want_dir in entries:
        problem = _layout_entry_problem(path, step_resolved, want_dir)
        if problem is not None:
            return (
                f"checkpoint layout is not trustworthy: {problem}. The restore tool only "
                "operates on real directories/files created by verl or itself inside the "
                "step dir; refusing (fail closed) before any lock or mutation — remove or "
                "fix the offending entry and re-run"
            )
    return None


def find_ckpt_tq_dir(ckpt_dir: Path) -> tuple[Path, Path] | None:
    """(global_step dir, its transfer_queue subdir) for the latest checkpoint.

    Fail-closed selection (see the comment on the regexes above): the pointer FILE
    itself must be a regular non-symlink file living in the checkpoint root (a
    symlinked pointer would let an external file steer which step dir the lock,
    heal and swap machinery operates on) and its content must be a bare integer —
    anything else is IGNORED (with a warning) rather than followed or
    interpolated, falling back to the scan. Any candidate dir that is a symlink or
    escapes the resolved ``--ckpt-dir`` is skipped — every path this function
    returns is strictly inside the checkpoint root, so the lock/heal/swap machinery
    can never touch a sibling directory.
    """
    try:
        root = ckpt_dir.resolve(strict=True)
    except OSError:
        return None
    latest = ckpt_dir / LATEST_FILE
    pointer_problem = _layout_entry_problem(latest, root, want_dir=False)
    if pointer_problem is not None:
        logger.warning(
            "restore_tool: pointer %s is not trustworthy (%s); ignoring the pointer "
            "(fail closed) and scanning global_step_* instead",
            latest,
            pointer_problem,
        )
    elif latest.exists():
        step = latest.read_text().strip()
        if _STEP_POINTER_RE.fullmatch(step) is None:
            logger.warning(
                "restore_tool: %s content %r is not a bare integer; ignoring the pointer "
                "(fail closed) and scanning global_step_* instead",
                latest,
                step[:120],
            )
        else:
            step_dir = _contained_step_dir(root, ckpt_dir / f"global_step_{step}")
            if step_dir is not None:
                return step_dir, step_dir / "transfer_queue"
    candidates = sorted(
        (
            d
            for d in ckpt_dir.glob("global_step_*")
            if _STEP_DIR_NAME_RE.fullmatch(d.name) is not None and _contained_step_dir(root, d) is not None
        ),
        key=lambda d: int(d.name.rsplit("_", 1)[-1]),
    )
    if candidates:
        return candidates[-1], candidates[-1] / "transfer_queue"
    return None


def _snapshot_step(snapshot_dir: Path) -> int | None:
    meta = snapshot_dir / "metadata.json"
    if not meta.exists():
        return None
    try:
        md = json.loads(meta.read_text())
        return int(md.get("user_metadata", md).get("global_steps"))
    except (ValueError, TypeError, KeyError):
        return None


def _snapshot_fence_ts(snapshot_dir: Path) -> float | None:
    """fence_ts of a TQ snapshot dir: journal metadata first, then the fence manifest."""
    meta = snapshot_dir / "metadata.json"
    if meta.exists():
        try:
            md = json.loads(meta.read_text())
            ts = md.get("user_metadata", md).get("fence_ts")
            if ts is not None:
                return float(ts)
        except (ValueError, TypeError, AttributeError):
            pass
    fence = snapshot_dir / "fence.json"
    if fence.exists():
        try:
            ts = json.loads(fence.read_text()).get("fence_ts")
            return float(ts) if ts is not None else None
        except (ValueError, TypeError):
            pass
    return None


def _journal_freshness(journal: Path, ckpt_tq: Path) -> tuple[bool | None, dict]:
    """Is the journal STRICTLY fresher than the checkpoint's TQ snapshot?

    Returns ``(verdict, evidence)`` where verdict is ``True`` (journal fresher),
    ``False`` (checkpoint fresher or equal) or ``None`` (cannot be established).
    ``global_steps`` decides when both sides carry it; ``fence_ts`` breaks step ties
    or missing steps; anything else is ambiguous. Equal is NOT fresher: applying an
    equal journal buys nothing and risks fencing objects the checkpoint knows about.
    """
    js, cs = _snapshot_step(journal), _snapshot_step(ckpt_tq)
    jt, ct = _snapshot_fence_ts(journal), _snapshot_fence_ts(ckpt_tq)
    evidence = {"journal_step": js, "ckpt_step": cs, "journal_fence_ts": jt, "ckpt_fence_ts": ct}
    if js is not None and cs is not None:
        if js != cs:
            return js > cs, evidence
        if jt is not None and ct is not None:
            return jt > ct, evidence
        return False, evidence  # same step, no usable tie-break: not strictly fresher
    if jt is not None and ct is not None:
        return jt > ct, evidence
    return None, evidence


def _snapshot_content_identity(snapshot_dir: Path) -> tuple[str, str] | None:
    """Content identity of a snapshot dir: ``(file, sha256)`` of its most specific
    always-rewritten artifact.

    For a journal ring slot that is ``journal_stamp.json`` (unique per tick:
    ``completed_at``+``fence_ts`` timestamps), so a RECYCLED slot — same path, new
    tick — yields a different identity. Binding the publication transaction to this
    digest is what stops a completion from pairing one tick's TQ state with another
    tick's ``data.pt``.

    Each candidate is lstat-validated before the hash: a symlinked/non-regular
    artifact yields ``None`` — identity read THROUGH a link proves nothing about
    the slot (and at completion time the source dir is untrusted marker input), so
    the caller's digest check then fails closed. Containment follows from the
    caller-validated parent chain plus the non-symlink final component.
    """
    from .journal import STAMP_FILE

    for name in (STAMP_FILE, "fence.json", "metadata.json"):
        p = Path(snapshot_dir) / name
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            continue
        except OSError:  # pragma: no cover - uninspectable artifact
            return None
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            return None  # a redirected artifact must never vouch for the slot (fail closed)
        try:
            return name, hashlib.sha256(p.read_bytes()).hexdigest()
        except FileNotFoundError:  # pragma: no cover - raced away between lstat and read
            continue
    return None


def _write_reconcile_state(
    step_dir: Path, state: str, source_journal, txn: str, source_identity, store_id: str | None
) -> None:
    """Publish the transaction marker atomically (same-dir tmp + ``os.replace``).

    The marker binds the transaction to its artifacts: ``txn`` also lives in the
    published TQ dir's metadata, ``source_identity`` pins the exact journal tick
    whose ``data.pt`` may complete the transaction, and ``store_id`` pins the store
    LINEAGE the transaction belongs to (a marker left behind by a deleted store must
    never complete against a recreated store at the same path).
    """
    payload = {
        "state": state,
        "source_journal": str(source_journal),
        "txn": txn,
        "source_identity": list(source_identity) if source_identity else None,
        "store_id": store_id,
        "ts": time.time(),
    }
    tmp = step_dir / f"{RECONCILE_STATE_FILE}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, step_dir / RECONCILE_STATE_FILE)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


_MARKER_STATES = ("BEGIN", "TQ_PUBLISHED", "DONE")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _valid_source_identity(value) -> bool:
    """Exactly the shape :func:`_write_reconcile_state` records: a two-element
    ``[artifact_name, sha256_hexdigest]`` pair. Completion re-verifies the source
    journal against this digest, so a marker without a valid pair can never prove
    the pairing and must not complete."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and bool(value[0])
        and isinstance(value[1], str)
        and _SHA256_RE.fullmatch(value[1]) is not None
    )


def _corrupt_marker(reason: str) -> dict:
    return {"state": "CORRUPT", "corrupt_reason": reason}


def _read_reconcile_state(step_dir: Path) -> dict | None:
    """The transaction marker, STRICTLY validated (fail closed).

    Returns ``None`` when no marker exists, the payload for a well-formed marker,
    and a ``CORRUPT`` marker for ANYTHING else: non-JSON, non-object JSON (an array
    used to crash the planner), a missing/unknown ``state``, or a pending state
    whose required transaction/store/source fields are missing or mistyped. A
    CORRUPT marker is never silently overwritten as DONE — it flows into
    :func:`_plan_transaction_completion`, which turns it into an EXPLICIT
    ``discard_marker`` entry in the shared plan (identical in dry and real mode).
    """
    try:
        raw = (step_dir / RECONCILE_STATE_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError:
        # Raised by read_text, so the json.loads ValueError clause below never sees
        # it — without this clause it leaked as a raw traceback instead of the
        # CORRUPT classification every other unreadable marker shape gets.
        return _corrupt_marker("the marker is not valid UTF-8")
    try:
        payload = json.loads(raw)
    except ValueError:
        return _corrupt_marker("the marker is not valid JSON")
    if not isinstance(payload, dict):
        return _corrupt_marker(f"the marker must be a JSON object, got {type(payload).__name__}")
    state = payload.get("state")
    if state not in _MARKER_STATES:
        return _corrupt_marker(f"unknown marker state {state!r} (known: {', '.join(_MARKER_STATES)})")
    if state == "DONE":
        return payload  # terminal: the caller treats DONE as nothing-pending
    # A PENDING marker must carry well-typed transaction/store/source fields — the
    # completion decision consumes every one of them.
    if not (isinstance(payload.get("txn"), str) and payload["txn"]):
        return _corrupt_marker(f"pending marker carries no usable transaction id ({payload.get('txn')!r})")
    if not isinstance(payload.get("source_journal"), str):
        return _corrupt_marker(f"pending marker carries no usable source_journal ({payload.get('source_journal')!r})")
    store_id = payload.get("store_id")
    if store_id is not None and not isinstance(store_id, str):
        return _corrupt_marker(f"pending marker carries a mistyped store_id ({store_id!r})")
    source_identity = payload.get("source_identity")
    if source_identity is not None and not _valid_source_identity(source_identity):
        return _corrupt_marker(f"pending marker carries a mistyped source_identity ({source_identity!r})")
    return payload


def _tq_dir_reconcile_stamp(ckpt_tq: Path) -> dict | None:
    """The transaction stamp this tool wrote into the live ``transfer_queue`` dir:
    ``{"txn": ..., "store_id": ...}`` — or ``None`` when the dir was not published
    by us (a dir published before lineage stamping carries ``"store_id": None``)."""
    try:
        md = json.loads((ckpt_tq / "metadata.json").read_text(encoding="utf-8"))
        um = md.get("user_metadata", {})
        if um.get("reconciled_by") != _RECONCILED_BY:
            return None
        return {"txn": um.get("reconcile_txn"), "store_id": um.get("reconcile_store_id")}
    except (FileNotFoundError, NotADirectoryError, ValueError, AttributeError):
        return None


def _apply_dataloader_state(source: Path, step_dir: Path, expected_identity=None, expected_lineage=None) -> str:
    """data.pt half of the transaction: journal's loader position, atomically.

    ``expected_identity`` (from the BEGIN marker) re-verifies that ``source`` is
    still the exact tick the transaction was begun for — a recycled ring slot must
    not donate its ``data.pt`` to another tick's published TQ state. It is
    REQUIRED: with no identity there is no proof, so the check fails closed and the
    checkpoint's ``data.pt`` is kept (the round-5 code skipped verification
    entirely when the identity was missing — the bypass this replaces).
    ``expected_lineage`` re-verifies that the tick belongs to the CURRENT store
    lineage — a slot surviving from a deleted store must not donate either.

    Every artifact consulted here is locally validated before use, on top of the
    caller's source-path provenance check (:func:`_marker_source_disposition`):
    the identity candidates (:func:`_snapshot_content_identity`) and the fence
    stamp (``fsck._read_fence_metadata``) refuse symlinked/non-regular files, and
    the dataloader file itself is lstat-checked below — regular, non-symlink and
    therefore contained (validated parent chain + non-symlink final component).
    """
    from .fsck import _read_fence_metadata
    from .journal import DATALOADER_FILE

    if not _valid_source_identity(list(expected_identity) if expected_identity else None):
        return "unavailable (no valid source identity to verify the journal slot against); checkpoint data.pt kept"
    current = _snapshot_content_identity(Path(source))
    if current is None or list(current) != list(expected_identity):
        return "unavailable (journal slot recycled or rewritten since BEGIN); checkpoint data.pt kept"
    if expected_lineage is not None and _read_fence_metadata(Path(source)).get("store_id") != expected_lineage:
        return "unavailable (journal slot belongs to a different store lineage); checkpoint data.pt kept"
    dl_state = Path(source) / DATALOADER_FILE
    try:
        dl_st = os.lstat(dl_state)
    except FileNotFoundError:
        return "checkpoint (duplicates possible up to save_freq)"
    if stat.S_ISLNK(dl_st.st_mode) or not stat.S_ISREG(dl_st.st_mode):
        # copy2 follows symlinks on READ; a redirected slot artifact must not donate
        # bytes from outside the journal (completion path bypasses the preflight's
        # artifact validation, so the guard lives at the copy itself).
        return "unavailable (the journal's dataloader state is not a regular non-symlink file); checkpoint data.pt kept"
    target = step_dir / "data.pt"
    backup = step_dir / ("data.pt" + PRE_RECONCILE_SUFFIX)
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    _replace_file_atomically(dl_state, target)
    return "journal"


# A marker-recorded source longer than this is not a plausible journal/checkpoint
# path (PATH_MAX on Linux is 4096 bytes): classified malformed before any syscall.
_MAX_SOURCE_PATH_CHARS = 4096


def _malformed_source_problem(source) -> str | None:
    """Why ``source`` is not even a plausible path — or ``None`` when it is.

    The marker's recorded source reaches ``os.lstat``/``Path.resolve`` during
    provenance validation, and those calls CRASH (``ValueError``/``TypeError``)
    rather than fail cleanly on a non-string, empty, NUL-carrying or absurdly long
    value — none of which the tool itself ever writes. Classifying these shapes up
    front lets :func:`_marker_source_disposition` turn them into the same
    fail-closed ``invalid`` (unprovable) verdict as an escaping path, identically
    in dry and real mode, instead of an uncaught exception (or, for the empty
    string, a silent validation bypass).
    """
    if not isinstance(source, str):
        return f"the recorded source is not a string but {type(source).__name__}"
    if not source:
        return "the recorded source is empty"
    if "\x00" in source:
        return "the recorded source contains an embedded NUL byte"
    if len(source) > _MAX_SOURCE_PATH_CHARS:
        return f"the recorded source is implausibly long ({len(source)} chars > {_MAX_SOURCE_PATH_CHARS})"
    return None


def _marker_source_disposition(source, journal_root: Path | None, step_dir: Path) -> tuple[str, str | None]:
    """Provenance of a marker-recorded source path: ``("ok", None)``,
    ``("missing", None)`` or ``("invalid", problem)``.

    The marker's ``source_journal`` is UNTRUSTED filesystem input (it survives
    crashes, edits and store swaps), and completing an interrupted transaction
    reads identity/fence/dataloader artifacts through it — so BEFORE any such read
    the path must be PROVEN to name a real directory under an allowed root (the
    ``--journal`` ring or the step dir): lexically contained (no ``..`` segments —
    ``relative_to`` is purely lexical and would happily keep them), every component
    below the root a real non-symlink directory, and the resolved dir still under
    the resolved root. A MALFORMED value (non-string, empty, embedded NUL,
    oversized — see :func:`_malformed_source_problem`) is classified ``invalid``
    up front, before any syscall could crash on it. ``"missing"`` is the one
    benign non-ok shape — a lexically contained source whose components are real
    until one is simply absent (a legitimately recycled ring slot): completion
    proceeds but keeps the checkpoint's ``data.pt``, exactly as before. Everything
    else — an escaping or traversal-carrying path, a symlinked parent, a symlinked
    snapshot dir itself — is ``"invalid"``: unprovable provenance, so the
    transaction is discarded with the checkpoint data intact (an explicit plan
    entry, identical in dry and real mode).
    """
    problem = _malformed_source_problem(source)
    if problem is not None:
        return "invalid", f"{problem}; it cannot be inspected, let alone proven to live under an allowed root"
    src = Path(source)
    roots = [r for r in (journal_root, step_dir) if r is not None]
    for root in roots:
        try:
            rel = src.relative_to(root)
        except ValueError:
            continue
        if not rel.parts or any(part == ".." for part in rel.parts):
            return "invalid", (
                f"source {source} does not name a path strictly below the allowed root {root} "
                f"(relative view {str(rel)!r})"
            )
        try:
            root_resolved = root.resolve()
        except OSError as exc:  # pragma: no cover - the tool already resolved these roots
            return "invalid", f"allowed root {root} cannot be resolved ({exc})"
        cur = root
        for part in rel.parts:
            cur = cur / part
            try:
                st = os.lstat(cur)
            except FileNotFoundError:
                return "missing", None  # contained and provably absent: a recycled slot
            except (OSError, ValueError) as exc:  # pragma: no cover - uninspectable component
                # ValueError belt-and-braces: the malformed-shape classifier above
                # already rejects NUL-carrying sources before any syscall.
                return "invalid", f"source component {cur} cannot be inspected ({exc})"
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                return "invalid", (
                    f"source component {cur} is not a real (non-symlink) directory (mode {stat.filemode(st.st_mode)})"
                )
        try:
            resolved = src.resolve()
        except (OSError, ValueError) as exc:  # pragma: no cover - unresolvable validated chain
            return "invalid", f"source {source} cannot be resolved ({exc})"
        if not resolved.is_relative_to(root_resolved):
            return "invalid", f"source {source} resolves to {resolved}, outside {root}"
        return "ok", None
    return "invalid", (
        f"source {source} is not under any allowed root (journal root: "
        f"{journal_root if journal_root is not None else 'none'}, step dir: {step_dir})"
    )


def _plan_transaction_completion(
    step_dir: Path, live_dir: Path, pending: dict, current_store_id: str | None, journal_root: Path | None
) -> dict | None:
    """Decide, from plain reads, what to do about an interrupted transaction.

    ``live_dir`` is the POST-HEAL view of the checkpoint's ``transfer_queue`` dir
    (the backup when the live dir is missing and healable). Returns ``None`` when
    there is nothing pending, else a plan dict with ``action``:

    * ``"discard_marker"`` — the pairing cannot be proven: the marker failed strict
      validation (non-object JSON, unknown state, mistyped fields — see
      :func:`_read_reconcile_state`), the marker's publication never became visible,
      the marker predates transaction ids, the live dir belongs to a DIFFERENT
      publication, the transaction belongs to a DIFFERENT STORE LINEAGE than the
      current ``--store`` (marker/published-dir lineage vs the store's
      ``_store_id``; a recreated store at the same path must never adopt a deleted
      lineage's half-finished transaction), the marker's recorded SOURCE PATH cannot
      be proven to live under the allowed journal/checkpoint roots
      (:func:`_marker_source_disposition`, applied UNCONDITIONALLY to every pending
      source — malformed values such as an empty string or an embedded NUL are
      classified invalid without ever being inspected; completion reads identity/
      fence/dataloader artifacts through that path, so a symlinked or escaping
      source could donate bytes from OUTSIDE those roots), or the marker carries NO VALID
      SOURCE IDENTITY/DIGEST (without one the data.pt half cannot be re-verified
      against the exact journal tick, so completion is not provable). Backups and
      dirs stay intact; the run decides from scratch.
    * ``"complete"`` — the live TQ dir provably IS this transaction's publication,
      the lineage matches, the source path is provably contained (or provably
      absent — a recycled slot, whose completion keeps the checkpoint's
      ``data.pt``), AND the marker carries a valid source identity; finish the
      ``data.pt`` half, re-verifying the source digest and lineage.

    Shared verbatim by the real completion and by ``--dry-run`` reporting, so the
    dry plan is exactly what a real run would do.
    """
    if pending.get("state") not in ("BEGIN", "TQ_PUBLISHED", "CORRUPT"):
        return None
    marker_txn = pending.get("txn")
    stamp = _tq_dir_reconcile_stamp(live_dir)
    live_txn = stamp.get("txn") if stamp else None
    if marker_txn is None or live_txn is None or marker_txn != live_txn:
        corrupt = ""
        if pending.get("corrupt_reason"):
            corrupt = f" (marker rejected by strict validation: {pending['corrupt_reason']})"
        return {
            "action": "discard_marker",
            "reason": (
                "the live transfer_queue dir cannot be proven to be THIS transaction's "
                f"publication (marker txn {marker_txn!r} vs live {live_txn!r}){corrupt}; backups are "
                "left intact and the run decides from scratch"
            ),
        }
    marker_lineage = pending.get("store_id")
    live_lineage = stamp.get("store_id") if stamp else None
    if current_store_id is None or marker_lineage != current_store_id or live_lineage != current_store_id:
        return {
            "action": "discard_marker",
            "cross_lineage": True,
            "reason": (
                f"cross-lineage transaction: marker lineage {marker_lineage!r} / published-dir "
                f"lineage {live_lineage!r} vs the current store lineage {current_store_id!r} "
                "(the transaction was begun against a different store — e.g. one deleted and "
                "recreated at the same path — or predates lineage stamping); its data.pt half "
                "must not be adopted. Backups are left intact and the run decides from scratch"
            ),
        }
    # UNCONDITIONAL provenance validation (round-9): the round-8 code skipped this
    # check for a falsy source, so an empty string — a value only marker
    # tampering/corruption can produce, never this tool — sailed on as
    # "completable", and a malformed one (embedded NUL) crashed os.lstat at
    # execution time. Every pending source now gets a verdict here AND again at
    # execution; malformed shapes classify invalid -> unprovable_source,
    # identically in dry and real mode.
    source = pending.get("source_journal")
    disposition, provenance = _marker_source_disposition(source, journal_root, step_dir)
    if disposition == "invalid":
        return {
            "action": "discard_marker",
            "unprovable_source": True,
            "reason": (
                f"the marker's recorded source cannot be proven to be a real directory under "
                f"the allowed journal/checkpoint roots: {provenance}. Completion reads "
                "identity/fence/dataloader artifacts through that path and copies dataloader "
                "state into data.pt, so a symlinked or escaping source could donate bytes from "
                "OUTSIDE those roots. Not provably completable -> discarded; the checkpoint's "
                "data (data.pt and the published TQ dir) stays intact and backups are left in "
                "place"
            ),
        }
    source_identity = pending.get("source_identity")
    if not _valid_source_identity(source_identity):
        return {
            "action": "discard_marker",
            "reason": (
                f"the marker carries no valid source identity/digest ({source_identity!r}): "
                "completion re-verifies the source journal slot against the digest captured "
                "at BEGIN, and without one a recycled ring slot could pair one tick's TQ "
                "state with another tick's data.pt. Not provably complete -> discarded; "
                "backups are left intact and the run decides from scratch"
            ),
        }
    return {
        "action": "complete",
        "resumed_from_state": pending.get("state"),
        "source_journal": source,
        "txn": marker_txn,
        "source_identity": source_identity,
        "store_id": marker_lineage,
    }


def _execute_transaction_plan(
    step_dir: Path, ckpt_tq: Path, plan: dict, store_id: str | None, journal_root: Path | None
) -> tuple[dict | None, dict | None]:
    """PHASE 2 for a pending transaction: ``(completed_evidence, discarded_report)``.

    Completion happens only on PROOF, never on the marker alone (see
    :func:`_plan_transaction_completion`); ownership AND the source's provenance are
    re-validated immediately before the ``data.pt`` half + DONE, even under the
    lock — no identity/fence/dataloader artifact is read until the recorded source
    path has (re-)passed :func:`_marker_source_disposition`.
    """
    if plan is None:
        return None, None
    if plan["action"] == "discard_marker":
        (step_dir / RECONCILE_STATE_FILE).unlink(missing_ok=True)
        logger.warning("restore_tool: discarded pending transaction: %s", plan["reason"])
        return None, plan
    # ownership revalidation (belt and braces under the exclusive lock): the live dir
    # must still be THIS transaction's publication in THIS store lineage.
    stamp = _tq_dir_reconcile_stamp(ckpt_tq)
    if stamp is None or stamp.get("txn") != plan["txn"] or stamp.get("store_id") != store_id:
        report = dict(plan)
        report["action"] = "discard_marker"
        report["reason"] = (
            f"ownership revalidation before the data.pt half failed (live stamp {stamp!r} vs "
            f"txn {plan['txn']!r} / lineage {store_id!r}); the transaction is discarded and "
            "backups are left intact"
        )
        (step_dir / RECONCILE_STATE_FILE).unlink(missing_ok=True)
        logger.warning("restore_tool: %s", report["reason"])
        return None, report
    source = plan["source_journal"]
    # Same UNCONDITIONAL verdict as the plan (round-9): no falsy-source bypass — a
    # malformed value must classify invalid here too, never reach a syscall or be
    # waved through as benignly "missing".
    disposition, provenance = _marker_source_disposition(source, journal_root, step_dir)
    if disposition == "invalid":
        # The plan proved provenance moments ago; failing NOW means the path changed
        # out-of-band. Same fail-closed direction as the plan: discard, read nothing.
        report = dict(plan)
        report["action"] = "discard_marker"
        report["unprovable_source"] = True
        report["reason"] = (
            f"the marker's recorded source failed execution-time revalidation ({provenance}); "
            "reading identity or dataloader artifacts through it could consume bytes from outside "
            "the journal/checkpoint roots, so the transaction is discarded and backups are left intact"
        )
        (step_dir / RECONCILE_STATE_FILE).unlink(missing_ok=True)
        logger.warning("restore_tool: %s", report["reason"])
        return None, report
    evidence: dict = {"resumed_from_state": plan["resumed_from_state"], "source_journal": source}
    if disposition == "ok":
        evidence["dataloader_state"] = _apply_dataloader_state(
            Path(source), step_dir, expected_identity=plan.get("source_identity"), expected_lineage=store_id
        )
    else:
        # The journal ring may have recycled the slot since the crash: the TQ half is
        # published, the matching loader position is gone. Keep the checkpoint's
        # data.pt (duplicate dispatches possible up to save_freq) and say so.
        evidence["dataloader_state"] = "unavailable (journal slot recycled); checkpoint data.pt kept"
    _write_reconcile_state(
        step_dir, "DONE", source, txn=plan["txn"], source_identity=plan.get("source_identity"), store_id=store_id
    )
    logger.warning("restore_tool: completed interrupted publication for %s (%s)", step_dir, evidence)
    return evidence, None


def _sweep_tq_protocol_siblings(ckpt_tq: Path) -> list[str]:
    """Delete TQ's ``.old``/``.tmp`` siblings of the live TQ dir. Returns what went.

    WHEN, and why that timing rather than a refusal. This runs under the exclusive
    checkpoint lock, after ``tq.save_checkpoint`` has written the complete reconciled
    snapshot beside the live dir and immediately BEFORE :func:`_publish_reconciled_dir`
    swaps it in. Never in the preflight, never in dry-run, and only while the live dir
    exists.

    A refusal in the preflight was the alternative and it is wrong, because the
    dangerous sibling and the load-bearing sibling are the same file. A crash between
    TQ's two renames leaves ``transfer_queue`` MISSING and ``transfer_queue.old``
    holding the only copy of that snapshot; TQ's roll-forward
    (``interface.py:1086-1092``) is the intended repair for exactly that state.
    Refusing on sight would turn a recoverable crash into an operator ticket, and
    deleting in the preflight would destroy the only snapshot. Hence the
    ``ckpt_tq.exists()`` gate at the call site: when the live dir is absent, the sibling
    is left completely alone.

    With the live dir present, the sibling is a resurrection hazard and nothing else,
    and the sweep is provably not destroying the only copy: at this point the live dir
    AND a complete, strictly newer reconciled dir both exist, so the state the sibling
    holds is superseded twice over. Sweeping BEFORE the swap is what matters, because
    the swap itself opens the window the roll-forward reacts to: between the two renames
    ``transfer_queue`` is absent, so a crash there would leave a checkpoint whose next
    ``tq.load_checkpoint`` renames the ``.old`` into place and resumes from PRE-reconcile
    buffer state (fenced objects expected again, dropped metadata live again, a
    ``data.pt`` half that no longer matches) INSTEAD of letting this tool heal from its
    own ``.pre_reconcile`` backup. Sweeping afterwards also left the gap between publish
    and sweep, which the completion path never closed. The ``.pre_reconcile`` backup is
    deliberately NOT swept: that one is the tool's own documented rollback, it is named
    so nothing else will read it, and no TQ code path resurrects it.

    A ``.tmp`` sibling is unconditional garbage (an interrupted build; TQ itself
    ``rmtree``s a pre-existing one at the start of its next save) and goes the same way.

    Symlinked siblings never reach here: :func:`_checkpoint_layout_problem` refuses the
    whole checkpoint before the lock. ``shutil.rmtree`` would refuse a symlink anyway.
    """
    removed: list[str] = []
    for sibling in _tq_protocol_siblings(ckpt_tq):
        try:
            shutil.rmtree(sibling)
        except OSError as exc:
            # Not fatal: the reconciled snapshot is already on disk and the publication
            # proceeds. Say so loudly, because the leftover is exactly what a
            # roll-forward would resurrect if the swap is interrupted.
            logger.warning(
                "restore_tool: could not remove TQ protocol leftover %s (%s). Remove it manually: "
                "if %s is missing when a resume runs, tq.load_checkpoint would roll this stale "
                "snapshot forward and resume from pre-reconcile buffer state",
                sibling,
                exc,
                ckpt_tq,
            )
            continue
        removed.append(str(sibling))
        logger.warning(
            "restore_tool: removed TQ protocol leftover %s (superseded by the reconciled snapshot "
            "about to be published as %s; left in place it could be rolled forward by a later "
            "tq.load_checkpoint)",
            sibling,
            ckpt_tq,
        )
    return removed


def _publish_reconciled_dir(ckpt_tq: Path, replacement: Path) -> None:
    """Swap ``replacement`` into place, keeping the original as a backup.

    Never leaves the checkpoint without a usable state: the original is *renamed*
    (not deleted) before the replacement is renamed in, and a crash between the two
    renames is healed on the next run. A backup left by an EARLIER reconcile is
    replaced by the current live dir (the newest pre-reconcile state is the one a
    heal must restore); the swap runs under the exclusive checkpoint lock.

    Healing that window is only the tool's to do because
    :func:`_sweep_tq_protocol_siblings` has already run: while ``ckpt_tq`` is absent
    between the two renames, a ``transfer_queue.old`` beside it would be what TQ's own
    ``load_checkpoint`` rolls forward, and pre-reconcile buffer state would win over the
    backup this function just wrote.
    """
    backup = ckpt_tq.parent / (ckpt_tq.name + PRE_RECONCILE_SUFFIX)
    if ckpt_tq.exists():
        if backup.exists():
            shutil.rmtree(backup)
        ckpt_tq.rename(backup)
    _PUBLISH_CRASH_HOOK()
    replacement.rename(ckpt_tq)


def _replace_file_atomically(source: Path, target: Path) -> None:
    """tmp-in-same-dir + os.replace: readers never see a torn/partial target.

    The tmp name carries pid + a uuid: pid alone is unique per host only, and the
    checkpoint dir can be on a shared filesystem.
    """
    tmp = target.parent / f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copy2(source, tmp)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _journal_candidates(journal_root: Path | None, journal_slot: str | None) -> list[Path]:
    """Ring slots to try, newest first — or exactly the one an operator named.

    Without a selector the whole ring is a candidate list rather than just its head:
    the fail-closed artifact validation below runs per candidate, so ONE poisoned slot
    no longer makes the journal unusable. With ``--journal-slot`` the list is that slot
    alone (empty when the ring does not hold it as a complete, non-redirected slot),
    which turns the previously undocumented mid-outage ``mv`` into a flag.
    """
    from .journal import complete_journal_slots, select_journal_slot

    if journal_root is None:
        return []
    if journal_slot is not None:
        chosen = select_journal_slot(journal_root, journal_slot)
        return [chosen] if chosen is not None else []
    return complete_journal_slots(journal_root)


def _validate_source(store: Path, source: Path, force_store_mismatch: bool) -> tuple[str | None, dict]:
    """Fail-closed validation of ONE candidate source; no mutation of any kind.

    ``(None, {"tail":..., "store_id":...})`` when the source can be trusted, else
    ``(status, {"detail": ...})`` with the controlled status the caller reports. The
    store-id read lives inside the same guard: a symlinked/non-regular ``_store_id``
    raises ``StoreIntegrityError`` (an external file must never act as the store's
    lineage authority) and must surface as the same ``store_mismatch``.
    """
    from .fs_kv import StoreIntegrityError, read_store_id
    from .fsck import StoreMismatchError, UnrecoverableJournalError, _read_oplog_tail, check_journal_recoverable
    from .fsck import check_store_identity as _check_store_identity

    try:
        check_journal_recoverable(source)  # fence stamp first: malformed fence.json is an
        # untrusted artifact (unrecoverable_journal), not an identity question
        _check_store_identity(store, source, force_store_mismatch)
        return None, {"tail": _read_oplog_tail(Path(store), Path(source)), "store_id": read_store_id(store)}
    except StoreMismatchError as exc:
        return "store_mismatch", {"detail": str(exc)}
    except UnrecoverableJournalError as exc:
        return "unrecoverable_journal", {"detail": str(exc)}
    except StoreIntegrityError as exc:
        # belt and braces: check_store_identity already converts this for its own
        # read; the direct read above can still race a swap (documented TOCTOU).
        return "store_mismatch", {"detail": str(exc)}


def _preflight(
    store: Path,
    journal_root: Path | None,
    step_dir: Path,
    ckpt_tq: Path,
    force: bool,
    force_store_mismatch: bool,
    dry_run: bool,
    journal_slot: str | None = None,
) -> dict:
    """PHASE 1: read-only. Validates everything and computes the complete plan.

    PLAIN FILE READS only — no TQ, no writes of any kind. The plan is shared
    verbatim by ``--dry-run`` (reported, then exit) and the real mode (executed
    under the exclusive checkpoint lock). ``plan["status"]`` is the planned terminal
    status: ``"reconciled"`` means phase 2 runs the full fsck + publication;
    ``"checkpoint_fresher"``/``"ordering_ambiguous"`` mean phase 2 executes only the
    heal/transaction repairs; every other status means phase 2 must not reconcile
    (``plan["validated"]`` is False — the artifacts cannot be trusted). Healing this
    tool's OWN interrupted swap is not gated on that: it is a rename of a backup this
    tool created and depends on no journal artifact (see :func:`_heal_only`).

    Journal selection walks the ring newest-first (the same validation
    ``journal.latest_journal`` applies, over every complete slot rather than only its
    head) and validates each slot, so a poisoned newest slot falls back to the
    next-older complete one and the report says which slot was used and why the newer
    ones were refused. ``journal_slot`` pins the choice to one slot instead.
    """
    backup = ckpt_tq.parent / (ckpt_tq.name + PRE_RECONCILE_SUFFIX)
    pending = _read_reconcile_state(step_dir)
    if pending is not None and pending.get("state") == "DONE":
        pending = None
    candidates = _journal_candidates(journal_root, journal_slot)
    journal = candidates[0] if candidates else None
    # The post-heal view of the checkpoint snapshot: what the real run will compare
    # freshness against (and reconcile from, absent a journal) AFTER healing.
    ckpt_probe = ckpt_tq if ckpt_tq.exists() else (backup if backup.exists() else None)
    source = journal if journal is not None else ckpt_probe
    plan: dict = {
        "dry_run": dry_run,
        "step_dir": step_dir,
        "ckpt_tq": ckpt_tq,
        "backup": backup,
        "journal_root": journal_root,
        "pending": pending,
        "heal_publication": (not ckpt_tq.exists()) and backup.exists(),
        "source": source,
        "source_kind": (
            "journal"
            if journal is not None
            else ("checkpoint" if ckpt_tq.exists() else ("checkpoint_backup" if ckpt_probe is not None else None))
        ),
        "txn_plan": None,
        "freshness": None,
        "tail": None,
        "store_id": None,
        # Which ring slots were considered, and every one that was refused (newest
        # first). Reported in both modes so an operator sees that an older slot is in
        # use and why the newer one could not be.
        "journal_slot_request": journal_slot,
        "journal_slots": [str(p) for p in candidates],
        "source_slot_rejections": [],
        # TQ's own checkpoint-protocol leftovers around the live dir. Reported so an
        # operator sees the checkpoint is not clean; they never influence source
        # selection (the live dir has a fixed name and the healable backup a distinct
        # suffix), and they are only removed once a complete reconciled snapshot exists
        # beside a live dir, immediately before the publication swap.
        "tq_protocol_siblings": [str(p) for p in _tq_protocol_siblings(ckpt_tq)],
        "force_store_mismatch": force_store_mismatch,
        "validated": False,
        "status": None,
        "detail": None,
    }
    if journal_root is not None and journal_slot is not None and not candidates:
        # No source at all: an explicit selection is a statement about WHICH slot to
        # reconcile from, so silently reconciling from something else (the newest slot,
        # or the checkpoint) would apply state the operator did not ask for.
        plan["source"] = None
        plan["source_kind"] = None
        plan["status"] = "journal_slot_not_found"
        plan["detail"] = (
            f"--journal-slot {journal_slot!r} does not name a complete ring slot under {journal_root} "
            "(the slot must be a real directory whose name matches journal-NNN and which carries a "
            "regular journal_stamp.json and controller_state.pkl); nothing was applied"
        )
        return plan
    if source is None:
        plan["status"] = "nothing_to_restore"
        plan["detail"] = "neither a journal nor a checkpoint TQ snapshot exists"
        return plan
    # ---- ALL fail-closed validation, before ANY reconciliation: identity,
    # oplog:false refusal, tick coherence, controller-schema pin, offsets schema, full
    # tail parse. Walked per candidate slot: the first trustworthy one wins, the rest
    # are reported as refused.
    verdict: tuple[str | None, dict] = ("nothing_to_restore", {"detail": "no candidate source"})
    for candidate in candidates or [source]:
        verdict = _validate_source(store, candidate, force_store_mismatch)
        if verdict[0] is None:
            source = candidate
            break
        if candidates:  # a ring slot; a checkpoint source has no next-older sibling
            plan["source_slot_rejections"].append(
                {"slot": str(candidate), "status": verdict[0], "detail": verdict[1]["detail"]}
            )
    if verdict[0] is not None:
        plan["status"], plan["detail"] = verdict[0], verdict[1]["detail"]
        if len(plan["source_slot_rejections"]) > 1:
            plan["detail"] += (
                f" (every one of the {len(plan['source_slot_rejections'])} complete journal slots in "
                "the ring was refused; see source_slot_rejections)"
            )
        return plan
    plan["source"] = source
    plan["tail"] = verdict[1]["tail"]
    plan["store_id"] = verdict[1]["store_id"]
    if plan["source_slot_rejections"]:
        logger.warning(
            "restore_tool: using the older journal slot %s; %d newer slot(s) were refused (%s)",
            source,
            len(plan["source_slot_rejections"]),
            ", ".join(r["status"] for r in plan["source_slot_rejections"]),
        )
    plan["validated"] = True
    if pending is not None:
        heal_target = ckpt_tq if ckpt_tq.exists() else backup
        plan["txn_plan"] = _plan_transaction_completion(step_dir, heal_target, pending, plan["store_id"], journal_root)
    if plan["source_kind"] == "journal" and ckpt_probe is not None and not force:
        fresher, evidence = _journal_freshness(source, ckpt_probe)
        plan["freshness"] = evidence
        if fresher is not True:
            plan["status"] = "checkpoint_fresher" if fresher is False else "ordering_ambiguous"
            plan["detail"] = (
                "the checkpoint's transfer_queue snapshot is not older than the newest "
                "journal; applying the journal would regress buffer state, so it was NOT "
                "applied (interrupted-publication repairs, if any were pending, are "
                "executed first and reported in this result). Re-run with --force to "
                "apply the journal anyway."
            )
            return plan
    plan["status"] = "reconciled"
    return plan


#: Every restore() result carries this ONE stable schema: ``status``, ``dry_run``,
#: ``checkpoint_step_dir``, ``plan_available``, ``detail``, ``checkpoint_tq_present``,
#: ``controller_probe`` and every plan field below are ALWAYS present. Results that ran
#: the preflight set ``plan_available: True`` (:func:`_plan_report`); statuses that never
#: could — ``no_checkpoint``, ``locked``, ``live_controller_present`` — set it to False
#: with every plan field explicitly ``None`` (:func:`_no_plan_report`) instead of
#: silently omitting advertised keys.
_PLAN_FIELDS = (
    "pending_transaction",
    "would_heal_publication",
    "would_complete_transaction",
    "source_selection",
    "source_journal",
    "journal_slot_request",
    "journal_slots",
    "source_slot_rejections",
    "freshness",
    "tq_protocol_siblings",
)


def _plan_report(plan: dict, status: str | None = None) -> dict:
    """The JSON-facing view of a plan; EVERY result carries the full stable schema."""
    return {
        "status": status or plan["status"],
        "dry_run": plan["dry_run"],
        "checkpoint_step_dir": str(plan["step_dir"]),
        "plan_available": True,
        "detail": plan["detail"],
        "pending_transaction": plan["pending"],
        "would_heal_publication": plan["heal_publication"],
        "would_complete_transaction": plan["txn_plan"],
        "source_selection": plan["source_kind"],
        "source_journal": str(plan["source"]) if plan["source"] is not None else None,
        # Ring-slot selection: what was asked for, what was available, and every slot
        # refused on the way to the one in use.
        "journal_slot_request": plan["journal_slot_request"],
        "journal_slots": plan["journal_slots"],
        "source_slot_rejections": plan["source_slot_rejections"],
        "freshness": plan["freshness"],
        # Observed, never acted on in the preflight: a real-directory .old/.tmp sibling
        # can be the ONLY copy of a snapshot after a crash mid-save. Swept under the lock
        # just before the publication swap, and only while the live dir exists
        # (:func:`_sweep_tq_protocol_siblings`).
        "tq_protocol_siblings": plan["tq_protocol_siblings"],
    }


def _no_plan_report(status: str, dry_run: bool, detail: str, step_dir: Path | None = None) -> dict:
    """The SAME stable schema for statuses that never ran a preflight
    (``no_checkpoint``, ``locked``, ``invalid_checkpoint_layout``): the plan is
    explicitly marked unavailable (``plan_available: False``, every plan field
    ``None``) rather than omitted."""
    out = {
        "status": status,
        "dry_run": dry_run,
        "checkpoint_step_dir": str(step_dir) if step_dir is not None else None,
        "plan_available": False,
        "detail": detail,
    }
    for key in _PLAN_FIELDS:
        out[key] = None
    return out


def _finalize(result: dict, ckpt_tq: Path | None, controller_probe: dict | None = None) -> dict:
    """Stamp the one fact an automated restart wrapper must not proceed past.

    verl's resume does ``if os.path.exists(tq_ckpt_path): tq.load_checkpoint(...)`` and
    skips a MISSING TransferQueue snapshot dir without a warning, so a checkpoint left
    with no live ``transfer_queue`` resumes with an empty controller: every buffered
    trajectory becomes orphaned bytes, no error is raised anywhere, and the only trace
    is a line in the output of a tool the operator ran before restarting. Reported here
    and turned into a non-zero exit in :func:`main`.

    ``controller_probe`` is the exclusivity probe's own account of itself (see
    :func:`_live_controller`), ``None`` for the statuses decided before the probe ran.
    Always present, like every other advertised key.
    """
    result["checkpoint_tq_present"] = None if ckpt_tq is None else (ckpt_tq.is_dir() and not ckpt_tq.is_symlink())
    result["controller_probe"] = controller_probe or None
    return result


def _heal_only(plan: dict) -> dict:
    """Execute the publication heal for a plan whose SOURCE was refused.

    ``heal_publication`` means the checkpoint currently has no ``transfer_queue`` dir at
    all, only this tool's ``transfer_queue.pre_reconcile`` backup, because an earlier
    run of this tool was killed inside its two-rename swap. Putting it back is a rename
    of the tool's own backup: it reads no journal artifact, no offsets fence, no op log
    and no store identity. Refusing it because a JOURNAL SLOT is corrupt conflates "I
    cannot trust this journal" with "I cannot trust my own backup" and leaves the
    checkpoint strictly worse than healing would — resumable buffer state discarded in
    exchange for nothing.

    So the gate is split: journal-artifact trust gates the journal-sourced reconcile and
    publication, while the heal runs whenever the checkpoint LAYOUT validated (which
    :func:`restore` proved before any lock). The pending transaction is NOT completed
    here — completion proves itself against the source journal's digest, which is
    exactly what could not be trusted.
    """
    if not plan["heal_publication"]:
        return _plan_report(plan)
    refused = {"status": plan["status"], "detail": plan["detail"]}
    plan["backup"].rename(plan["ckpt_tq"])
    logger.warning(
        "restore_tool: healed interrupted publication (restored %s from backup) although the "
        "source was refused as %s; the journal was NOT applied",
        plan["ckpt_tq"],
        refused["status"],
    )
    out = _plan_report(plan, status="healed_only")
    out["healed_publication"] = True
    out["refused"] = refused
    out["detail"] = (
        f"the source could not be trusted ({refused['status']}), so nothing was reconciled; the "
        "interrupted publication left by an earlier run of this tool WAS healed, because restoring "
        f"its own backup depends on no journal artifact. {plan['ckpt_tq'].name} is live again with "
        "pre-reconcile buffer state, so the run resumes from the checkpoint's own snapshot instead "
        f"of an empty buffer. Resolve the refusal before resuming: {refused['detail']}"
    )
    return out


def _live_controller_detail(controller: dict, probe: dict) -> str:
    """The ``live_controller_present`` detail, which differs by what the probe ESTABLISHED.

    A sighting and a failure to decide are both refused, and they need different
    instructions. Telling an operator that "a controller is reachable, stop the job" when
    the probe never reached a cluster sends them looking for a job a stale pointer
    invented, and leaves ``--allow-live-controller``, which skips the check whether or not
    a controller is live, as the only escape the report names. The safe remedy comes
    first: clearing a pointer left behind by a dead cluster costs nothing and restores the
    check, while the override removes it.
    """
    common = (
        "reconciling deletes store objects and rewrites controller metadata, so it must only run "
        "against a buffer whose controller and writers are gone. Nothing was written to the store "
        "or the checkpoint in either mode"
    )
    if _presence_is_a_sighting(controller):
        return (
            f"a TransferQueue controller is reachable ({controller}); {common}. Stop the job (or "
            "wait for its controller actor to go away) and re-run; --allow-live-controller proceeds "
            "anyway against a scratch controller of this tool's own (DANGEROUS: the running job "
            "keeps publishing objects this run would fence)"
        )
    if probe.get("outcome") in (PROBE_DISCOVERY_FAILED, PROBE_CLUSTER_UNJOINABLE):
        named = probe.get("discovered") or []
        reason = "the probe never reached a cluster to ask one"
        if named:
            remedy = (
                f"Ray's discovery named {', '.join(named)}, and an address is only a POINTER: "
                "RAY_ADDRESS if it is set, /tmp/ray/ray_current_cluster, which `ray start` writes "
                "and nothing removes, and any local gcs_server process. If the cluster it names is "
                "gone, clear the pointer and re-run: unset a stale RAY_ADDRESS (or point it at the "
                "head that is actually up) and delete the address file of the dead cluster. With "
                "nothing left to discover, the tool finds no controller and proceeds on its own"
            )
        else:
            remedy = (
                "Cluster discovery itself broke, so it named no address to check. What it reads is "
                "RAY_ADDRESS and /tmp/ray/ray_current_cluster, which `ray start` writes and nothing "
                "removes, and both outlive the cluster they name: unset a stale RAY_ADDRESS (or "
                "point it at the head that is actually up) and delete the address file of the dead "
                "cluster, then re-run. With nothing left to discover, the tool finds no controller "
                "and proceeds on its own. `ray status` is worth a look if a cluster is supposed to "
                "be up, but after a total crash it will keep failing and clearing the pointers is "
                "what settles this"
            )
    else:
        reason = "the probe reached a cluster and could not get an answer out of the actor name"
        remedy = (
            'Check that cluster for the job (`ray list actors --filter "name=TransferQueueController"`) '
            "and re-run once the actor is gone or answering"
        )
    return (
        f"a live TransferQueue controller could not be ruled out ({controller}); {common}, and doubt "
        f"counts as presence. No controller answered: {reason}. {remedy}. Only once that is settled "
        "does --allow-live-controller apply (DANGEROUS: it drops the check itself, including for the "
        "case where the controller IS live and publishing objects this run would fence)"
    )


def restore(
    store: Path,
    journal_root: Path | None,
    ckpt_dir: Path,
    partition: str = "train",
    group_size: int | None = None,
    serializer: str = "tq",
    dry_run: bool = False,
    force: bool = False,
    force_store_mismatch: bool = False,
    journal_slot: str | None = None,
    allow_live_controller: bool = False,
) -> dict:
    found = find_ckpt_tq_dir(ckpt_dir)
    if found is None:
        return _finalize(_no_plan_report("no_checkpoint", dry_run, f"no global_step_* under {ckpt_dir}"), None)
    step_dir, ckpt_tq = found
    # Layout containment BEFORE any lock (even the dry probe reads the lock file) or
    # mutation: symlinked live/backup TQ dirs, data.pt halves or marker/lock files
    # would redirect the tool's reads/renames/copies outside the checkpoint.
    layout_problem = _checkpoint_layout_problem(step_dir, ckpt_tq)
    if layout_problem is not None:
        return _finalize(
            _no_plan_report("invalid_checkpoint_layout", dry_run, layout_problem, step_dir=step_dir), ckpt_tq
        )

    # A restore already running on this checkpoint is reported BEFORE the controller
    # probe, in both modes. Its scratch TransferQueue is indistinguishable from a job's
    # (TQ pins the actor name and namespace), so probing first would report the more
    # generic live_controller_present and lose the holder's identity. The advisory read
    # is not the gate: the real mode's O_CREAT|O_EXCL below still decides the race.
    lock_path = step_dir / RESTORE_LOCK_FILE
    if lock_path.exists():
        held = _RestoreLockHeld(lock_path, _read_lock_holder(lock_path))
        detail = (
            f"a mutating restore holds {lock_path}; its plan would be stale — retry when released"
            if dry_run
            else str(held)
        )
        out = _no_plan_report("locked", dry_run, detail, step_dir=step_dir)
        out["lock"] = held.holder
        return _finalize(out, ckpt_tq)

    # STORE EXCLUSIVITY, before the preflight and in BOTH modes: reconciliation is only
    # correct against a buffer whose controller is gone, and dry-run is not exempt
    # because its own phase 2 calls tq.load_checkpoint on whatever controller tq.init
    # resolved. The probe runs BEFORE any isolation (see _live_controller): a get_actor
    # inside the private Ray the tool starts for itself holds nothing by construction.
    controller_probe: dict = {}
    controller = _live_controller(probe_record=controller_probe)
    if controller is not None and not allow_live_controller:
        out = _no_plan_report(
            "live_controller_present",
            dry_run,
            _live_controller_detail(controller, controller_probe),
            step_dir=step_dir,
        )
        out["controller"] = controller
        return _finalize(out, ckpt_tq, controller_probe)

    if dry_run:
        # ZERO writes of any kind — the lock itself would be one, so dry-run only
        # OBSERVES it (the advisory read above; the real mode's O_EXCL is the gate).
        plan = _preflight(
            store, journal_root, step_dir, ckpt_tq, force, force_store_mismatch, dry_run=True, journal_slot=journal_slot
        )
        if plan["status"] != "reconciled":
            return _finalize(_plan_report(plan), ckpt_tq, controller_probe)
        return _finalize(
            _reconcile(plan, store, partition, group_size, serializer, dry_run=True), ckpt_tq, controller_probe
        )

    try:
        with _exclusive_checkpoint_lock(step_dir):
            # The preflight runs UNDER the lock: the plan cannot go stale between
            # computation and execution.
            plan = _preflight(
                store,
                journal_root,
                step_dir,
                ckpt_tq,
                force,
                force_store_mismatch,
                dry_run=False,
                journal_slot=journal_slot,
            )
            if not plan["validated"]:
                # Fail closed on the JOURNAL: nothing is reconciled and no transaction
                # completes. The tool's own interrupted swap is healed anyway.
                return _finalize(_heal_only(plan), ckpt_tq, controller_probe)
            return _finalize(_execute(plan, store, partition, group_size, serializer), ckpt_tq, controller_probe)
    except _RestoreLockHeld as exc:
        out = _no_plan_report("locked", False, str(exc), step_dir=step_dir)
        out["lock"] = exc.holder
        return _finalize(out, ckpt_tq, controller_probe)


def _execute(plan: dict, store: Path, partition: str, group_size: int | None, serializer: str) -> dict:
    """PHASE 2 (real mode, under the exclusive lock): repairs, then — when the plan
    says so — the full fsck + publication.

    The preflight ``plan`` is IMMUTABLE: it is the exact view ``--dry-run`` reports,
    and the real result must carry it verbatim (one shared plan, both modes).
    Post-rename path fixups — the heal moves a checkpoint-backup source into place —
    live in ``work``, an execution-local deep copy handed to :func:`_reconcile` for
    EXECUTION while the immutable plan keeps driving the REPORT (the round-5 code
    mutated ``plan["source"]`` here, so ``source_journal`` differed dry vs real).
    """
    step_dir, ckpt_tq, backup = plan["step_dir"], plan["ckpt_tq"], plan["backup"]
    work = copy.deepcopy(plan)

    healed = False
    if plan["heal_publication"]:
        backup.rename(ckpt_tq)
        healed = True
        logger.warning("restore_tool: healed interrupted publication (restored %s from backup)", ckpt_tq)
        if work["source"] == backup:  # checkpoint-backup source: the heal moved it into place
            work["source"] = ckpt_tq
    completed, discarded = _execute_transaction_plan(
        step_dir, ckpt_tq, plan["txn_plan"], plan["store_id"], plan["journal_root"]
    )

    if plan["status"] != "reconciled":  # checkpoint_fresher / ordering_ambiguous
        out = _plan_report(plan)
        _attach_repairs(out, healed, completed, discarded)
        return out
    out = _reconcile(work, store, partition, group_size, serializer, dry_run=False, report_plan=plan)
    _attach_repairs(out, healed, completed, discarded)
    return out


def _attach_repairs(out: dict, healed: bool, completed: dict | None, discarded: dict | None) -> None:
    if healed:
        out["healed_publication"] = True
    if completed is not None:
        out["completed_interrupted_transaction"] = completed
    if discarded is not None:
        out["discarded_transaction"] = discarded


def _reconcile(
    plan: dict,
    store: Path,
    partition: str,
    group_size: int | None,
    serializer: str,
    dry_run: bool,
    report_plan: dict | None = None,
) -> dict:
    """The fsck + publication half of phase 2 (also the dry-run reporting path, which
    executes it with ``dry_run=True`` end to end: zero writes, complete plan).

    ``plan`` is the execution-local working state (its ``source`` may point at the
    post-heal location); ``report_plan`` — defaulting to ``plan`` itself, as in dry
    mode where nothing was renamed — is the IMMUTABLE preflight plan every report is
    built from, so dry and real results carry the identical plan view.
    """
    from .fs_kv import StoreIntegrityError, backend_config
    from .fsck import StoreMismatchError, UnrecoverableJournalError, recover

    report_plan = plan if report_plan is None else report_plan
    step_dir, ckpt_tq = plan["step_dir"], plan["ckpt_tq"]
    source = plan["source"]

    import transfer_queue as tq

    from . import fs_kv  # noqa: F401  (registration)

    t0 = time.perf_counter()
    # Isolate the scratch TQ: with no Ray of our own, tq.init's first act
    # (_init_from_existing -> ray.get_actor) auto-joins whatever RAY_ADDRESS names and
    # then IGNORES create=False, so the tool's handle would be a cluster controller's.
    # restore() probed the caller's Ray, or the discoverable cluster, and found no
    # controller there; the private instance keeps a cluster this node could not see
    # from being joined by accident either.
    _ensure_isolated_ray()
    # create=False: attach to the EXISTING store only. The preflight proved it exists
    # and carries a matching identity; bootstrap/client must never mkdir or stamp —
    # in dry-run that would be a write, and on a wrong --store path it would
    # manufacture the identity the fail-closed checks exist to demand.
    try:
        tq.init(backend_config(store, serializer=serializer, create=False))
    except (FileNotFoundError, StoreIntegrityError) as exc:
        # unreachable after a passing preflight unless the store vanished (or its
        # identity stamp was swapped for a symlink) in between; still a controlled
        # result, never a traceback (fail closed).
        out = _plan_report(report_plan, status="store_mismatch")
        out["detail"] = f"the store disappeared or lost its trustworthy identity stamp after the preflight: {exc}"
        return out
    try:
        try:
            report = recover(
                tq,
                store,
                source,
                partition,
                group_size=group_size,
                dry_run=dry_run,
                force_store_mismatch=plan["force_store_mismatch"],
                # --group-size is the operator's assertion that the run used a uniform
                # rollout.n; without it, prompts lacking a persisted per-prompt count
                # are never promoted (see promote_complete_groups).
                assume_uniform_group_size=group_size is not None,
                _precomputed_tail=plan["tail"],  # the preflight's VALIDATED tail
            )
        except StoreMismatchError as exc:
            out = _plan_report(report_plan, status="store_mismatch")
            out["detail"] = str(exc)
            return out
        except UnrecoverableJournalError as exc:
            out = _plan_report(report_plan, status="unrecoverable_journal")
            out["detail"] = str(exc)
            return out
        promotion = report.pop("promotion", None)

        result = _plan_report(report_plan, status="dry_run" if dry_run else "reconciled")
        result["source_step"] = _snapshot_step(source)
        result["fsck"] = report
        result["promotion"] = promotion

        if not dry_run:
            # Write the reconciled state next to where verl's stock resume looks, then
            # swap it in (original kept as *.pre_reconcile — see module docstring). The
            # TQ-dir swap and the data.pt replace are two separate irreversible steps,
            # so they run inside a reconcile.state transaction (BEGIN -> TQ_PUBLISHED
            # -> DONE) BOUND to its artifacts: the txn id AND the store lineage live in
            # the marker AND in the published dir's metadata, and the marker pins the
            # source journal's content digest. An interruption anywhere is resolved by
            # the next run BEFORE its freshness logic (see _execute_transaction_plan).
            txn = uuid.uuid4().hex
            store_id = plan["store_id"]
            source_identity = _snapshot_content_identity(source)

            def _mark(state: str) -> None:
                _write_reconcile_state(
                    step_dir, state, source, txn=txn, source_identity=source_identity, store_id=store_id
                )

            # staging dirs carry the txn in their name; stale ones from crashed runs
            # are swept here, under the lock (they are garbage by construction: a
            # crashed publication either never renamed its staging dir in, or renamed
            # it in — in which case it is no longer here).
            for stale in step_dir.glob(ckpt_tq.name + ".reconciled*"):
                shutil.rmtree(stale, ignore_errors=True)
            reconciled = step_dir / f"{ckpt_tq.name}.reconciled.{txn}"
            _PRE_BEGIN_CRASH_HOOK()
            _mark("BEGIN")
            _POST_BEGIN_CRASH_HOOK()
            tq.save_checkpoint(
                str(reconciled),
                metadata={
                    "global_steps": _snapshot_step(source) or 0,
                    "reconciled_by": _RECONCILED_BY,
                    "reconcile_txn": txn,
                    "reconcile_store_id": store_id,
                    "fence_ts": time.time(),
                },
            )
            # TQ's own protocol leftovers around the LIVE dir are strictly older than the
            # reconciled snapshot that now exists beside them, so they go BEFORE the swap:
            # the swap's two-rename window leaves the live dir temporarily absent, which is
            # exactly the state tq.load_checkpoint repairs by renaming a .old back into
            # place — over this tool's backup (see the function's docstring). Skipped when
            # the live dir is absent: there the sibling may be the only copy and TQ's
            # roll-forward is the intended repair.
            result["removed_tq_protocol_siblings"] = _sweep_tq_protocol_siblings(ckpt_tq) if ckpt_tq.exists() else []
            _publish_reconciled_dir(ckpt_tq, reconciled)
            _POST_TQ_PUBLISH_CRASH_HOOK()
            _mark("TQ_PUBLISHED")
            result["wrote"] = str(ckpt_tq)

            # ownership revalidation before the data.pt half + DONE: the live dir must
            # be OUR publication (belt and braces — the exclusive lock already
            # serializes restorers, so a failure here means out-of-band interference).
            stamp = _tq_dir_reconcile_stamp(ckpt_tq)
            if stamp is None or stamp.get("txn") != txn or stamp.get("store_id") != store_id:
                raise RuntimeError(
                    f"restore_tool: ownership revalidation failed after the TQ swap (live stamp "
                    f"{stamp!r} vs txn {txn!r} / lineage {store_id!r}): the checkpoint was mutated "
                    "out-of-band while this restore held the exclusive lock. The transaction is "
                    "left in TQ_PUBLISHED; the next run will complete or discard it on proof."
                )
            # v1.1: prefer the journal's dataloader position over the checkpoint's,
            # re-verified against the identity captured at BEGIN (slot-reuse guard).
            result["dataloader_state"] = _apply_dataloader_state(
                Path(source), step_dir, expected_identity=source_identity, expected_lineage=store_id
            )
            _PRE_DONE_CRASH_HOOK()
            _mark("DONE")
    finally:
        try:
            tq.close()
        except Exception:  # noqa: BLE001 - never mask the real failure with a close error
            logger.warning("restore_tool: tq.close() failed during cleanup", exc_info=True)

    result["seconds"] = round(time.perf_counter() - t0, 2)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True, type=Path, help="durable store root (backend's DurableFS.root)")
    parser.add_argument("--journal", type=Path, default=None, help="journal ring root (JournalWriter root)")
    parser.add_argument("--ckpt-dir", required=True, type=Path, help="trainer.default_local_dir")
    parser.add_argument("--partition", default="train")
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=(
            "rollout.n; enables the group-completeness rescue. Supplying it is an OPERATOR "
            "ASSERTION that the run used a uniform group size: prompts whose TQ record carries "
            "a persisted per-prompt __rollout_n__ are always judged by that value, but prompts "
            "with no persisted count are promoted at THIS size only because you assert it "
            "(without the flag such prompts are never promoted — a batch-level override that "
            "raised n is not persisted and cannot be detected)"
        ),
    )
    parser.add_argument("--serializer", default="tq", choices=("tq", "torch"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--journal-slot",
        default=None,
        help=(
            "reconcile from THIS ring slot (a journal-NNN directory name) instead of the newest "
            "trustworthy one. Without it the ring is walked newest-first and a slot whose "
            "artifacts fail validation falls back to the next-older complete slot, with every "
            "refusal reported under source_slot_rejections"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="apply the journal even when the checkpoint is as fresh or freshness is unknown",
    )
    parser.add_argument(
        "--force-store-mismatch",
        action="store_true",
        help=(
            "proceed when the journal's store identity is VALID but DIFFERENT from the store's "
            "(you vouch for a known foreign lineage — DANGEROUS). A missing/empty/invalid "
            "identity on either side always aborts: there is no lineage to vouch for, and an "
            "unstamped store is indistinguishable from a wrong --store path"
        ),
    )
    parser.add_argument(
        "--allow-live-controller",
        action="store_true",
        help=(
            "reconcile even though a TransferQueue controller is reachable (DANGEROUS: the running "
            "job keeps publishing objects this run would fence, and its buffer index is rewritten "
            "underneath it). Use only when the recorded controller is a leftover you have verified "
            "belongs to no live run. Reachability is decided against this process's own Ray when it "
            "has one, else against EVERY Ray instance this host can discover (RAY_ADDRESS, "
            "/tmp/ray/ray_current_cluster, and every local gcs_server), each joined only to probe "
            "and then left; one of them holding the controller is a refusal. A discovered address is "
            "a POINTER, not a live cluster: the first two sources outlive the cluster they name, so "
            "a stale one is pre-checked with a TCP connection and counts as no cluster rather than "
            "stalling the join, and a cluster that answers but cannot be joined is reported as doubt "
            "(refused, with the pointer to clear named in the detail) rather than as a sighting. "
            "This flag is not the remedy for that case: clearing the stale pointer is, and it keeps "
            "the check. A run on a cluster this host cannot discover or reach, or a writer "
            "publishing into --store from elsewhere, is NOT detected: there is no store-level "
            "writer lease, so store exclusivity is assumed and only the controller half is checked"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = restore(
        store=args.store,
        journal_root=args.journal,
        ckpt_dir=args.ckpt_dir,
        partition=args.partition,
        group_size=args.group_size,
        serializer=args.serializer,
        dry_run=args.dry_run,
        force=args.force,
        force_store_mismatch=args.force_store_mismatch,
        journal_slot=args.journal_slot,
        allow_live_controller=args.allow_live_controller,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if _exit_ok(result) else 1


#: Statuses that leave a resumable checkpoint and need no operator action.
_OK_STATUSES = ("reconciled", "dry_run", "checkpoint_fresher", "ordering_ambiguous")


def _exit_ok(result: dict) -> bool:
    """Zero exit only for an OK status that also left a live TransferQueue dir.

    The second half is the load-bearing one for an automated restart wrapper: any
    outcome that leaves the checkpoint without ``transfer_queue`` makes verl resume with
    an empty buffer and no warning (see :func:`_finalize`), so it must be a non-zero
    exit whatever the status says. ``healed_only`` is deliberately non-zero as well: the
    checkpoint is resumable again, but the journal was refused and that refusal is the
    operator's to resolve.
    """
    return result["status"] in _OK_STATUSES and result.get("checkpoint_tq_present") is not False


if __name__ == "__main__":
    sys.exit(main())
