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
"""Journal: frequent metadata-only snapshots of the TransferQueue controller.

With a durable storage backend the data is already safe; only the *metadata* (key ->
index map, tags, production/consumption status, sampler state) lives in the controller
process. A tick calls ``tq.save_checkpoint(dir, metadata=...)`` with the DEFAULT
``include_storage=True`` — for this backend the "storage checkpoint" is the fence
manifest (op-log offsets / ``fence.json``), not a data copy, so the snapshot stays
small (≈1.5 MB of controller pickle at 50k objects in the cost benchmark) and can be
taken every few seconds instead of once per trainer checkpoint. The loss window
becomes the journal cadence rather than ``trainer.save_freq``.

Two extras this module adds:

* **Ring of snapshots** (``root/journal-000`` …) so a crash *during* a snapshot cannot
  destroy the previous good one; ``latest_journal()`` picks the newest complete dir.
  The ring does NOT exist because ``tq.save_checkpoint`` leaves a window with no
  journal. Since TQ#155 it writes into ``<dir>.tmp``, renames any existing ``<dir>``
  aside to ``<dir>.old``, renames the tmp into place and then deletes the ``.old``
  (``transfer_queue/interface.py:1008-1052``), and ``load_checkpoint`` rolls forward
  from a surviving ``.old`` when ``<dir>`` is missing (``interface.py:1086-1092``). The
  ring earns its keep for three other reasons. (a) Per-tick history: freshness
  comparison and fence pairing need the PREVIOUS snapshot to still be readable, and
  TQ deletes its ``.old`` on success, so that is not a history. (b) A tick can fail
  after the controller pickle lands but before the offsets and stamp are written, which
  publishes an incomplete slot that ``latest_journal`` must skip in favour of an older
  complete one; with a single path there is nothing older to fall back to. (c) The slot
  a tick writes is not the slot a concurrent restore is reading, so this package's own
  artifacts (op-log offsets fence, dataloader state, stamp) are never overwritten in
  place underneath it. Independence from TQ's protocol is the point: the ring holds
  regardless of what TQ's ``.old`` lifecycle does, and this module never selects a
  ``.tmp`` or ``.old`` sibling as a slot (see :func:`latest_journal`).
* **Dataloader state** — optionally journals ``StatefulDataLoader.state_dict()`` next to
  the controller snapshot. Restoring the loader from the trainer checkpoint (older than
  the journal) makes the restart re-serve every prompt dispatched since that
  checkpoint, although their trajectories are already buffered — minutes of duplicate
  generation. Journaling the loader state shrinks that to the two bounded windows
  around a single tick (duplicate-dispatch and skip; see the class docstring and the
  README's guarantees) — it does NOT make prompt accounting exact.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import stat
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .fs_kv import OPLOG_OFFSETS_FILE

logger = logging.getLogger(__name__)

JOURNAL_PREFIX = "journal-"
STAMP_FILE = "journal_stamp.json"
DATALOADER_FILE = "dataloader_state.pt"
#: Writer lease: one ring root has exactly one journal writer (see
#: :class:`JournalWriter` and :func:`JournalWriter._acquire_lease`).
WRITER_LEASE_FILE = "journal_writer.lock"

# The EXACT ring-slot grammar ``JournalWriter._tick_locked`` emits: the prefix followed
# by the zero-padded slot number and nothing else. Selection must full-match this and
# never merely ``startswith(JOURNAL_PREFIX)``, because ``tq.save_checkpoint`` creates
# ``<slot>.tmp`` and ``<slot>.old`` SIBLINGS inside this same ring root while it swaps a
# slot into place (TQ#155, ``transfer_queue/interface.py:1008-1052``). A ``.tmp`` is
# rejected anyway (no stamp yet, since the stamp is written after save_checkpoint
# returns), but a ``.old`` is a complete previous snapshot and passes every completeness
# gate while carrying the NEWEST timestamp, so a prefix test selects it over the older
# real slot. Its content is genuine, so this is not corruption, but the path is outside
# the slot namespace that the ring's fence pairing and the restore tool's
# "slot recycled or rewritten since BEGIN" check reason about.
# ``[0-9]+`` rather than ``[0-9]{3}``: the formatter is ``{slot:03d}``, which is
# zero-PADDED, not truncated, so a ring larger than 1000 emits four digits. Both stay in
# the grammar and both exclude any name containing a dot. ``[0-9]`` rather than ``\d``,
# which in Python matches Unicode decimal digits (``journal-٠٠١`` would full-match and
# then never resolve to a slot the writer composed).
_SLOT_NAME_RE = re.compile(re.escape(JOURNAL_PREFIX) + r"[0-9]+")


def _regular_non_symlink_file(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode)


def _slot_stamp(slot: Path) -> dict | None:
    """The slot's parsed ``journal_stamp.json`` when the slot is COMPLETE, else ``None``.

    Completeness is decided by the last-written artifact (the stamp) plus the
    controller pickle, both of which must be regular non-symlink files: redirected
    artifacts would make recovery read a controller snapshot from outside the ring.
    """
    stamp = slot / STAMP_FILE
    if not _regular_non_symlink_file(stamp) or not _regular_non_symlink_file(slot / "controller_state.pkl"):
        return None  # incomplete (crash during snapshot) or redirected artifacts -> ignore
    try:
        payload = json.loads(stamp.read_text())
        float(payload["completed_at"])
    except (ValueError, KeyError, TypeError, OSError):  # pragma: no cover - torn stamp
        return None
    return payload if isinstance(payload, dict) else None


def _slot_order(stamp: dict) -> tuple[int, float]:
    """Ranking key for a complete slot: the writer's tick SEQUENCE, then its clock.

    ``tick_seq`` is a counter the writer persists in each stamp and continues above the
    highest sequence already in the ring, so it survives a driver restart and orders
    slots without trusting a clock. ``completed_at`` remains the tie-break and the
    fallback for slots written before the counter existed (they rank below any slot
    that carries one, which is right: a stamp with a sequence is newer code and, in a
    single-writer ring, a newer tick).
    """
    seq = stamp.get("tick_seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        seq = -1
    return seq, float(stamp["completed_at"])


def _ring_root(root: str | Path) -> tuple[Path, Path] | None:
    root = Path(root)
    if not root.exists():
        return None
    try:
        return root, root.resolve()
    except OSError:  # pragma: no cover - unreadable ring root
        return None


def _selectable_slot(candidate: Path, root: Path, root_resolved: Path) -> dict | None:
    """Validate one ring entry as a selectable slot; the parsed stamp, or ``None``."""
    if _SLOT_NAME_RE.fullmatch(candidate.name) is None:
        if candidate.name.startswith(JOURNAL_PREFIX):
            # A TQ protocol leftover (.tmp/.old) or a foreign name wearing the
            # prefix. Never a slot; log it so an operator sees the ring is not clean.
            logger.debug(
                "durable_buffer.journal: ignoring %s (not a ring slot name; never selected)",
                candidate,
            )
        return None
    try:
        st = os.lstat(candidate)
    except OSError:
        return None  # absent: an explicit selection naming a slot the ring does not hold
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        logger.warning("durable_buffer.journal: ignoring %s (not a real directory; never selected)", candidate)
        return None
    try:
        if not candidate.resolve().is_relative_to(root_resolved):  # belt and braces containment
            logger.warning("durable_buffer.journal: ignoring %s (resolves outside %s)", candidate, root)
            return None
    except OSError:  # pragma: no cover - unresolvable candidate
        return None
    return _slot_stamp(candidate)


def complete_journal_slots(root: str | Path) -> list[Path]:
    """Every complete slot under ``root``, NEWEST FIRST (see :func:`_slot_order`).

    :func:`latest_journal` is this list's head. The rest is what a consumer falls back
    to when the newest slot's artifacts cannot be trusted: the ring holds history, and
    before this existed nothing could reach it, so one corrupt slot made the whole
    journal unusable and the operator's only recovery was to ``mv`` the poisoned slot
    out of the ring mid-outage.
    """
    resolved = _ring_root(root)
    if resolved is None:
        return []
    root, root_resolved = resolved
    ranked: list[tuple[tuple[int, float], Path]] = []
    for d in root.iterdir():
        stamp = _selectable_slot(d, root, root_resolved)
        if stamp is not None:
            ranked.append((_slot_order(stamp), d))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [d for _, d in ranked]


def select_journal_slot(root: str | Path, slot: str) -> Path | None:
    """The named ring slot when it is selectable, else ``None``.

    ``slot`` is a slot directory name (``journal-001``); a path is accepted and only
    its final component is used, because a slot is always a child of the ring root.
    Validation is identical to :func:`latest_journal`'s, so an explicit selection can
    reach exactly the slots automatic selection can — the flag documents the manual
    step an operator otherwise performs with ``mv``, it does not widen what is
    recoverable.
    """
    resolved = _ring_root(root)
    if resolved is None:
        return None
    root, root_resolved = resolved
    name = Path(str(slot)).name
    candidate = root / name
    return candidate if _selectable_slot(candidate, root, root_resolved) is not None else None


def latest_journal(root: str | Path) -> Path | None:
    """Newest *complete* journal directory under ``root`` (or ``None``).

    Candidate validation is FAIL CLOSED at selection time: only a name that full-matches
    the ring-slot grammar (:data:`_SLOT_NAME_RE`) is even considered, so TQ's own
    checkpoint-protocol siblings (``journal-NNN.tmp`` / ``journal-NNN.old``, TQ#155) can
    never be selected as a slot; and a matching entry that is a symlink (or otherwise not
    a real directory resolving under the resolved ring root), or whose
    completion-critical artifacts (stamp, ``controller_state.pkl``) are not regular
    non-symlink files, is never selected either. A redirected slot would make recovery
    read a controller snapshot from OUTSIDE the ring. The artifacts recovery actually
    consumes are re-validated at use time by ``fsck.check_journal_recoverable``.

    Every journal consumer in this package routes its selection through this function
    or through :func:`complete_journal_slots` (``restore_tool._preflight`` and
    ``fsck``'s offline entry points included), so the grammar check is what keeps a
    ``.old`` sibling out of the restore tool's freshness comparison and out of its
    published source path.

    Ordering is the writer's persisted tick sequence, falling back to the stamp's
    ``completed_at`` (:func:`_slot_order`).

    Scope: SELECTION, and nothing beyond it. The chosen slot path is what
    ``fsck.recover`` hands to ``tq.load_checkpoint``, whose own ``.old`` roll-forward is
    keyed on ``<that path>.old`` (``interface.py:1088-1092``) and is not filtered by this
    grammar: refusing to SELECT a sibling is not the same as stopping TQ from rolling one
    forward, which it would do if the selected slot vanished between selection and load.
    Nothing here rewrites or removes anything either; the ring's own recycling
    (``JournalWriter``) and the restore tool's sweep are the writers.
    """
    slots = complete_journal_slots(root)
    return slots[0] if slots else None


class JournalPrecaptureError(RuntimeError):
    """The op-log offsets could not be captured BEFORE the controller snapshot.

    A tick that proceeded anyway would publish a complete-looking journal whose only
    offsets are the LATE ones captured inside the storage hook (after the controller
    pickle) — the unsafe direction: a delete+rewrite pair landing between the pickle
    and that capture escapes the fence, and under TQ's index reuse a restore would
    serve another sample's bytes. The tick must FAIL (no stamp file), so
    ``latest_journal`` skips the slot and recovery uses the previous good snapshot.
    """


class JournalWriterLeaseError(RuntimeError):
    """Another journal writer holds the lease on this ring root.

    A ring root has exactly ONE writer. Two writers both start at ``journal-000`` and
    interleave: TQ's ``save_checkpoint`` builds every snapshot in a fixed ``<dir>.tmp``
    (so one deletes the other's staging dir, which at least raises), and the artifact
    writes that follow it resolve their paths at write time, so one writer's offsets,
    dataloader state and stamp can land on top of the controller pickle the other just
    renamed into place. That slot passes every completeness gate while its fence is
    captured LATER than its controller state, which is the ordering the whole design
    exists to prevent.

    The lease makes the second writer fail at construction instead of silently
    interleaving. It is deliberately not a general mutual exclusion: a lease whose
    holder has stopped ticking is taken over (a crashed driver must not leave a ring
    root unusable on the restart path this package serves), and two writers racing that
    takeover would both proceed. The read-time cross-check of the tick's ``fence_ts``
    and writer id in ``journal_stamp.json`` against TQ's ``metadata.json``
    (``fsck.check_tick_coherence``) is what catches a mispaired slot in that case.
    """


def _current_oplog_offsets() -> dict[str, int] | None:
    """Op-log offsets of the durable store behind the live TQ client.

    Returns ``None`` only when the storage backend is NOT a KV client with an op log
    (e.g. SimpleStorage) — such snapshots carry no fence artifacts at all and this
    package's recovery refuses them at the identity preflight, so nothing is lost by
    proceeding. For an op-log-capable client, ANY capture failure raises
    :class:`JournalPrecaptureError` and the tick fails closed (no stamp published).

    Reached from whichever thread ticks. On the normal path nothing here touches ZMQ: it
    reads the process global that ``tq.get_client()`` returns and then calls a plain
    method on the durable store's own client, which stats the op-log files. The fallback
    branch below is the exception — ``tq.interface._maybe_create_tq_client`` CONSTRUCTS a
    ``TransferQueueClient``, which brings up a ``zmq.asyncio.Context`` and its event-loop
    thread, exactly as the first data-plane call would. It runs at most once, when the
    first tick beats the first data-plane call.

    The offsets themselves are op-log file sizes, with no locking anywhere in the store
    client: an append that another writer has buffered but not flushed makes a size read
    EARLY. Early offsets only over-fence, which is the safe direction (the same property
    ``write_manifest`` documents in those words). The RPC-marshalling argument for the
    rest of the tick is in the :class:`JournalWriter` docstring.
    """
    import transfer_queue as tq

    try:
        try:
            client = tq.get_client()
        except AssertionError:
            # First tick can beat the first data-plane call; TQ creates its client
            # lazily, so use the same factory save_checkpoint itself would use. Still
            # the path save_checkpoint takes at TQ@main (interface.py:1016) and still
            # accepts conf=None, resolving through _init_from_existing (interface.py:49,
            # :91-120).
            client = tq.interface._maybe_create_tq_client()
    except Exception as exc:  # noqa: BLE001 - fail closed: cannot prove a fence
        raise JournalPrecaptureError(f"cannot reach the TQ client for offset pre-capture: {exc}") from exc
    storage_client = getattr(getattr(client, "storage_manager", None), "storage_client", None)
    if storage_client is None or not hasattr(storage_client, "oplog_offsets"):
        return None  # not a KV-with-oplog backend: no offsets artifact applies to this snapshot
    if not getattr(storage_client, "oplog_enabled", True):
        raise JournalPrecaptureError(
            "the storage client reports its op log disabled; a journal without a "
            "trustworthy offsets fence would poison recovery"
        )
    try:
        return storage_client.oplog_offsets()
    except Exception as exc:  # noqa: BLE001 - fail closed: no stamp for this slot
        raise JournalPrecaptureError(f"op-log offset pre-capture failed: {exc}") from exc


class JournalWriter:
    """Takes metadata-only snapshots on a cadence, optionally in a background thread.

    Usage (driver process)::

        jw = JournalWriter("/shared/fs/runs/<run>/journal", interval_s=2.0,
                           dataloader_fn=lambda: trainer.train_dataloader.state_dict())
        jw.start()      # background thread; or call jw.maybe_tick() from a poll loop
        ...
        jw.stop()

    Thread-safety, re-derived against TransferQueue@main (2247b95): ticking from a
    non-owning thread is safe because every sync TQ call is MARSHALLED onto the client's
    own event loop, not because of anything about socket lifetime.
    ``interface._maybe_create_tq_client`` builds a ``TransferQueueClient``, the SYNC
    subclass (``transfer_queue/client.py:1194``), whose constructor starts a dedicated
    background event-loop thread (``client.py:1226-1229``) and binds every public sync
    method through ``_run_coroutine``, i.e.
    ``asyncio.run_coroutine_threadsafe(coro, self._loop).result()``
    (``client.py:1238-1241``). ``tq.save_checkpoint`` reaches TQ only through those sync
    wrappers (``save_controller_checkpoint`` -> ``_save_controller_checkpoint``,
    ``client.py:1708-1717``), so the coroutine and every socket it opens live entirely on
    the client's loop thread and never cross a thread boundary. The caller's thread only
    blocks on a future.

    Two premises from the fork-era comment here have changed, neither fatally. The
    ZMQ *context* is now one long-lived ``zmq.asyncio.Context`` per client
    (``client.py:99``, TQ#145) instead of one per call, which is safe by TQ's own
    statement that "contexts are thread-safe and loop-agnostic, so sharing one is safe"
    (``transfer_queue/utils/zmq_utils.py:367-370``). The *socket* is still per-call
    (``zmq_utils.py:407-428``, created inside the decorated coroutine), which is what
    keeps concurrent RPCs off a shared non-thread-safe socket. So the original
    conclusion holds; only its stated reason was incomplete.

    The one real constraint this leaves: ``with_controller_socket`` is bound without a
    timeout (``client.py:51-56``), so ``future.result()`` in ``_run_coroutine`` waits
    indefinitely on an unreachable controller. A wedged controller therefore parks the
    journal thread inside one tick while holding the tick lock, which makes subsequent
    ``maybe_tick`` calls no-op (counted as ``skipped_busy``) and makes ``stop()`` return
    after its bounded join with the daemon thread still parked. Journaling stalls;
    training is not blocked and no partial slot is published, since the stamp is written
    only after ``save_checkpoint`` returns. Whole ticks are additionally serialized on an
    internal lock (non-blocking: a tick requested while one is running is skipped, never
    queued), so the poll-driven ``maybe_tick`` backstop and the background thread can
    coexist.

    One writer per ring root, enforced by a lease file (:class:`JournalWriterLeaseError`)
    taken at construction and re-asserted at the start of every tick. Each published
    slot carries the writer id and the tick's ``fence_ts`` in BOTH TQ's
    ``metadata.json`` and ``journal_stamp.json``, which is what lets a reader reject a
    slot assembled from two ticks, plus a monotonic ``tick_seq`` that orders slots
    without trusting a clock and a schema pin (the TransferQueue version and a
    fingerprint of the controller-snapshot attributes reconciliation reads).

    Ordering inside a tick (both are correctness properties, tested):

    1. **Op-log offsets are captured BEFORE ``tq.save_checkpoint``** and written over
       the manager-captured ones afterwards. Early offsets mean fsck replays a few
       records that the controller snapshot already covers — those get fenced and
       regenerated (wasteful, safe). Late offsets would mean a delete+rewrite pair
       landing between the controller pickle and the offset capture escapes the fence
       entirely, which under TQ's index reuse serves another sample's bytes (unsafe).
       If the pre-capture FAILS, the tick fails (no stamp) — publishing with only the
       late offsets would poison recovery (:class:`JournalPrecaptureError`).
    2. **Dataloader state is captured BEFORE ``tq.save_checkpoint``.** Prompt
       accounting across a restore is NOT exact; it has exactly two bounded windows,
       and this ordering is what keeps both bounded (the reverse ordering — loader
       saved after the TQ cut — would silently lose every prompt fetched during the
       snapshot):

       * **Duplicate-dispatch window**: a prompt fetched AFTER the loader capture
         whose group still lands in the restored buffer is re-served on restart even
         though its trajectories are already there — duplicate generation (and
         potentially duplicate training under a fresh uid). Bounded by the dispatches
         completing between the loader capture and the TQ cut, i.e. within one tick.
         In THIS journal path the offsets pre-capture usually fences those groups
         (they were registered after the captured offsets), converting the duplicate
         into a regeneration; checkpoint-sourced restores (late offsets, loader from
         ``data.pt``) retain the duplicate direction.
       * **Skip window**: a dispatch spanning BOTH cuts — prompt fetched before the
         loader capture, registered after the fence cut — is recorded as served yet
         absent from the restored buffer, so a restore skips it for the current
         epoch. Bounded by the dispatches in flight at the tick instant; verl
         dispatches (fetch -> registration) synchronously on the trainer thread,
         which bounds it to about one dispatch. verl's group-refill keeps training
         batches full with other prompts, and the skipped prompt returns next epoch.
         Not detectable package-side (the prompt exists in neither the store nor the
         snapshot, and ``StatefulDataLoader`` state is opaque); closing it needs a
         dispatch WAL in verl or a controller-side WAL in TQ.

       There is deliberately NO blanket "never loses / never duplicates" claim
       anywhere in this package; see the README's guarantees section.
    """

    def __init__(
        self,
        root: str | Path,
        interval_s: float = 2.0,
        ring: int = 3,
        dataloader_fn: Callable[[], Any] | None = None,
        metadata_fn: Callable[[], dict] | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.interval_s = float(interval_s)
        self.ring = max(2, int(ring))
        self.dataloader_fn = dataloader_fn
        self.metadata_fn = metadata_fn
        self._slot = 0
        self._last_tick = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.stats = {"ticks": 0, "total_s": 0.0, "last_s": 0.0, "failures": 0, "skipped_busy": 0}
        # Writer identity and the ring lease: one writer per root (see
        # :class:`JournalWriterLeaseError`). The id is stamped into every slot this
        # writer publishes, on BOTH sides of the coherence pair.
        self.writer_id = uuid.uuid4().hex
        self._lease_path = self.root / WRITER_LEASE_FILE
        self._holds_lease = False
        self._acquire_lease()
        # Tick sequence: continue above the highest sequence already in the ring so
        # slot ordering survives a driver restart without trusting a clock.
        self._tick_seq = self._next_tick_seq()
        #: Observed controller-snapshot fingerprint, filled on the first tick
        #: (:meth:`_snapshot_schema`).
        self._snapshot_schema_fp: str | None = None

    # ------------------------------------------------------------------- lease
    def _lease_record(self) -> dict:
        return {
            "writer": self.writer_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "interval_s": self.interval_s,
            "heartbeat_at": time.time(),
        }

    def _stale_after(self) -> float:
        """How long a lease may go unrefreshed before another writer may take it over.

        Generous on purpose: the point of the takeover is to keep a crashed driver from
        wedging journaling forever, not to arbitrate between two live writers.
        """
        return max(30.0, self.interval_s * 5.0)

    def _read_lease(self) -> dict | None:
        try:
            payload = json.loads(self._lease_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_lease(self) -> None:
        """Publish the lease record atomically (same-dir tmp + ``os.replace``)."""
        tmp = self._lease_path.with_name(f"{WRITER_LEASE_FILE}.{os.getpid()}.{self.writer_id}.tmp")
        tmp.write_text(json.dumps(self._lease_record()), encoding="utf-8")
        os.replace(tmp, self._lease_path)

    def _acquire_lease(self) -> None:
        try:
            fd = os.open(self._lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = self._read_lease()
            beat = holder.get("heartbeat_at") if holder else None
            age = time.time() - float(beat) if isinstance(beat, int | float) else None
            if age is None or age <= self._stale_after():
                raise JournalWriterLeaseError(
                    f"another journal writer holds {self._lease_path} (holder {holder!r}); a ring "
                    "root has exactly one writer, because two interleave their post-snapshot "
                    "artifact writes and can publish a slot whose op-log fence was captured "
                    "after its controller snapshot. Point this writer at its own root, or — if "
                    "the recorded holder is genuinely dead — remove the lease file after "
                    "verifying that"
                ) from None
            logger.warning(
                "durable_buffer.journal: taking over the writer lease on %s; the recorded holder "
                "%r has not ticked for %.0fs (more than %.0fs), so it is dead or wedged",
                self.root,
                holder,
                age,
                self._stale_after(),
            )
            self._write_lease()
            self._holds_lease = True
            return
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self._lease_record(), fh)
        self._holds_lease = True

    def _refresh_lease(self) -> None:
        """Re-assert the lease at the START of a tick; raise if it changed hands.

        A lease taken over by someone else means a second writer is ticking into this
        ring, so this tick must not run: it would be exactly the interleaving the lease
        exists to stop. A lease that has been DELETED is re-created (with a warning)
        rather than treated as lost, because an operator clearing a stale-looking file
        should not silently stop journaling.
        """
        holder = self._read_lease()
        if holder is None and not self._lease_path.exists():
            logger.warning("durable_buffer.journal: writer lease %s disappeared; re-creating it", self._lease_path)
            self._write_lease()
            return
        if holder is not None and holder.get("writer") not in (None, self.writer_id):
            self._holds_lease = False
            raise JournalWriterLeaseError(
                f"the writer lease on {self.root} is now held by {holder!r}, not by this writer "
                f"({self.writer_id}); a second journal writer is ticking into this ring root and "
                "this tick would risk pairing its fence with the other writer's controller "
                "snapshot, so it fails instead (no slot is published)"
            )
        self._write_lease()

    def _release_lease(self) -> None:
        if not self._holds_lease:
            return
        self._holds_lease = False
        holder = self._read_lease()
        if holder is not None and holder.get("writer") != self.writer_id:
            return  # taken over by someone else: never remove another writer's lease
        self._lease_path.unlink(missing_ok=True)

    def _next_tick_seq(self) -> int:
        """One past the highest ``tick_seq`` any complete slot in the ring records."""
        highest = -1
        for slot in complete_journal_slots(self.root):
            stamp = _slot_stamp(slot)
            if stamp is None:  # pragma: no cover - raced away between listing and read
                continue
            seq, _ = _slot_order(stamp)
            highest = max(highest, seq)
        return highest + 1

    # ------------------------------------------------------------------ ticking
    def tick(self) -> dict | None:
        """Take one snapshot now (or skip if another tick is already running).

        Returns timing info, or ``None`` on skip/failure (a failed journal must never
        kill training).
        """
        if not self._lock.acquire(blocking=False):
            self.stats["skipped_busy"] += 1
            return None
        try:
            return self._tick_locked()
        finally:
            self._lock.release()

    def _tick_locked(self) -> dict | None:
        import transfer_queue as tq

        from .fsck import _transfer_queue_version

        slot = self._slot % self.ring
        self._slot += 1
        tick_seq = self._tick_seq
        self._tick_seq += 1
        target = self.root / f"{JOURNAL_PREFIX}{slot:03d}"
        fence_ts = time.time()  # stamped BEFORE the snapshot: concurrent writes get fenced
        t0 = time.perf_counter()
        try:
            # (0) re-assert the ring lease: a second writer that took it over would
            # interleave its artifact writes with this tick's (see _refresh_lease).
            self._refresh_lease()
            # (1) fence token first: offsets captured before the controller pickle can
            # only over-fence (see class docstring). A capture FAILURE raises and
            # fails the whole tick — no stamp is published, so latest_journal skips
            # this slot instead of consuming late (post-pickle) offsets.
            pre_offsets = _current_oplog_offsets()
            # (2) loader position next, still before the TQ cut: this bounds both
            # prompt-accounting windows (see class docstring point 2).
            dataloader_state = self.dataloader_fn() if self.dataloader_fn is not None else None
            # The coherence pair goes into TQ's own metadata and into this tick's stamp.
            # Caller metadata is applied FIRST so the reserved keys always describe the
            # tick that actually wrote the slot: a metadata_fn returning fence_ts would
            # otherwise break the pairing check that detects two writers on one root.
            metadata = dict(self.metadata_fn()) if self.metadata_fn is not None else {}
            metadata["fence_ts"] = fence_ts
            metadata["journal_writer"] = self.writer_id
            # (3) the metadata cut. The default include_storage=True is what we want:
            # for this backend "storage checkpoint" means "write the fence manifest",
            # not "copy the data".
            tq.save_checkpoint(str(target), metadata=metadata)
            if pre_offsets is not None:
                (target / OPLOG_OFFSETS_FILE).write_text(json.dumps(pre_offsets), encoding="utf-8")
            if dataloader_state is not None:
                import torch

                torch.save(dataloader_state, target / DATALOADER_FILE)
            # The stamp is written LAST (it is what makes the slot selectable) and
            # carries: the coherence pair, the monotonic tick sequence that orders
            # slots without a clock, and the schema pin — the TransferQueue version
            # this pickle was produced by plus a fingerprint of the controller-snapshot
            # attributes THIS SNAPSHOT CARRIES, read back from the slot just written.
            # TQ's pickle has no schema version of its own, so without the pin an
            # upgrade between the snapshot and the restore is undetectable
            # (fsck.check_snapshot_schema judges it).
            (target / STAMP_FILE).write_text(
                json.dumps(
                    {
                        "completed_at": time.time(),
                        "fence_ts": fence_ts,
                        "journal_writer": self.writer_id,
                        "tick_seq": tick_seq,
                        "tq_version": _transfer_queue_version(),
                        "snapshot_schema": self._snapshot_schema(target),
                    }
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - a failed journal must never kill training
            self.stats["failures"] += 1
            logger.warning("durable_buffer.journal: tick failed (%s: %s)", type(exc).__name__, exc)
            return None
        dt = time.perf_counter() - t0
        self.stats["ticks"] += 1
        self.stats["total_s"] += dt
        self.stats["last_s"] = dt
        self._last_tick = time.time()
        return {"dir": str(target), "tick_s": round(dt, 4), "fence_ts": fence_ts, "tick_seq": tick_seq}

    def _snapshot_schema(self, target: Path) -> str:
        """Fingerprint of the controller-snapshot attributes the installed TransferQueue
        actually produces, observed on the slot this tick just wrote.

        Read back from the pickle rather than derived from ``fsck.SNAPSHOT_ATTRIBUTES``,
        because a fingerprint computed from a tuple hardcoded in verl has the same value
        either side of any TransferQueue upgrade and therefore cannot detect the drift the
        pin exists for (``fsck.observed_snapshot_schema_fingerprint``).

        Computed once per writer and cached: the snapshot's shape comes from the
        TransferQueue classes loaded in this process, which cannot change while it runs,
        so paying an extra unpickle of a snapshot that is O(live metadata) on every tick
        would buy nothing. A read-back that finds no controller pickle, or one it cannot
        parse, is left to raise: either way the slot carries no usable controller snapshot
        and ``latest_journal`` would not select it, so the tick fails and publishes no
        stamp rather than leaving an unpinned slot behind.
        """
        if self._snapshot_schema_fp is None:
            from .fsck import snapshot_schema_fingerprint_of

            self._snapshot_schema_fp = snapshot_schema_fingerprint_of(target)
        return self._snapshot_schema_fp

    def maybe_tick(self) -> dict | None:
        """Tick if ``interval_s`` has elapsed (for callers with their own poll loop)."""
        if time.time() - self._last_tick < self.interval_s:
            return None
        return self.tick()

    # ------------------------------------------------------------ background use
    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.tick()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tq-journal", daemon=True)
        self._thread.start()
        logger.info("durable_buffer.journal: started, root=%s interval=%.1fs", self.root, self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
            self._thread = None
        self._release_lease()

    # ----------------------------------------------------------------- reporting
    def summary(self) -> dict:
        ticks = max(self.stats["ticks"], 1)
        return {
            # A writer that exists is journaling. The signal matters for its absence:
            # both wiring sites degrade to journaling-off when the ring lease is held
            # (register.start_journal returns None, DurableReplayBuffer keeps
            # self._journal None), and the buffer reports the same key as 0 there, so a
            # dashboard shows a run whose journal never started.
            "journaling_enabled": True,
            "journal_ticks": self.stats["ticks"],
            "journal_mean_s": round(self.stats["total_s"] / ticks, 4),
            "journal_last_s": round(self.stats["last_s"], 4),
            "journal_failures": self.stats["failures"],
            # Whether slots actually carry the loader position. The wiring that
            # supplies it is best-effort discovery through the trainer, so an upstream
            # refactor can switch it off without breaking anything else: the only
            # symptom would be duplicate generation after a crash, discovered in the
            # postmortem. Reported so it reaches the trainer's metrics under the
            # durable_buffer/ prefix and shows up on a dashboard beforehand.
            "dataloader_journaled": self.dataloader_fn is not None,
        }
