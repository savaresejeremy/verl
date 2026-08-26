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
"""Restore-time reconciliation ("fsck") between a TQ journal and a durable store.

Why this is needed at all: the journal is a *metadata* snapshot taken at time T while
the store keeps changing (trajectories are written as produced, consumed samples are
deleted). On restart the controller state is as of T and the store is as of the crash,
so the two must be reconciled before verl samples anything.

Rules
-----
``DROP``   Every ``(index, field)`` the journal expects but the store no longer has:
           that sample was consumed (trained) after T, so its metadata must go. The set
           is closed over trajectory groups — a group that lost one member loses the
           rest, and a group with nothing left loses its prompt too — so no restore can
           publish a half group that nothing will ever complete.
``FENCE``  Every store object the journal does not expect, **or whose write identity
           changed since T**. The identity clause is the correctness-critical one: TQ
           recycles freed sample indexes, so an index the journal still attributes to a
           consumed sample can hold a *different* sample's bytes after the crash.
           Fencing on index membership alone silently serves mislabeled data.
``PROMOTE`` (optional) A ``running`` prompt whose group is fully present with all
           sessions ``success`` is re-tagged ``finished`` instead of being re-issued —
           verl publishes the terminal prompt tag only after every session settles, so
           a crash in that window would otherwise clear and regenerate a complete group.

Snapshot/mutation straddles resolve to fence/drop (wasteful, never wrong): TQ mutates
STORAGE and CONTROLLER metadata in two steps (put: object published, then metadata;
clear: object unlinked, then metadata), so a snapshot can land in between. An object
the snapshot does not expect is FENCED (put-straddle: regenerated); metadata whose
object is gone is DROPPED (clear-straddle: otherwise the restored controller would
re-serve a key whose read can only fail). Both directions hold in every fence mode —
the op-log mode adds a names-only store listing (no per-object ``stat``) exactly for
this membership check. A clear batch unlinks index-major, so a snapshot can also land
part-way through one and expect a MIX of gone and present objects; the group closure on
the DROP rule is what keeps that from surviving as a permanently incomplete group (the
three states are worked through above :class:`Journal`).

SCOPE OF THE WRITE-IDENTITY GUARANTEE: "never serves mislabeled bytes" holds for
EARLY-OFFSET snapshots — those whose op-log offsets were captured BEFORE the
controller pickle, which is what ``JournalWriter.tick`` does (early offsets only
over-fence, the safe direction). The direct trainer-checkpoint fallback (verl's own
``tq.save_checkpoint`` at ``save_freq``, used when no journal exists or the journal
is not fresher) captures its offsets INSIDE the storage hook, AFTER the controller
pickle: a delete-and-rewrite of one name landing in that gap is invisible to the
replay, so a checkpoint-sourced restore carries a residual (milliseconds-wide)
window in which rewritten bytes can be served under the snapshotted key. That
window is inherent to verl's call order, not closable here; see the README's
"Guarantees, exactly" for the same qualification on the operator-facing claim.

Fence sourcing is FAIL CLOSED: op-log replay is the only recovery-grade source, and
any untrusted artifact (missing/malformed offsets, shrunk/vanished logs, an
``oplog: false`` snapshot from the removed full-manifest mode) ABORTS recovery with
:class:`UnrecoverableJournalError` — there is no automatic degradation.
``mode="mtime"`` exists only as an explicitly-requested legacy/audit flow.

Liveness/expectation sources (learned the hard way): use
``partition.global_indexes`` and ``partition.field_metadata[f].global_indexes``.
Do **not** use ``index_manager["allocated_indexes"]`` — it holds only the most recent
allocation batch, and fencing on it discards most of a live buffer.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import stat
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .fs_kv import (
    FENCE_FILE,
    MANIFEST_FILE,
    OBJECT_SUFFIX,
    OPLOG_DIR,
    OPLOG_OFFSETS_FILE,
    StoreIntegrityError,
    _dec,
    _enc_key,
    read_store_id,
)

logger = logging.getLogger(__name__)

FENCE_MODE_OPLOG = "oplog"  # O(changes since the snapshot) — the ONLY default recovery fence
FENCE_MODE_MTIME = "mtime"  # explicit legacy/audit escape hatch; never auto-selected
FENCE_MODE_INDEX = "index"  # unsafe under index reuse; kept for A/B experiments


class StoreMismatchError(RuntimeError):
    """The journal cannot be proven to belong to the store being reconciled.

    Reconciling store B against store A's journal would fence most of B and drop most
    of the journal's metadata — a destructive no-op. Raised before ANY mutation, and
    raised FAIL-CLOSED: a missing, empty or malformed identity on either side is a
    mismatch (only a positively matching pair proceeds), because an absent stamp is
    exactly what a wrong ``--store`` path or a hand-assembled journal looks like.

    The store identity is *logical lineage*, not location: it is stamped once at
    bootstrap (``_store_id``) and travels with the bytes, so a copied/moved store
    keeps its identity and its journals stay usable. It is deliberately separate from
    the freshness override (``--force`` in the restore tool): ``--force`` says "apply
    an older/unordered journal to MY store", ``force_store_mismatch`` says "I know
    this journal came from a different lineage — reconcile anyway" (DANGEROUS).

    ``force_store_mismatch`` overrides ONLY a mismatch between two VALID identities
    (the operator vouches for a known, different lineage). A missing/empty/invalid
    identity on either side is NEVER overridable: there is no lineage to vouch for —
    an unstamped store is indistinguishable from a wrong ``--store`` path, and the
    attach-only tooling (``create=False``) cannot even open it.
    """


class UnrecoverableJournalError(RuntimeError):
    """The snapshot's fence artifacts cannot prove write identity — recovery aborts.

    Raised FAIL-CLOSED, with no automatic degradation, when:

    * the snapshot records ``oplog: false`` (the removed full-manifest tick mode: its
      manifest was written AFTER the controller pickle, so a same-name overwrite in
      the gap is recorded as valid and a restore could serve replacement bytes under
      an old key);
    * the offsets file is missing or fails strict schema validation (a mapping of
      ``str -> nonnegative int``, nothing else);
    * an op log shrank, disappeared, contains bytes that are not valid UTF-8, or
      contains a record that is malformed or names anything other than a canonical
      encoded backend-object basename (``{digits}@{encoded-field}.bin`` — no path
      separators, no dot-prefixes/traversal, no absolute paths). Record names feed
      ``unlink`` calls under the store root, so a log that cannot be proven to name
      only store objects must not fence (or delete) anything.

    In every case the untrusted artifact could otherwise let a post-snapshot
    same-name rewrite be served as the snapshotted object. :func:`recover` raises
    this BEFORE ``tq.load_checkpoint``, so the caller's controller is never left
    holding metadata that points at unfenced bytes. There is deliberately no
    ``force`` for this error; ``mode="mtime"`` exists as an explicitly-labeled
    legacy/audit flow for stores with no usable fence artifacts at all.
    """


def _require_regular_artifact(path: Path, what: str) -> None:
    """FAIL CLOSED unless ``path`` is absent or a regular, non-symlink file.

    Absence stays legal — every reader has its own missing-file branch (and some
    artifacts are optional). Anything present that is not a plain regular file
    aborts: recovery-critical artifacts are read with following ``open`` calls, so a
    symlink here would make recovery consume bytes from OUTSIDE the snapshot dir.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:  # pragma: no cover - uninspectable artifact
        raise UnrecoverableJournalError(
            f"{what} {path} cannot be inspected ({exc}); recovery aborts (fail closed)"
        ) from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise UnrecoverableJournalError(
            f"{what} {path} is not a regular non-symlink file (mode {stat.filemode(st.st_mode)}); "
            "recovery-critical artifacts must be exactly what the writer created — a link would "
            "redirect the read outside the snapshot, so recovery aborts (fail closed)"
        )


def _read_regular_file(path: Path, what: str, *, root: Path | None = None, binary: bool = False) -> str | bytes:
    """The ONE validated reader for snapshot/journal artifacts (round-8): lstat says
    ``path`` is a regular non-symlink file (:func:`_require_regular_artifact`) and —
    when ``root`` is given — it resolves under ``root``; only then is it read.

    Any present-but-untrustworthy node raises :class:`UnrecoverableJournalError`;
    ``FileNotFoundError`` propagates so every caller keeps its own absence
    semantics. Every direct artifact read outside :func:`recover`'s preflight
    (exported :func:`load_journal`, the :func:`_read_fence_metadata` probe, the
    offline :func:`_read_manifest`) routes through this gate, so a symlinked
    artifact can never donate bytes from outside its snapshot dir no matter which
    entry point reaches it. Validation is check-then-read — the residual TOCTOU is
    the same documented limitation as the store-side checks (see ``fs_kv``).
    """
    _require_regular_artifact(path, what)
    if root is not None:
        try:
            resolved = path.resolve()
        except OSError as exc:  # pragma: no cover - unresolvable artifact path
            raise UnrecoverableJournalError(f"{what} {path} cannot be resolved ({exc}); fail closed") from exc
        if not resolved.is_relative_to(root.resolve()):
            raise UnrecoverableJournalError(
                f"{what} {path} resolves to {resolved}, outside {root}; recovery aborts (fail closed)"
            )
    return path.read_bytes() if binary else path.read_text(encoding="utf-8")


#: Every artifact recovery (or the restore tool) READS out of a snapshot/journal dir.
#: Validated as regular non-symlink files by :func:`check_journal_recoverable` — the
#: shared first gate of recover(), fsck() and the restore tool's preflight.
def _snapshot_artifacts() -> tuple[tuple[str, str], ...]:
    from .journal import DATALOADER_FILE, STAMP_FILE  # function-local: journal imports fs_kv only

    return (
        (FENCE_FILE, "fence stamp"),
        (OPLOG_OFFSETS_FILE, "offsets fence"),
        (MANIFEST_FILE, "audit manifest"),
        ("metadata.json", "snapshot metadata"),
        ("controller_state.pkl", "controller snapshot"),
        (STAMP_FILE, "journal stamp"),
        (DATALOADER_FILE, "journaled dataloader state"),
    )


def check_journal_recoverable(journal_dir: str | Path) -> None:
    """FAIL-CLOSED validation of the snapshot's ``fence.json`` recovery-mode stamp.

    A plain file read (no TQ required), so offline tools can — and ``recover``/the
    restore tool do — run it before ``tq.load_checkpoint`` or any other mutation.
    Since round 7 it FIRST validates the snapshot's shape: the dir itself and every
    recovery-critical artifact in it (fence stamp, offsets, metadata, controller
    pickle, journal stamp, dataloader state) must be real non-symlink nodes — a
    symlink would resolve the read outside the snapshot
    (:class:`UnrecoverableJournalError`; absence keeps its per-reader meaning).

    ``fence.json`` must decode to a JSON OBJECT whose ``oplog`` field is strict-bool
    ``True`` — exactly what ``write_manifest`` stamps. Anything else aborts with
    :class:`UnrecoverableJournalError`: a missing file, malformed JSON, a non-object
    payload (which would crash ``.get`` readers later), a missing ``oplog`` field,
    ``null``, ``0``, ``"false"`` and the removed ``oplog: false`` tick mode itself.
    The stamp is the snapshot's own assertion that its fence artifacts are
    op-log-grade; a snapshot that cannot positively make that assertion must not be
    recovered from (the old ``is False`` check accepted every one of those shapes).

    Two further fail-closed gates run last, both keyed on artifacts a journal tick
    writes and a trainer checkpoint does not (so a checkpoint source is unaffected):
    :func:`check_tick_coherence` refuses a slot whose stamp and TQ metadata came from
    different ticks (two writers on one ring root), and
    :func:`check_snapshot_schema` refuses a snapshot whose recorded
    controller-snapshot fingerprint is not the one this code reads.
    """
    journal_dir = Path(journal_dir)
    # Containment first (round-7 comment 2): the snapshot dir itself and every
    # artifact recovery might read out of it must be exactly what the writer
    # created — real (non-symlink) nodes. A symlinked dir or artifact resolves the
    # read OUTSIDE the snapshot; absence stays legal (each reader has its own
    # missing-file branch, judged below and by the offsets/journal loaders).
    try:
        dir_st = os.lstat(journal_dir)
    except FileNotFoundError:
        dir_st = None  # missing dir -> the fence read below raises the no-fence refusal
    except OSError as exc:  # pragma: no cover - uninspectable snapshot dir
        raise UnrecoverableJournalError(f"snapshot {journal_dir} cannot be inspected ({exc}); recovery aborts") from exc
    if dir_st is not None:
        if stat.S_ISLNK(dir_st.st_mode) or not stat.S_ISDIR(dir_st.st_mode):
            raise UnrecoverableJournalError(
                f"snapshot {journal_dir} is not a real (non-symlink) directory "
                f"(mode {stat.filemode(dir_st.st_mode)}); a redirected snapshot must not be "
                "recovered from — recovery aborts (fail closed)"
            )
        for name, what in _snapshot_artifacts():
            _require_regular_artifact(journal_dir / name, what)
    fence_path = journal_dir / FENCE_FILE
    try:
        raw = fence_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise UnrecoverableJournalError(
            f"snapshot {journal_dir} carries no {FENCE_FILE}; without the recovery-mode "
            "stamp the snapshot cannot prove its fence artifacts are op-log-grade, so "
            "recovery aborts (fail closed)"
        ) from None
    except UnicodeDecodeError as exc:
        # NOT an OSError and raised by read_text (not json.loads), so without this
        # clause it escaped as a raw traceback instead of the fail-closed refusal.
        raise UnrecoverableJournalError(
            f"fence stamp {fence_path} is not valid UTF-8 ({exc}); recovery aborts (fail closed)"
        ) from exc
    except OSError as exc:  # pragma: no cover - unreadable file
        raise UnrecoverableJournalError(f"fence stamp {fence_path} unreadable: {exc}") from exc
    try:
        fence = json.loads(raw)
    except ValueError as exc:
        raise UnrecoverableJournalError(
            f"fence stamp {fence_path} is not valid JSON ({exc}); recovery aborts (fail closed)"
        ) from exc
    if not isinstance(fence, dict):
        raise UnrecoverableJournalError(
            f"fence stamp {fence_path} must be a JSON object, got {type(fence).__name__}; recovery aborts (fail closed)"
        )
    oplog = fence.get("oplog")
    if oplog is False:
        raise UnrecoverableJournalError(
            f"snapshot {journal_dir} was taken with the op log disabled (the removed "
            "'oplog: false' mode). Its full manifest is written AFTER the controller "
            "snapshot, so a same-name overwrite in that gap is recorded as valid and a "
            "restore could serve replacement bytes under an old key. Recovery from such "
            "snapshots is not supported; restore from a snapshot taken with the op log "
            "enabled."
        )
    if oplog is not True:
        raise UnrecoverableJournalError(
            f"fence stamp {fence_path} does not positively record oplog: true "
            f"(got {oplog!r}); a snapshot that cannot assert an op-log-grade fence must "
            "not be recovered from — recovery aborts (fail closed)"
        )
    check_tick_coherence(journal_dir)
    check_snapshot_schema(journal_dir)


#: The controller-snapshot attribute names :func:`load_journal` reads. TQ's controller
#: pickle is ``{"controller_id", "partitions": {pid: p.to_snapshot()}, ...}`` where
#: ``to_snapshot`` is ``cls.__new__(cls)`` plus a shallow ``__dict__`` copy
#: (``transfer_queue/controller.py``), so it carries live references to TQ's own
#: partition-status class and NO schema version. verl installs TransferQueue from
#: ``@main``, which means a container rebuild between the crash and the restore can
#: rename or move any of these attributes; the cross-version evidence for this backend
#: covers stored object BYTES (the storage codec), not the controller pickle. A journal
#: tick therefore stamps ``transfer_queue.__version__`` into ``journal_stamp.json``
#: together with a fingerprint of the attributes THE SNAPSHOT ITSELF CARRIES
#: (:func:`snapshot_schema_fingerprint_of`, read back from the pickle the tick just
#: wrote), and :func:`check_snapshot_schema` refuses a snapshot whose fingerprint is not
#: the one THIS code reads. Deriving the recorded value from the real snapshot is what
#: makes a TransferQueue-side rename detectable: a fingerprint computed from this tuple
#: alone would be identical either side of any TransferQueue upgrade.
SNAPSHOT_ATTRIBUTES = ("custom_meta", "field_metadata", "global_indexes", "keys_mapping", "revert_keys_mapping")


def snapshot_schema_fingerprint() -> str:
    """Fingerprint of the controller-snapshot attributes THIS CODE READS.

    The reader half of the schema pin: what :func:`load_journal` needs to find on a
    snapshot object. The written half is :func:`snapshot_schema_fingerprint_of`, which
    reports what a snapshot ACTUALLY carries, and
    :func:`check_snapshot_schema` compares the two.

    Sorted and joined rather than hashed, so a mismatch message names the two field
    sets and an operator can see exactly which attribute moved.
    """
    return ",".join(sorted(SNAPSHOT_ATTRIBUTES))


def observed_snapshot_schema_fingerprint(snapshots: Iterable[object]) -> str:
    """Fingerprint of the attributes reconciliation reads that ``snapshots`` really have.

    Computed from the live snapshot objects rather than from :data:`SNAPSHOT_ATTRIBUTES`,
    which is the whole point: a fingerprint derived from a tuple hardcoded in verl has
    the same value either side of a TransferQueue upgrade, so it can only detect a change
    to verl's own reader contract, never the drift it exists to catch. An attribute
    reconciliation reads that TQ renames is simply absent from the object, drops out of
    this fingerprint, and no longer matches :func:`snapshot_schema_fingerprint` at
    restore time.

    A name counts as present only when EVERY snapshot object has it, because
    ``load_journal`` walks every partition. Presence is ``hasattr``, not a ``__dict__``
    key, for the same reason: attribute access is how the attribute is read. With no
    snapshot objects at all there is nothing to observe, so the reader contract is
    recorded unchanged — such a snapshot carries no partitions and is refused on those
    terms (``partition not in parts``) rather than by a schema verdict invented from no
    evidence.
    """
    snapshots = list(snapshots)
    if not snapshots:
        return snapshot_schema_fingerprint()
    present = [name for name in sorted(SNAPSHOT_ATTRIBUTES) if all(hasattr(snap, name) for snap in snapshots)]
    return ",".join(present)


def snapshot_schema_fingerprint_of(journal_dir: str | Path) -> str:
    """:func:`observed_snapshot_schema_fingerprint` of a slot's controller pickle.

    Called by the journal tick against the slot it has just written, so the recorded
    fingerprint describes the snapshot in that slot. Raises whatever the parse raises:
    a controller pickle the WRITING process cannot read back is a slot that would poison
    recovery, and the tick failing (no stamp, slot not selectable) is the honest outcome.
    """
    raw = _read_regular_file(Path(journal_dir) / "controller_state.pkl", "controller snapshot", binary=True)
    return observed_snapshot_schema_fingerprint(pickle.loads(raw)["partitions"].values())


def _read_json_object(path: Path, what: str) -> dict | None:
    """A snapshot artifact as a JSON object; ``None`` when absent.

    Fail closed on anything present but unusable (not UTF-8, not JSON, not an object):
    these artifacts drive the coherence and schema judgements below, and a reader that
    treats a malformed one as "no opinion" would wave through exactly the slots those
    judgements exist to reject.
    """
    try:
        raw = _read_regular_file(path, what)
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as exc:
        raise UnrecoverableJournalError(f"{what} {path} is not valid UTF-8 ({exc}); recovery aborts") from exc
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise UnrecoverableJournalError(f"{what} {path} is not valid JSON ({exc}); recovery aborts") from exc
    if not isinstance(payload, dict):
        raise UnrecoverableJournalError(
            f"{what} {path} must be a JSON object, got {type(payload).__name__}; recovery aborts (fail closed)"
        )
    return payload


def _user_metadata(journal_dir: Path) -> dict | None:
    """TQ's ``metadata.json`` ``user_metadata`` block (the tick's own metadata)."""
    payload = _read_json_object(journal_dir / "metadata.json", "snapshot metadata")
    if payload is None:
        return None
    inner = payload.get("user_metadata", payload)
    return inner if isinstance(inner, dict) else {}


def check_tick_coherence(journal_dir: str | Path) -> None:
    """Refuse a slot whose stamp and TQ metadata were written by DIFFERENT ticks.

    ``JournalWriter`` serializes ticks on a per-INSTANCE lock, and its post-save
    artifact writes (offsets, dataloader state, stamp) resolve their paths after
    ``tq.save_checkpoint`` has already renamed the controller pickle into place. Two
    writers on one ring root therefore both target ``journal-000`` first and can
    interleave: writer A lands its offsets and stamp on top of the pickle writer B
    just published. The result passes every completeness gate — real directory, real
    stamp, real pickle, ``fence.json`` with ``oplog: true`` — while its fence is
    captured LATER than its controller state, which is precisely the ordering the
    design refuses to recover from whenever it can see it.

    One tick writes ONE ``fence_ts`` into both TQ's ``metadata.json``
    (``metadata={"fence_ts": ...}``) and ``journal_stamp.json``, so for a coherent
    slot the two values are the identical float and for a mispaired slot they differ.
    The writer lease id travels the same way and is compared when both sides carry it,
    which catches a mispairing even between ticks that happened to share a timestamp.
    A slot missing either artifact (a trainer checkpoint carries no journal stamp) is
    not judged here — there is no pair to compare.
    """
    journal_dir = Path(journal_dir)
    from .journal import STAMP_FILE

    stamp = _read_json_object(journal_dir / STAMP_FILE, "journal stamp")
    if stamp is None:
        return
    md = _user_metadata(journal_dir)
    if md is None:
        return
    for field_name in ("fence_ts", "journal_writer"):
        mine, theirs = stamp.get(field_name), md.get(field_name)
        if mine is None or theirs is None:
            continue
        if mine != theirs:
            raise UnrecoverableJournalError(
                f"snapshot {journal_dir} is not the product of a single journal tick: its "
                f"{STAMP_FILE} records {field_name}={mine!r} while its metadata.json records "
                f"{theirs!r}. Two journal writers sharing one ring root interleave their "
                "post-save artifact writes, which can pair one tick's op-log fence with "
                "another tick's controller snapshot — a fence captured after the pickle it "
                "is supposed to cover. Recovery aborts (fail closed); a ring root has "
                "exactly one writer"
            )


def check_snapshot_schema(journal_dir: str | Path) -> None:
    """Refuse a snapshot whose controller-state schema is not the one this code reads.

    The recorded fingerprint says which of :data:`SNAPSHOT_ATTRIBUTES` the snapshot
    OBJECTS actually carried when the tick wrote them
    (:func:`snapshot_schema_fingerprint_of`); the current one says which this reader
    needs (:func:`snapshot_schema_fingerprint`). A MISMATCH therefore covers both
    directions of drift: TransferQueue renaming or moving an attribute reconciliation
    reads (the recorded set loses a name), and verl changing what it reads. Either way
    ``load_journal`` would crash or, worse, read a renamed attribute's stale namesake and
    reconcile against drifted semantics, so it is fail-closed.

    A bare ``transfer_queue.__version__`` difference with a MATCHING fingerprint only
    warns: the fields reconciliation depends on are all present and unmoved, so the
    snapshot is readable.

    A snapshot with no fingerprint at all (written before this stamp existed, or a
    trainer checkpoint, which carries no journal stamp) is not judged: there is
    nothing to compare, and refusing would make every pre-existing journal
    unrecoverable.
    """
    journal_dir = Path(journal_dir)
    from .journal import STAMP_FILE

    stamp = _read_json_object(journal_dir / STAMP_FILE, "journal stamp")
    if stamp is None:
        return
    recorded = stamp.get("snapshot_schema")
    if recorded is None:
        return
    current = snapshot_schema_fingerprint()
    recorded_version = stamp.get("tq_version")
    current_version = _transfer_queue_version()
    if recorded != current:
        raise UnrecoverableJournalError(
            f"snapshot {journal_dir} was taken against a different TransferQueue controller "
            f"schema: its controller objects carry {recorded!r} (TransferQueue {recorded_version!r}) "
            f"while this code reads {current!r} (TransferQueue {current_version!r}). The controller "
            "pickle carries live references to TransferQueue's own classes and no schema version of "
            "its own, so reconciling across that change would read attributes that have moved "
            "— recovery aborts (fail closed). Restore with the TransferQueue version that "
            "wrote the snapshot, or regenerate the buffer"
        )
    if recorded_version is not None and current_version is not None and recorded_version != current_version:
        logger.warning(
            "durable_buffer.fsck: snapshot %s was taken under TransferQueue %s and is being "
            "restored under %s; the controller-snapshot fields reconciliation reads (%s) are "
            "unchanged, so the snapshot is readable — proceeding",
            journal_dir,
            recorded_version,
            current_version,
            current,
        )


def _transfer_queue_version() -> str | None:
    """``transfer_queue.__version__`` if the installed TQ exposes one."""
    try:
        import transfer_queue
    except Exception:  # noqa: BLE001 - offline tooling may inspect a journal without TQ
        return None
    version = getattr(transfer_queue, "__version__", None)
    return str(version) if version is not None else None


_STORE_ID_RE = re.compile(r"[0-9a-f]{32}")  # exactly what _ensure_store_id writes (uuid4 hex)


def _valid_store_id(value) -> bool:
    return isinstance(value, str) and _STORE_ID_RE.fullmatch(value) is not None


@dataclass
class PartitionView:
    """Per-partition key<->index maps (reconciliation drops metadata per partition)."""

    keys_mapping: dict  # user key -> global index
    revert: dict  # global index -> user key


#: TQ#141 ``mark_clearing`` and what a snapshot can observe mid-clear.
#:
#: TQ@main clears samples in THREE controller-visible steps, not two
#: (``transfer_queue/client.py:542-576``, ``async_clear_samples``):
#:
#:   1. ``_mark_clearing_in_controller`` (``client.py:579-594``) -> the controller's
#:      ``mark_clearing`` (``controller.py:1469-1502``, ZMQ handler ``:2038``);
#:   2. ``storage_manager.clear_data(metadata)`` -> for this backend, an op-log ``D``
#:      record for the whole batch and then the unlinks (``fs_kv.DurableFSClient.clear``);
#:   3. ``_clear_meta_in_controller`` -> ``clear_meta``, which removes the indexes from
#:      the partition's metadata.
#:
#: What step 1 leaves behind is narrower than the name suggests. ``mark_clearing`` does
#: exactly one thing: ``partition.production_status[existing, :] = 0``
#: (``controller.py:1501-1502``). There is NO clearing-marked set in
#: ``DataPartitionStatus`` (fields at ``controller.py:340-365``), so ``to_snapshot``
#: (``controller.py:871-889``, a deepcopy of ``__dict__`` with tensors cloned) has no such
#: set to copy. A marked index therefore stays in ``global_indexes``, in
#: ``keys_mapping`` / ``revert_keys_mapping``, in ``field_metadata[f].global_indexes`` and
#: in ``custom_meta``, and the ONLY difference a snapshot carries is a zeroed
#: ``production_status`` row.
#:
#: :func:`load_journal` reads none of that: it reads ``global_indexes``,
#: ``field_metadata[*].global_indexes``, ``keys_mapping``, ``revert_keys_mapping`` and
#: ``custom_meta``, and never ``production_status`` or ``consumption_status``. So a
#: snapshot taken anywhere inside the mark/delete window is INDISTINGUISHABLE, to this
#: module, from one taken before the clear began. Yes to the question "can an index be
#: marked-clearing yet still present in ``global_indexes`` at snapshot time, with its
#: object deleted afterwards": that is precisely a snapshot between steps 1 and 2 whose
#: unlinks (step 2) then complete before the crash.
#:
#: A snapshot can land in THREE places inside that window, and they do not all resolve
#: the same way. The first two are resolved by the rules that already resolve the
#: two-step straddles in the module docstring; the third needs the group closure in
#: :func:`_lost_group_siblings`:
#:
#:   * snapshot between steps 1 and 2 (objects still present, metadata present): the
#:     snapshot is self-consistent. If step 2 runs afterwards, the op-log ``D`` record
#:     lands AFTER the snapshot's captured offsets, so the tail replay reports the name in
#:     ``deleted_after`` and :func:`_fsck_oplog` finishes the delete and DROPs the
#:     metadata (see the ``deleted_after`` branch: "consumption of that sample already
#:     began, so finishing the delete avoids duplicate training"). If the crash lands
#:     before step 2, nothing was deleted and the sample survives, correctly.
#:   * snapshot after step 2 completed (objects gone, metadata present): the classic
#:     clear-straddle. The store listing has no object for an ``(index, field)`` the
#:     snapshot expects, so it is missing -> **DROP**.
#:   * snapshot INSIDE step 2's unlink loop (some of the batch's objects gone, the rest
#:     still there, metadata present): a mixed state, and the one that needs the group
#:     closure. ``DurableFSClient.clear`` / ``clear_by_index`` write ONE ``D`` record for
#:     the whole batch and FLUSH it before the first unlink (``fs_kv.py:544-556``,
#:     ``:565-577``), and ``JournalWriter.tick`` pre-captures the offsets before
#:     ``tq.save_checkpoint``, so the whole batch sits BEHIND the fence: nothing is
#:     replayed as ``deleted_after``. The already-unlinked prefix DROPs by the
#:     missing-object rule; the not-yet-unlinked suffix is present AND expected, so on
#:     its own it would be KEPT. For a finished 2-trajectory group whose unlink loop was
#:     interrupted after one object, that leaves 1 of 2 trajectories with a ``finished``
#:     prompt: the group can never be completed (production ended), nothing re-issues a
#:     finished prompt, and TQ's ``load_checkpoint`` installs the snapshot's zeroed
#:     ``production_status`` verbatim (``controller.py:2234``) so the metadata says
#:     nothing was ever produced. :func:`_lost_group_siblings` therefore drops the whole
#:     group whenever one member is missing, and the orphan-prompt rule then drops the
#:     prompt, which hands the work back to the re-issue path.
#:
#: The mixed state is the only one that needed new behaviour, and specifically none of the
#: three is an ORPHAN object (the FENCE direction): step 2 deletes objects, it never
#: creates them, so marking cannot produce store content the snapshot fails to expect.
#: The negative half of the claim (nothing about ``mark_clearing`` reaches the fields this
#: module reads) is what stays a comment plus the tests in
#: ``tests/experimental/durable_buffer/test_tq_main_compat_on_cpu.py``.
#:
#: One consequence worth stating because it is invisible here: a restored controller does
#: NOT inherit a zeroed ``production_status`` in a way that matters, because the metadata
#: is dropped in every state where an object went (whole group at a time). This module
#: deliberately does not READ ``production_status`` to make that decision: a zeroed row is
#: indistinguishable from an index registered by an in-flight put whose object is not
#: written yet, so keying on it would drop live work. Nor does it try to repair the field,
#: which would mean reconciling state whose only writer is TQ.
@dataclass
class Journal:
    """The subset of a TQ controller snapshot that reconciliation needs.

    ``live``/``expected`` are **store-wide** (union over every partition in the
    snapshot) because one durable store backs all partitions of a run — a ``train``
    restore must not fence ``val``'s objects. For the same reason ``per_partition``
    carries every partition's key maps: a missing object drops its metadata in
    WHICHEVER partition owns it, not just the one being restored.
    """

    live: set
    expected: set  # {(global_index, field_name)}
    per_partition: dict  # partition -> PartitionView
    fence_ts: float | None
    partitions: tuple = ()
    tags: dict = field(default_factory=dict)
    store_id: str | None = None

    def view(self, partition: str) -> PartitionView:
        return self.per_partition[partition]


def load_journal(journal_dir: str | Path, partition: str = "train") -> Journal:
    """Parse a snapshot's controller state into the reconciliation view.

    Exported (offline tooling calls it directly, outside :func:`recover`'s
    preflight), so its artifact reads are individually gated by
    :func:`_read_regular_file`: a symlinked ``controller_state.pkl`` or
    ``metadata.json`` raises :class:`UnrecoverableJournalError` instead of being
    read through (``fence.json`` goes through the :func:`_read_fence_metadata`
    probe, which treats a redirected stamp as absent — every consumer then fails
    closed on the missing identity).

    A pickle that cannot be parsed, or whose payload does not carry the partition and
    the controller-snapshot attributes reconciliation reads
    (:data:`SNAPSHOT_ATTRIBUTES`), is an UNTRUSTED ARTIFACT and raises
    :class:`UnrecoverableJournalError` like every other one: this is a pure file read
    with no TQ involved, so :func:`recover` runs it in the preflight and a corrupt or
    version-drifted pickle can no longer be discovered AFTER ``tq.load_checkpoint``
    has already loaded it into the caller's live controller.
    """
    journal_dir = Path(journal_dir)
    try:
        return _load_journal_unchecked(journal_dir, partition)
    except (FileNotFoundError, UnrecoverableJournalError, StoreMismatchError):
        raise  # already classified (missing artifact keeps its per-caller meaning)
    except (
        pickle.UnpicklingError,
        AttributeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        ModuleNotFoundError,
        EOFError,
    ) as exc:
        raise UnrecoverableJournalError(
            f"controller snapshot in {journal_dir} cannot be parsed into the reconciliation "
            f"view ({type(exc).__name__}: {exc}); the snapshot's metadata cannot be trusted "
            "against this store, so recovery aborts (fail closed) before any mutation. A "
            "TransferQueue upgrade between the snapshot and this restore is the usual cause "
            f"(this code reads {snapshot_schema_fingerprint()!r} under "
            f"TransferQueue {_transfer_queue_version()!r})"
        ) from exc


def _load_journal_unchecked(journal_dir: Path, partition: str) -> Journal:
    """The parse itself; :func:`load_journal` classifies whatever it raises."""
    state = pickle.loads(_read_regular_file(journal_dir / "controller_state.pkl", "controller snapshot", binary=True))
    parts = state["partitions"]
    if partition not in parts:
        raise KeyError(f"partition {partition!r} not in journal {journal_dir} (has {sorted(parts)})")

    live: set = set()
    expected: set = set()
    per_partition: dict[str, PartitionView] = {}
    for pname, part in parts.items():  # store-wide: never fence another partition's objects
        part_live = set(part.global_indexes)
        live |= part_live
        expected |= {
            (idx, fname)
            for fname, meta in part.field_metadata.items()
            for idx in meta.global_indexes
            if idx in part_live
        }
        per_partition[pname] = PartitionView(
            keys_mapping=dict(part.keys_mapping),
            revert=dict(part.revert_keys_mapping),
        )

    mine = parts[partition]
    fence_ts = None
    try:
        raw_md = _read_regular_file(journal_dir / "metadata.json", "snapshot metadata")
    except FileNotFoundError:
        raw_md = None
    if raw_md is not None:
        try:
            md = json.loads(raw_md)
            fence_ts = md.get("user_metadata", md).get("fence_ts")
        except (ValueError, AttributeError):  # pragma: no cover
            fence_ts = None
    fence_md = _read_fence_metadata(journal_dir)
    store_id = fence_md.get("store_id")
    if fence_ts is None:
        fence_ts = fence_md.get("fence_ts")
    return Journal(
        live=live,
        expected=expected,
        per_partition=per_partition,
        fence_ts=fence_ts,
        partitions=tuple(sorted(parts)),
        # Plain attribute access, like the other four: custom_meta is listed in
        # SNAPSHOT_ATTRIBUTES as an attribute reconciliation reads, so a getattr default
        # here would absorb exactly the rename the schema pin exists to refuse. The
        # AttributeError is classified as unrecoverable by load_journal, in the preflight,
        # before anything is loaded into a controller.
        tags={i: dict(t) for i, t in mine.custom_meta.items()},
        store_id=store_id,
    )


def _read_fence_metadata(journal_dir: Path) -> dict:
    """The snapshot's ``fence.json`` payload (``{}`` when missing/unreadable/non-object).

    Best-effort reader for identity/freshness probing only — non-object JSON is
    normalized to ``{}`` so ``.get`` callers cannot crash on it, and a symlinked or
    otherwise non-regular stamp is REJECTED like a missing one (``{}``) instead of
    being read through: every consumer then fails closed on the absent field (no
    journal ``store_id`` -> :class:`StoreMismatchError`, never overridable; no
    lineage -> the restore tool's completion declines and keeps the checkpoint's
    ``data.pt``). The fail-closed judgement of the stamp itself lives in
    :func:`check_journal_recoverable`.
    """
    fence_path = Path(journal_dir) / FENCE_FILE
    try:
        payload = json.loads(_read_regular_file(fence_path, "fence stamp"))
    except (FileNotFoundError, ValueError):
        return {}
    except UnrecoverableJournalError as exc:
        logger.warning("durable_buffer.fsck: %s — treating the stamp as absent (fail closed)", exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def check_store_identity(store_root: str | Path, journal_dir: str | Path, force: bool = False) -> None:
    """FAIL-CLOSED identity preflight; runs before any controller or store mutation.

    Reads only the identity artifacts (never the controller pickle), so callers can —
    and :func:`recover`/the restore tool do — run it BEFORE ``tq.load_checkpoint``.
    Raises :class:`StoreMismatchError` unless both sides carry a valid identity AND
    they are equal. ``force=True`` (``--force-store-mismatch``) overrides ONLY the
    valid-but-different case (the operator vouches for a known foreign lineage);
    missing/empty/invalid identities are never overridable — there is no lineage to
    vouch for, and the attach-only tooling cannot even open an unstamped store.
    """
    journal_id = _read_fence_metadata(journal_dir).get("store_id")
    try:
        store_id = read_store_id(store_root)
    except StoreIntegrityError as exc:
        # The store-side lineage authority itself cannot be trusted (a symlinked
        # ``_store_id`` could let an EXTERNAL file — even one holding the journal's
        # exact UUID — vouch for the store in front of destructive reconciliation).
        # Classified as a mismatch and NEVER overridable: force_store_mismatch
        # vouches for a known lineage, and an untrustworthy stamp has none.
        raise StoreMismatchError(
            f"store {store_root} identity cannot be trusted: {exc}. Reconciling could destroy this "
            "store's contents, so recovery aborts before any mutation; this is never overridable "
            "(there is no lineage to vouch for through a redirected identity stamp)"
        ) from exc
    problem = None
    if not _valid_store_id(journal_id) and not _valid_store_id(store_id):
        problem = f"neither the journal ({journal_id!r}) nor the store ({store_id!r}) carries a valid identity"
    elif not _valid_store_id(journal_id):
        problem = f"the journal carries no valid store identity ({journal_id!r}; store has {store_id})"
    elif not _valid_store_id(store_id):
        problem = f"store {store_root} carries no valid identity ({store_id!r}; journal has {journal_id})"
    elif journal_id != store_id:
        problem = f"journal store_id {journal_id} != store {store_root} id {store_id}"
        if force:
            logger.warning(
                "durable_buffer.fsck: %s; reconciling could destroy this store's contents "
                "(forced via force_store_mismatch, proceeding)",
                problem,
            )
            return
    if problem is None:
        return
    msg = (
        f"{problem}; reconciling could destroy this store's contents. Identity is the store's "
        "logical lineage (a copied store keeps it); it is checked fail-closed. "
        "force_store_mismatch (--force-store-mismatch) overrides only a mismatch between two "
        "VALID identities — a missing/empty/invalid identity is never overridable, because an "
        "unstamped store is indistinguishable from a wrong --store path."
    )
    if force and (not _valid_store_id(journal_id) or not _valid_store_id(store_id)):
        msg += " The override was requested but does not apply here."
    raise StoreMismatchError(msg)


def _parse_traj_key(key: str) -> tuple[str, int, int] | None:
    """``{uid}_{session}_{index}`` -> (uid, session, index); ``None`` for non-trajectory keys.

    The uid may itself contain underscores, so parse from the RIGHT and require the two
    tail segments to be integers (verl's agent loop emits exactly this shape).
    """
    parts = key.rsplit("_", 2)
    if len(parts) != 3:
        return None
    uid, session, index = parts
    if not uid:
        return None
    try:
        return uid, int(session), int(index)
    except ValueError:
        return None


def _canonical_object_stem(stem: str) -> tuple[int, str] | None:
    """Decode a name stem and prove it is EXACTLY what the writer would emit.

    The escape grammar alone accepts spellings the writer never produces — an
    over-encoding like ``1@%41`` decodes to the same key as ``1@A``, and two
    spellings of one ``(index, field)`` pair would alias in the membership sets. So
    the stem is decoded and RE-ENCODED through the writer's own shared function
    (``fs_kv._enc_key``) and must match byte-for-byte; the index must additionally
    be the canonical decimal the writer composes (``str(int)`` — no leading zeros),
    and the name must be valid UTF-8 (scandir can smuggle surrogates for non-UTF-8
    on-disk bytes, which no writer-emitted name contains).
    """
    try:
        stem.encode("utf-8")  # strict: reject surrogate-carrying (non-UTF-8) names
        decoded = _dec(stem)
    except (UnicodeEncodeError, ValueError):
        return None
    if "@" not in decoded:
        return None
    idx_str, fname = decoded.split("@", 1)
    try:
        idx = int(idx_str)
    except ValueError:
        return None
    if str(idx) != idx_str:
        return None  # leading zeros / '+' — the writer composes str(int(index))
    if _enc_key(decoded) != stem:
        return None  # a non-canonical encoding the writer never emits
    return idx, fname


def _parse_object_name(name: str) -> tuple[int, str] | None:
    """``(index, field)`` for a CANONICAL writer-emitted object file name, else None.

    Grammar first (:data:`_CANONICAL_OBJECT_NAME_RE`), then the strict re-encode
    equality (:func:`_canonical_object_stem`): a store entry in a non-canonical
    spelling is treated as not-an-object rather than aliased onto a canonical key.
    """
    if _CANONICAL_OBJECT_NAME_RE.fullmatch(name) is None:
        return None
    return _canonical_object_stem(name[: -len(OBJECT_SUFFIX)])


# Exactly what the writers compose and log (``DurableFSClient._path``/``clear_by_index``):
# ``{decimal index}@{encoded field}`` + OBJECT_SUFFIX. The encoded alphabet contains no
# path separator, no NUL, no '\t'/'\n'/'\r' (all escaped as uppercase %XX by ``_enc``)
# and never starts with '.', so a matching name cannot traverse ('..'), be absolute,
# hide in the dot namespace, or smuggle a second path component.
_CANONICAL_OBJECT_NAME_RE = re.compile(r"[0-9]+@(?!\.)(?:%[0-9A-F]{2}|[^%/\x00\t\n\r])+" + re.escape(OBJECT_SUFFIX))


def _validate_oplog_record_name(name: str, log_name: str) -> None:
    """FAIL CLOSED: an op-log record may only name a canonical store-object basename.

    Every name returned by :func:`_read_oplog_tail` later reaches
    ``(store_root / name).unlink()`` in the fence step, so a record like
    ``P ../outside.bin`` or ``P /etc/passwd`` would delete OUTSIDE the store. Writers
    only ever log names they composed themselves, so anything non-canonical proves
    the log is corrupt or tampered with — abort before any name is returned. Beyond
    the grammar, the name is decoded and re-encoded through the writer's own shared
    encoding function and must match byte-for-byte
    (:func:`_canonical_object_stem`): the escape alphabet also matches spellings the
    writer never emits (an over-encoded ``1@%41.bin`` aliases ``1@A.bin``), and a
    log carrying one cannot be a writer's log.
    """
    if _CANONICAL_OBJECT_NAME_RE.fullmatch(name) is None or _canonical_object_stem(name[: -len(OBJECT_SUFFIX)]) is None:
        raise UnrecoverableJournalError(
            f"op-log record in {log_name} names a non-canonical object {name[:120]!r} "
            "(records may only name encoded backend-object basenames exactly as the "
            f"writer emits them: digits@field{OBJECT_SUFFIX}, no path separators, no "
            "absolute paths, no '..'/dot prefixes, no over-encoded escapes); the log "
            "cannot be trusted and recovery aborts (fail closed) before any mutation"
        )


def _read_manifest(journal_dir: Path) -> dict[str, tuple[int, int]] | None:
    """Reader for the OFFLINE AUDIT manifest (``write_manifest(full=True)``).

    Recovery never consumes this artifact (it is written after the controller
    pickle, so it can record gap overwrites as valid — the reason the ``oplog:
    false`` tick mode was removed); it exists for offline store audits and tests.
    The read is gated by :func:`_read_regular_file`: a symlinked/non-regular
    manifest is as unusable as a malformed one (``None``, with a warning) — an
    audit must not report rows read from OUTSIDE the snapshot dir.
    Split on literal ``'\\n'`` ONLY: encoded object names cannot contain ``\\n``/``\\r``
    /``\\t`` but MAY contain other Unicode line separators (U+0085/U+2028/U+2029/\\v/\\f)
    that ``str.splitlines()`` would split on, corrupting records for legal names. A
    malformed line means the manifest cannot be trusted — return ``None``.
    """
    path = journal_dir / MANIFEST_FILE
    try:
        text = _read_regular_file(path, "audit manifest")
    except FileNotFoundError:
        return None
    except UnrecoverableJournalError as exc:
        logger.warning("durable_buffer.fsck: %s; manifest unusable", exc)
        return None
    manifest: dict[str, tuple[int, int]] = {}
    for line in text.split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            logger.warning("durable_buffer.fsck: malformed manifest line %r; manifest unusable", line[:120])
            return None
        name, inode, size, _mtime = parts
        try:
            manifest[name] = (int(inode), int(size))
        except ValueError:
            logger.warning("durable_buffer.fsck: malformed manifest line %r; manifest unusable", line[:120])
            return None
    return manifest


def _validate_offsets(journal_dir: Path) -> dict[str, int]:
    """STRICT schema validation of the snapshot's offsets fence — fail closed.

    The accepted schema is exactly what ``write_manifest``/``JournalWriter`` produce:
    a JSON object mapping op-log file name (``str``) to a nonnegative ``int`` byte
    offset. Anything else — missing file, malformed JSON, a non-object payload, a
    non-integer / boolean / negative / fractional offset — raises
    :class:`UnrecoverableJournalError`: an untrusted offset could hide a post-cut
    same-name rewrite from the replay (fencing from the wrong position), which under
    TQ's index reuse serves another sample's bytes. It must never fall through to a
    weaker fence source.
    """
    offsets_path = journal_dir / OPLOG_OFFSETS_FILE
    _require_regular_artifact(offsets_path, "offsets fence")  # direct tail readers get the same gate
    try:
        raw = offsets_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise UnrecoverableJournalError(
            f"snapshot {journal_dir} carries no op-log offsets fence ({OPLOG_OFFSETS_FILE}); "
            "recovery cannot establish write identity and aborts (fail closed). If this "
            "store genuinely has no usable fence artifacts, mode='mtime' exists as an "
            "explicitly-labeled legacy flow that conservatively over-fences by wall clock."
        ) from None
    except UnicodeDecodeError as exc:
        # read_text raises this BEFORE json.loads can; it is not an OSError either,
        # so it needs its own fail-closed clause (same class of escape as fence.json).
        raise UnrecoverableJournalError(
            f"offsets fence {offsets_path} is not valid UTF-8 ({exc}); recovery aborts (fail closed)"
        ) from exc
    except OSError as exc:  # pragma: no cover - unreadable file
        raise UnrecoverableJournalError(f"offsets fence {offsets_path} unreadable: {exc}") from exc
    try:
        offsets = json.loads(raw)
    except ValueError as exc:
        raise UnrecoverableJournalError(
            f"offsets fence {offsets_path} is not valid JSON ({exc}); recovery aborts (fail closed)"
        ) from exc
    if not isinstance(offsets, dict):
        raise UnrecoverableJournalError(
            f"offsets fence {offsets_path} must be an object mapping log name to byte "
            f"offset, got {type(offsets).__name__}; recovery aborts (fail closed)"
        )
    for name, value in offsets.items():
        if not isinstance(name, str) or type(value) is not int or value < 0:
            raise UnrecoverableJournalError(
                f"offsets fence {offsets_path} has invalid entry {name!r}: {value!r} "
                "(offsets must be nonnegative integers); recovery aborts (fail closed)"
            )
    return offsets


#: Backward scan window for :func:`_align_start_to_record_boundary`, doubled until a
#: boundary is found. One op-log record is a two-character op plus an encoded object
#: name, so 64 KiB spans thousands of them and the first window all but always hits.
_BOUNDARY_SCAN_WINDOW = 1 << 16


def _align_start_to_record_boundary(fh, start: int, log_name: str) -> int:
    """Round ``start`` DOWN to the first byte after the preceding record boundary.

    The offsets fence is a file SIZE sampled by the journal in the driver while
    writers append from other nodes. ``_log_ops`` flushes at the end of every call, so
    a log rests on a record boundary between calls and a locally observed size lands
    on one. Nothing guarantees that for a size sampled DURING a large multi-page
    append and observed from a different node, where the size is derived from
    server-side state that advances as the write proceeds.

    Aborting on such an offset would refuse a valid, internally coherent journal,
    which is the worst available outcome when the alternative is free: replaying from
    the previous boundary re-reads the straddling record and treats it as post-cut,
    which OVER-fences — the direction every other decision in this package prefers.
    An offset with no boundary anywhere behind it replays the whole log, which fences
    more and loses nothing. A record that is still unparseable after alignment keeps
    its abort.
    """
    if start <= 0:
        return 0
    window = _BOUNDARY_SCAN_WINDOW
    while True:
        begin = max(0, start - window)
        fh.seek(begin)
        chunk = fh.read(start - begin)
        cut = chunk.rfind(b"\n")
        if cut != -1:
            return begin + cut + 1
        if begin == 0:
            logger.warning(
                "durable_buffer.fsck: op log %s has no record boundary before the snapshot "
                "offset %d; replaying the whole log (over-fencing, safe)",
                log_name,
                start,
            )
            return 0
        window *= 2


def _read_oplog_tail(store_root: Path, journal_dir: Path) -> tuple[set[str], set[str]]:
    """Replay every writer's op log *past* the offsets recorded at snapshot time.

    Returns ``(written_after, deleted_after)`` object names. FAIL CLOSED: any
    untrustworthy artifact — invalid offsets (:func:`_validate_offsets`), a log that
    shrank or vanished, bytes that are not valid UTF-8, a malformed record, or a
    record naming anything but a canonical store-object basename
    (:func:`_validate_oplog_record_name` — the names feed ``unlink`` under the store
    root, so a traversal like ``P ../x`` must never be returned) — raises
    :class:`UnrecoverableJournalError` instead of degrading to a weaker fence
    source. A plain filesystem read (no TQ), so ``recover`` runs it BEFORE
    ``tq.load_checkpoint``.

    A recorded offset that falls INSIDE a record is not an untrusted artifact: it is
    aligned down to the previous boundary and the straddling record is replayed as
    post-snapshot (:func:`_align_start_to_record_boundary`).
    """
    offsets = _validate_offsets(journal_dir)
    oplog_dir = store_root / OPLOG_DIR
    try:
        dir_st = os.lstat(oplog_dir)
    except FileNotFoundError:
        dir_st = None
    if dir_st is None:
        if not offsets:
            return set(), set()  # no writers had logs at the snapshot and none exist now
        raise UnrecoverableJournalError(
            f"op-log directory {oplog_dir} disappeared but the snapshot recorded offsets "
            f"for {sorted(offsets)}; the delta since the snapshot cannot be reconstructed"
        )
    # The log INFRASTRUCTURE itself must be exactly what the writers create: a real
    # directory holding regular files. A symlinked dir or log would replay records
    # from OUTSIDE the store — an untrusted artifact, so recovery aborts before any
    # controller or store mutation (recover() runs this before tq.load_checkpoint).
    if stat.S_ISLNK(dir_st.st_mode) or not stat.S_ISDIR(dir_st.st_mode):
        raise UnrecoverableJournalError(
            f"op-log directory {oplog_dir} is not a real (non-symlink) directory; a redirected "
            "log tree cannot fence anything and recovery aborts (fail closed) before any mutation"
        )
    resolved = oplog_dir.resolve()
    if not resolved.is_relative_to(store_root.resolve()):
        raise UnrecoverableJournalError(
            f"op-log directory {oplog_dir} resolves to {resolved}, outside the store root "
            f"{store_root}; recovery aborts (fail closed) before any mutation"
        )

    written: set[str] = set()
    deleted: set[str] = set()
    seen_files = set()
    for entry in sorted(oplog_dir.iterdir()):
        if not entry.name.endswith(".log"):
            continue
        log_st = entry.lstat()
        if stat.S_ISLNK(log_st.st_mode) or not stat.S_ISREG(log_st.st_mode):
            raise UnrecoverableJournalError(
                f"op log {entry.name} is not a regular non-symlink file "
                f"(mode {stat.filemode(log_st.st_mode)}); its records would come from "
                "outside the store and recovery aborts (fail closed) before any mutation"
            )
        seen_files.add(entry.name)
        start = offsets.get(entry.name, 0)
        size = log_st.st_size
        if size < start:
            raise UnrecoverableJournalError(
                f"op log {entry.name} shrank ({size} < recorded offset {start}); its delta "
                "cannot be trusted and recovery aborts (fail closed)"
            )
        if size == start:
            continue
        with open(entry, "rb") as fh:
            aligned = _align_start_to_record_boundary(fh, start, entry.name)
            if aligned != start:
                logger.warning(
                    "durable_buffer.fsck: op log %s snapshot offset %d falls inside a record; "
                    "replaying from the previous boundary %d instead (the straddling record is "
                    "treated as post-snapshot — over-fencing, safe)",
                    entry.name,
                    start,
                    aligned,
                )
            fh.seek(aligned)
            data = fh.read()
        # Truncate the torn last record (a crashed writer) at the BYTE level, then
        # decode STRICTLY: everything before the final b'\n' was written by a
        # flushed record and must be valid UTF-8 — errors="replace" would silently
        # rewrite invalid bytes into names that never existed (fail open).
        data = data[: data.rfind(b"\n") + 1]  # rfind==-1 -> empty (whole tail torn)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnrecoverableJournalError(
                f"op log {entry.name} contains invalid UTF-8 past the snapshot offset "
                f"({exc}); the log cannot be trusted and recovery aborts (fail closed)"
            ) from exc
        # Literal-'\n' split ONLY: names may legally contain U+0085/U+2028/U+2029/\v/\f,
        # which str.splitlines() would split on — silently DROPPING a 'P' record for
        # such a name, i.e. a published-but-unfenced object (the unsafe direction).
        for line in text.split("\n"):
            if not line:
                continue  # the final-'\n' split artifact; records are never empty
            if len(line) < 3 or line[0] not in "PD" or line[1] != " ":
                raise UnrecoverableJournalError(
                    f"malformed op-log record {line[:120]!r} in {entry.name}; an unparseable "
                    "log cannot fence anything and recovery aborts (fail closed)"
                )
            op, name = line[0], line[2:]
            _validate_oplog_record_name(name, entry.name)
            if op == "P":
                written.add(name)
                deleted.discard(name)
            elif op == "D":
                deleted.add(name)
                written.discard(name)
    missing_logs = set(offsets) - seen_files
    if missing_logs:
        raise UnrecoverableJournalError(
            f"op logs {sorted(missing_logs)} recorded by the snapshot have disappeared; "
            "their deltas cannot be reconstructed and recovery aborts (fail closed)"
        )
    return written, deleted


def _fsck_oplog(
    tq,
    store_root: Path,
    journal_dir: Path,
    j: Journal,
    dry_run: bool,
    tail: tuple[set[str], set[str]] | None = None,
) -> dict:
    """Delta reconciliation: op-log replay for write identity, one NAMES-ONLY listing
    for membership. Cost is O(churn) records + O(store) directory names — but ZERO
    per-object ``stat`` calls, which is what makes full scans expensive on FSx
    (~0.3 ms per ``stat``; a readdir is batched).

    ``tail`` is the pre-read op-log delta (:func:`recover` reads it BEFORE
    ``tq.load_checkpoint`` so an untrusted artifact aborts before any mutation);
    reading it here covers direct :func:`fsck` callers. Untrustworthy artifacts
    raise :class:`UnrecoverableJournalError` — there is no fallback fence source.

    The names listing exists because TQ mutates STORAGE and CONTROLLER in two steps
    (put: storage publish, then metadata; clear: storage unlink, then metadata) and a
    snapshot can land in between, on the *storage-already-mutated* side — whose op-log
    record then falls BEFORE the captured offsets and is invisible to the tail replay:

    * put-straddle — the object exists but the snapshot never learned its field
      metadata: **unexpected object -> FENCE** (regenerated; wasteful, never wrong);
    * clear-straddle — the object is gone but the snapshot still carries its
      metadata: **metadata-without-object -> DROP** (otherwise the restored controller
      re-serves a key whose read can only fail), and with it the rest of that
      trajectory group (:func:`_lost_group_siblings`), because a clear batch unlinks
      index-major and a snapshot can land part-way through one.
    """
    written_after, deleted_after = tail if tail is not None else _read_oplog_tail(store_root, journal_dir)

    # Names-only membership listing (no stat): what is actually on disk right now.
    # Only REGULAR non-symlink entries count as objects: a symlink wearing a
    # canonical name is not writer-emitted data — counting it as present would make
    # the restored controller serve bytes from OUTSIDE the store through it. Such
    # entries are FENCED below (unlink removes the link itself, never the target).
    t0 = time.perf_counter()
    on_disk: dict[str, tuple[int, str]] = {}
    non_regular: set[str] = set()
    with os.scandir(store_root) as entries:
        for entry in entries:
            parsed = _parse_object_name(entry.name)
            if parsed is None:
                continue
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                non_regular.add(entry.name)
                continue
            on_disk[entry.name] = parsed
    t_list = time.perf_counter() - t0

    t0 = time.perf_counter()
    fenced = set(non_regular)
    for name in written_after:
        # Written after the snapshot: either an unknown sample, or — the dangerous case —
        # a REUSED index whose object now holds another sample's bytes. Intent logging
        # means this can also be a record whose publish never happened; unlink with
        # missing_ok and let DROP clean the metadata either way.
        fenced.add(name)
    for name in deleted_after:
        # Intent-logged delete whose unlink may not have executed (crash in between):
        # consumption of that sample already began, so finishing the delete (and
        # dropping the metadata below) avoids duplicate training. At worst it discards
        # an untrained sample, which verl regenerates.
        if name in on_disk:
            fenced.add(name)
    for name, (idx, fname) in on_disk.items():
        # Membership fence (the put-straddle direction): an object the snapshot does
        # not expect, whose 'P' record predates the captured offsets.
        if idx not in j.live or (idx, fname) not in j.expected:
            fenced.add(name)
    if not dry_run:
        for name in fenced:
            try:
                (store_root / name).unlink(missing_ok=True)  # on a symlink: removes the link only
            except OSError as exc:  # e.g. a canonical-named DIRECTORY (never writer-emitted)
                raise UnrecoverableJournalError(
                    f"store entry {name!r} must be fenced but cannot be unlinked ({exc}); "
                    "the store carries a non-writer-emitted node under an object name — "
                    "remove it manually and re-run recovery"
                ) from exc
    t_fence = time.perf_counter() - t0

    # Missing = expected by the snapshot but not on disk (or fenced above). Ground
    # truth comes from the listing, which also covers the clear-straddle direction
    # (D records that predate the captured offsets) and out-of-band deletions.
    present = {parsed for name, parsed in on_disk.items() if name not in fenced}
    missing = j.expected - present

    t0 = time.perf_counter()
    dropped_keys, prompts_dropped, dropped_by_partition, closure_pairs = _drop_missing(tq, j, missing, dry_run)
    closure_names = _remove_closure_objects(store_root, closure_pairs, dry_run)
    t_drop = time.perf_counter() - t0

    return {
        "mode": FENCE_MODE_OPLOG,
        "oplog_records_replayed": len(written_after) + len(deleted_after),
        "objects_listed": len(on_disk),  # names-only readdir — no per-object stat
        "objects_statted": 0,
        "expected_objects": len(j.expected),
        "fenced_objects": len(fenced),
        "missing_objects": len(missing),
        "group_closure_objects": len(closure_names),  # present objects of group-atomically dropped siblings
        "dropped_samples": len(dropped_keys),
        "dropped_by_partition": dropped_by_partition,
        "prompts_dropped": len(prompts_dropped),
        "t_list_s": round(t_list, 4),
        "t_fence_s": round(t_fence, 4),
        "t_drop_s": round(t_drop, 4),
        "t_total_s": round(t_list + t_fence + t_drop, 4),
        "fence_ts": j.fence_ts,
    }


def _is_prompt_key(key: str, live: dict) -> bool:
    """Does ``key`` name a PROMPT record rather than a trajectory?

    The live view's ``is_prompt`` tag decides when the key is there. For a key the view
    does not carry, the name grammar decides: a prompt uid that happens to look like
    ``{uid}_{session}_{index}`` is then read as a trajectory, which errs towards more
    dropping (safe: verl re-issues) rather than towards a half-group survivor.
    """
    tag = live.get(key)
    if tag is not None:
        return bool(tag.get("is_prompt"))
    return _parse_traj_key(key) is None


def _lost_group_siblings(lost_keys: set, live: dict) -> set:
    """Surviving trajectories of every group that lost at least one member.

    GROUP ATOMICITY. A prompt's trajectories are one training unit: verl samples a
    group, and a group that can never be completed is dead weight the restored
    controller would keep forever. The missing-object rule drops trajectories one key
    at a time, so on its own it can leave a *finished* group holding 1 of ``n``
    trajectories — see the third straddle state documented above :class:`Journal` (the
    whole ``D`` batch lands behind the fence, the already-unlinked prefix DROPs and the
    not-yet-unlinked suffix is present-and-expected, so it is KEPT). The prompt stays
    ``finished``, so neither ``_reissue_inflight_prompts`` nor
    :func:`promote_complete_groups` (which only re-tags ``pending``/``running``
    prompts) will ever touch it, and the missing trajectory cannot be produced again.

    So whichever way a group loses a member, the rest of that group goes too, and the
    now-groupless prompt follows through the orphan rule in :func:`_drop_missing`. The
    decision reads only the missing set and the key grammar; it deliberately does NOT
    read ``production_status``, whose zeroed row (TQ#141 ``mark_clearing``) is
    indistinguishable from a registered-but-unwritten index and would drop in-flight
    puts. For a ``pending``/``running`` group this matches what verl itself does on
    resume: ``_reissue_inflight_prompts`` clears the prompt's old trajectory keys
    before re-dispatching it (``trainer_base.py:867-873``).
    """
    lost_groups = set()
    for key in lost_keys:
        if _is_prompt_key(key, live):
            continue
        parsed = _parse_traj_key(key)
        if parsed is not None:
            lost_groups.add(parsed[0])
    if not lost_groups:
        return set()
    siblings = set()
    for key, tag in live.items():
        if key in lost_keys or tag.get("is_prompt"):
            continue
        parsed = _parse_traj_key(key)
        if parsed is not None and parsed[0] in lost_groups:
            siblings.add(key)
    return siblings


def _remove_closure_objects(store_root: Path, closure_pairs: set, dry_run: bool) -> list[str]:
    """Unlink the objects of samples dropped for group atomicity. Returns the names.

    Those objects were present AND expected, so the fence rule left them alone, and the
    metadata that pointed at them is gone by the time this runs. Left behind they would
    sit in the store until some later restore fenced them as unexpected, so they are
    removed here. Scope: this covers exactly the sibling samples the group-atomicity
    rule drops. An index that is itself partially missing keeps its surviving field
    objects (its metadata drops, and the orphans wait for a later restore's fence);
    sweeping those too would need a per-field closure this rule does not compute.
    ``missing_ok`` because a concurrent
    op-log replay may have unlinked the same name already.
    """
    names = sorted(f"{_enc_key(f'{idx}@{fname}')}{OBJECT_SUFFIX}" for idx, fname in closure_pairs)
    if dry_run:
        return names
    for name in names:
        try:
            (store_root / name).unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - same non-writer-emitted-node case as the fence step
            raise UnrecoverableJournalError(
                f"store entry {name!r} belongs to a group-atomically dropped sample but cannot be "
                f"unlinked ({exc}); remove it manually and re-run recovery"
            ) from exc
    return names


def _drop_missing(tq, j: Journal, missing: set, dry_run: bool) -> tuple[list, list, dict, set]:
    """Rule DROP: samples whose objects are gone, their group siblings, then the
    now-groupless prompts.

    Store-wide by construction: ``missing`` holds (index, field) pairs from EVERY
    partition in the snapshot (one durable store backs them all), so each partition's
    own revert map decides which metadata keys to clear. Restoring ``train`` while a
    ``val`` sample's objects were consumed must drop the val metadata too — otherwise
    the restored controller serves keys whose reads can only fail.

    The drop set is closed over trajectory GROUPS (:func:`_lost_group_siblings`): a
    group that lost one member loses all of them, so no half-group can survive a
    restore in a state nothing will ever complete. ``dropped_by_partition`` therefore
    counts the siblings too.

    Fourth return value: the ``(index, field)`` pairs of the SIBLINGS, whose objects are
    still on disk (they were present and expected, so the fence rule had no reason to
    touch them). The caller removes them; doing it after the metadata clear is the
    fail-safe order, since a crash in between leaves objects no metadata points at,
    which the next restore fences as unexpected.
    """
    missing_idx = {idx for idx, _ in missing}
    all_dropped: list[str] = []
    all_prompts_dropped: list[str] = []
    by_partition: dict[str, int] = {}
    closure_pairs: set[tuple[int, str]] = set()
    # One PRE-clear listing, shared by every partition: the sibling closure needs to see
    # the group members that are still live, and the ``is_prompt`` tags that say which
    # keys are trajectories at all. (The orphan check below still uses a POST-clear view.)
    live_before = tq.kv_list() if missing_idx else {}
    for pname, view in j.per_partition.items():
        lost_keys = {view.revert[idx] for idx in missing_idx if idx in view.revert}
        before = live_before.get(pname, {})
        siblings = _lost_group_siblings(lost_keys, before)
        if siblings:
            # Only pay the O(expected) scan when a group actually lost a member: on the
            # normal path (nothing missing, or whole groups consumed) this stays empty.
            sibling_idx = {view.keys_mapping[k] for k in siblings if k in view.keys_mapping}
            closure_pairs |= {(idx, fname) for idx, fname in j.expected if idx in sibling_idx}
        dropped_keys = sorted(lost_keys | siblings)
        if dropped_keys:
            by_partition[pname] = len(dropped_keys)
        all_dropped.extend(dropped_keys)
        if not dropped_keys:
            continue
        if dry_run:
            # Simulate the post-drop view so the DRY plan reports the SAME orphan
            # prompts a real run would drop (an early return here under-reported them).
            gone = set(dropped_keys)
            live_now = {k: tag for k, tag in before.items() if k not in gone}
        else:
            for off in range(0, len(dropped_keys), 512):
                tq.kv_clear(keys=dropped_keys[off : off + 512], partition_id=pname)
            live_now = tq.kv_list().get(pname, {})  # after the clears: the orphan check needs the post-drop view
        # O(N): index surviving trajectories by group uid instead of scanning all keys
        # per prompt (that is quadratic and dominates fsck at buffer scale).
        traj_groups = set()
        for k, tag in live_now.items():
            if not tag.get("is_prompt"):
                parsed = _parse_traj_key(k)
                if parsed is not None:
                    traj_groups.add(parsed[0])
        journal_traj_groups = {p[0] for p in map(_parse_traj_key, view.keys_mapping) if p is not None}
        prompts_dropped = [
            k
            for k, tag in live_now.items()
            if tag.get("is_prompt")
            and tag.get("status") in ("finished", "failure")
            and k not in traj_groups
            and k in journal_traj_groups
        ]
        if not dry_run:
            for off in range(0, len(prompts_dropped), 512):
                tq.kv_clear(keys=prompts_dropped[off : off + 512], partition_id=pname)
        all_prompts_dropped.extend(prompts_dropped)
    return all_dropped, all_prompts_dropped, by_partition, closure_pairs


def fsck(
    tq,
    store_root: str | Path,
    journal_dir: str | Path,
    partition: str = "train",
    *,
    mode: str = FENCE_MODE_OPLOG,
    mtime_margin: float = 1.0,
    dry_run: bool = False,
    force_store_mismatch: bool = False,
    _precomputed_tail: tuple[set[str], set[str]] | None = None,
    _precomputed_journal: Journal | None = None,
) -> dict:
    """Reconcile store and restored metadata. Call AFTER ``tq.load_checkpoint``.

    Raises :class:`StoreMismatchError` unless the journal and ``store_root`` carry
    the same store identity, and :class:`UnrecoverableJournalError` when the fence
    artifacts cannot prove write identity (see the class docstrings; both are
    fail-closed with NO automatic degradation to a weaker fence source). Prefer
    :func:`recover`, which runs both preflights BEFORE ``tq.load_checkpoint``; a
    direct caller that sees either error must abort the resume — the loaded
    controller metadata cannot be trusted against this store.

    ``mode``: ``"oplog"`` (default; the only recovery-grade fence),
    ``"mtime"`` (EXPLICIT legacy/audit flow for stores with no usable fence
    artifacts: trusts client clocks and conservatively over-fences by
    ``mtime_margin``; never auto-selected), or ``"index"`` (A/B experiments only —
    unsafe under index reuse).

    ``_precomputed_journal`` lets :func:`recover` hand over the ``Journal`` it already
    parsed in its preflight, so the controller pickle is read (and judged) exactly
    once, before ``tq.load_checkpoint``.
    """
    store_root = Path(store_root)
    journal_dir = Path(journal_dir)
    check_journal_recoverable(journal_dir)  # the fence stamp is judged FIRST: a malformed
    # fence.json is an untrusted artifact (UnrecoverableJournalError), not an identity question
    check_store_identity(store_root, journal_dir, force_store_mismatch)
    j = load_journal(journal_dir, partition) if _precomputed_journal is None else _precomputed_journal

    if mode == FENCE_MODE_OPLOG:
        report = _fsck_oplog(tq, store_root, journal_dir, j, dry_run, tail=_precomputed_tail)
        logger.info("durable_buffer.fsck: %s", report)
        return report
    if mode not in (FENCE_MODE_MTIME, FENCE_MODE_INDEX):
        raise ValueError(f"unknown fence mode {mode!r}; valid: oplog (default), mtime (legacy), index (A/B)")
    if mode == FENCE_MODE_MTIME:
        if j.fence_ts is None:
            raise UnrecoverableJournalError(
                f"mtime fencing needs the snapshot's fence_ts and {journal_dir} carries none; "
                "recovery aborts (fail closed)"
            )
        logger.warning(
            "durable_buffer.fsck: EXPLICIT mtime fencing requested — a legacy/audit flow that trusts "
            "client clocks within %.1fs and conservatively over-fences; the op-log mode is the only "
            "recovery-grade fence source",
            mtime_margin,
        )

    t0 = time.perf_counter()
    listing = []
    non_regular: list[str] = []  # canonical-named symlinks etc.: fenced, never statted through
    with os.scandir(store_root) as entries:
        for entry in entries:
            parsed = _parse_object_name(entry.name)
            if parsed is None:
                continue
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                non_regular.append(entry.name)
                continue
            st = entry.stat(follow_symlinks=False)
            listing.append((Path(entry.path), parsed[0], parsed[1], st.st_mtime, st.st_ino, st.st_size))
    t_list = time.perf_counter() - t0

    t0 = time.perf_counter()
    fenced: list[str] = list(non_regular)
    if not dry_run:
        for name in non_regular:
            (store_root / name).unlink(missing_ok=True)  # removes the link itself, never the target
    present: set[tuple[int, str]] = set()
    for path, idx, fname, mtime, _inode, _size in listing:
        unexpected = (idx not in j.live) or ((idx, fname) not in j.expected)
        rewritten = False
        if mode == FENCE_MODE_MTIME:
            rewritten = mtime >= (j.fence_ts - mtime_margin)
        if unexpected or rewritten:
            fenced.append(path.name)
            if not dry_run:
                path.unlink(missing_ok=True)
        else:
            present.add((idx, fname))
    t_fence = time.perf_counter() - t0

    t0 = time.perf_counter()
    missing = j.expected - present
    dropped_keys, prompts_dropped, dropped_by_partition, closure_pairs = _drop_missing(tq, j, missing, dry_run)
    closure_names = _remove_closure_objects(store_root, closure_pairs, dry_run)
    t_drop = time.perf_counter() - t0

    report = {
        "mode": mode,
        "objects_listed": len(listing),
        "objects_statted": len(listing),  # full modes stat every object (identity/mtime)
        "expected_objects": len(j.expected),
        "fenced_objects": len(fenced),
        "missing_objects": len(missing),
        "group_closure_objects": len(closure_names),  # present objects of group-atomically dropped siblings
        "dropped_samples": len(dropped_keys),
        "dropped_by_partition": dropped_by_partition,
        "prompts_dropped": len(prompts_dropped),
        "t_list_s": round(t_list, 4),
        "t_fence_s": round(t_fence, 4),
        "t_drop_s": round(t_drop, 4),
        "t_total_s": round(t_list + t_fence + t_drop, 4),
        "fence_ts": j.fence_ts,
    }
    logger.info("durable_buffer.fsck: %s", report)
    return report


def _prompt_rollout_n(tq, partition: str, uid: str) -> tuple[int | None, bool]:
    """Per-prompt expected session count from the prompt's own TQ record.

    verl's dispatch (``_submit_batch_to_rollout``) persists the prompt payload's
    tensor / per-sample NonTensor columns into TQ, so a per-sample ``__rollout_n__``
    override (see ``agent_loop_tq._run_prompt``) appears as a field on the prompt's
    record. Returns ``(n, True)`` when the record carries a parseable override,
    ``(None, True)`` when the record verifiably has NO such field, and
    ``(None, False)`` when the record cannot be inspected or the value is not
    EXACTLY one positive integer scalar — promotion must then be disabled for that
    prompt (fail closed). Strictness matters: the count sizes the completeness
    check, so a boolean (``True`` -> 1), a fractional value (``2.5`` -> 2) or a
    multi-element field silently coerced to an int would promote groups at a size
    the record never durably stated.
    """
    try:
        meta = tq.get_client().kv_retrieve_meta(keys=[uid], partition_id=partition, create=False)
        fields = set(getattr(meta, "field_names", None) or [])
    except Exception:  # noqa: BLE001 - any failure to inspect disables promotion for the prompt
        return None, False
    if "__rollout_n__" not in fields:
        return None, True
    try:
        data = tq.kv_batch_get(keys=[uid], partition_id=partition, select_fields=["__rollout_n__"])
        value = data["__rollout_n__"]
        if getattr(value, "is_nested", False):
            value = value.values()  # jagged (1, j1) -> flat; reshape() rejects nested shapes
        elements = list(value.reshape(-1)) if hasattr(value, "reshape") else list(value)
        if len(elements) != 1:
            return None, False  # one prompt, one count: anything else is unverifiable
        scalar = elements[0]
        if hasattr(scalar, "item"):
            scalar = scalar.item()  # torch/numpy 0-d -> python scalar (preserves bool/float)
        if isinstance(scalar, bool):  # bool IS an int subclass: reject explicitly
            return None, False
        if isinstance(scalar, float):
            if not scalar.is_integer():
                return None, False  # 2.5 sessions is not a count
            scalar = int(scalar)
        if not isinstance(scalar, int):
            return None, False  # strings & friends: never coerce
        n = scalar
    except Exception:  # noqa: BLE001
        return None, False
    return (n, True) if n > 0 else (None, False)


def promote_complete_groups(
    tq,
    partition: str,
    group_size: int,
    dry_run: bool = False,
    *,
    assume_uniform_group_size: bool = False,
) -> dict:
    """Re-tag ``running``/``pending`` prompts whose group is fully present and successful.

    "Fully present" is judged per SESSION, not per key: trajectory keys are
    ``{uid}_{session}_{index}`` and one session can emit several outputs, so counting
    keys would let 4 outputs of session 0 impersonate a complete ``group_size=4``
    group. A group promotes only when every session ``0..n-1`` has at least one
    trajectory and every trajectory in the group is ``success``.

    The expected session count ``n`` must be DURABLY KNOWN, per prompt — promotion
    fails closed otherwise:

    * A per-sample ``__rollout_n__`` persisted in the prompt's own TQ record (verl's
      async dispatch persists the payload's tensor / ``NonTensorStack`` columns,
      ``trainer_base._submit_batch_to_rollout``) is authoritative.
    * When the record carries NO such field — which is the NORMAL case: a batch-level
      ``NonTensorData`` override is consumed by the agent loop
      (``agent_loop_tq._run_prompt``) but never persisted, and sync mode persists no
      payload at all — the true count is NOT durably knowable. The caller's
      ``group_size`` (rollout.n) is then used ONLY under
      ``assume_uniform_group_size=True``, an explicit operator assertion that the run
      used a uniform ``n`` (the restore tool wires its ``--group-size`` flag to this
      assertion). Without it the prompt is left for verl's re-issue
      (``skipped_no_durable_n``) — otherwise a hidden override (e.g. ``n=8``) with
      sessions ``0..3`` present would promote a half group as "complete".
    * Even under the assertion, observed evidence wins: a prompt whose record cannot
      be inspected, or whose observed sessions reach past the assumed ``n`` (proof
      the assumption is wrong), is skipped (``skipped_unverifiable_n``).

    Conservative on purpose: a group containing any ``failure`` session is left to
    verl's failure-refill path, and a partially produced group is left to
    ``_reissue_inflight_prompts`` exactly as today.
    """
    live = tq.kv_list().get(partition, {})
    by_group: dict[str, list[tuple[int, str]]] = {}  # uid -> [(session, status)]
    for key, tag in live.items():
        if tag.get("is_prompt"):
            continue
        parsed = _parse_traj_key(key)
        if parsed is not None:
            by_group.setdefault(parsed[0], []).append((parsed[1], tag.get("status")))

    promoted, partial, with_failures, skipped, skipped_no_n = [], [], [], [], []
    sessions_rescued = 0
    for uid, tag in list(live.items()):
        if not tag.get("is_prompt") or tag.get("status") not in ("pending", "running"):
            continue
        members = by_group.get(uid, [])
        if not members:
            continue  # nothing produced yet: plain re-issue territory, as today
        statuses = [status for _, status in members]
        if any(s == "failure" for s in statuses):
            with_failures.append(uid)
            continue
        per_prompt_n, determinable = _prompt_rollout_n(tq, partition, uid)
        if not determinable:
            skipped.append(uid)  # fail closed: the record cannot be inspected
            continue
        if per_prompt_n is not None:
            expected_n = per_prompt_n
        elif assume_uniform_group_size:
            expected_n = group_size  # operator-asserted uniform rollout.n
        else:
            skipped_no_n.append(uid)  # fail closed: no durable count, no assertion
            continue
        if max(session for session, _ in members) >= expected_n:
            skipped.append(uid)  # fail closed: the expected session count is provably wrong
            continue
        success_sessions = {session for session, status in members if status == "success"}
        if all(s == "success" for s in statuses) and success_sessions >= set(range(expected_n)):
            promoted.append(uid)
            sessions_rescued += expected_n
            if not dry_run:
                new_tag = dict(tag)
                new_tag["status"] = "finished"
                tq.kv_put(key=uid, partition_id=partition, tag=new_tag)
        else:
            partial.append(uid)

    report = {
        "promoted": sorted(promoted),
        "partial_left_for_reissue": sorted(partial),
        "had_failures_left_for_reissue": sorted(with_failures),
        "skipped_unverifiable_n": sorted(skipped),
        "skipped_no_durable_n": sorted(skipped_no_n),
        "trajectories_rescued": sessions_rescued,
    }
    if promoted or skipped or skipped_no_n:
        logger.info("durable_buffer.promote_complete_groups: %s", report)
    return report


def recover(
    tq,
    store_root: str | Path,
    journal_dir: str | Path,
    partition: str = "train",
    *,
    group_size: int | None = None,
    mode: str = FENCE_MODE_OPLOG,
    dry_run: bool = False,
    force_store_mismatch: bool = False,
    assume_uniform_group_size: bool = False,
    _precomputed_tail: tuple[set[str], set[str]] | None = None,
) -> dict:
    """Full restore path: fail-closed preflights -> ``load_checkpoint`` -> ``fsck`` ->
    optional group rescue.

    Every fail-closed check runs BEFORE ``tq.load_checkpoint``, because loading
    already mutates the caller's controller — a rejected journal must leave it
    untouched, holding no metadata that could serve unfenced bytes:

    * store identity (:func:`check_store_identity`, :class:`StoreMismatchError`);
    * refusal of ``oplog: false`` snapshots, of a slot that is not the product of one
      journal tick, and of a controller-snapshot schema this code does not read
      (:func:`check_journal_recoverable`);
    * the controller pickle itself (:func:`load_journal`): a pure file read, so a
      corrupt, truncated or version-drifted pickle is classified as an unrecoverable
      journal here rather than crashing after the controller was already loaded. The
      parsed view is handed to :func:`fsck`, which therefore reads it once;
    * in the default op-log mode, the full offsets validation + tail replay
      (:class:`UnrecoverableJournalError` on any untrusted artifact) — plain file
      reads, no TQ needed.

    ``group_size`` enables the group-completeness rescue;
    ``assume_uniform_group_size=True`` is the operator's assertion that the run used
    a uniform rollout.n, without which prompts lacking a persisted per-prompt count
    are never promoted (see :func:`promote_complete_groups`).

    ``dry_run=True`` reports what fencing/dropping/promotion WOULD do without touching
    the store or the controller metadata (``load_checkpoint`` still runs, but only on
    the caller's TQ instance).

    ``_precomputed_tail`` lets a caller that already ran the fail-closed tail read
    (the restore tool's preflight) hand over its VALIDATED delta instead of reading
    it twice.
    """
    check_journal_recoverable(journal_dir)  # the fence stamp is judged FIRST: a malformed
    # fence.json is an untrusted artifact (UnrecoverableJournalError), not an identity question
    check_store_identity(store_root, journal_dir, force_store_mismatch)
    # The controller pickle is parsed HERE, in the preflight, for the same reason the
    # three checks above are: it is a plain file read that needs no TQ instance, and a
    # pickle that cannot be parsed is an untrusted artifact. Left inside fsck() it was
    # the one rejection class still evaluated AFTER tq.load_checkpoint had loaded the
    # snapshot into the caller's live controller, which then held metadata pointing at
    # unfenced bytes while the failure surfaced as a raw traceback.
    journal = load_journal(journal_dir, partition)
    tail = _precomputed_tail
    if mode == FENCE_MODE_OPLOG and tail is None:
        tail = _read_oplog_tail(Path(store_root), Path(journal_dir))  # raises before any mutation
    t0 = time.perf_counter()
    tq.load_checkpoint(str(journal_dir))
    t_load = time.perf_counter() - t0
    report = fsck(
        tq,
        store_root,
        journal_dir,
        partition,
        mode=mode,
        dry_run=dry_run,
        force_store_mismatch=force_store_mismatch,
        _precomputed_tail=tail,
        _precomputed_journal=journal,
    )
    report["t_load_checkpoint_s"] = round(t_load, 4)
    if group_size:
        report["promotion"] = promote_complete_groups(
            tq, partition, group_size, dry_run=dry_run, assume_uniform_group_size=assume_uniform_group_size
        )
    return report
