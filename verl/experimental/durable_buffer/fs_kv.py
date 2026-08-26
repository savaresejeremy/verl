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
"""Durable POSIX/FSx storage backend for TransferQueue.

Registers three TQ extension points (no TQ core changes):

* ``StorageBootstrapProvider("DurableFS")``  — provisions/attaches the store directory.
* ``StorageClientFactory("DurableFSClient")`` — per-process KV client.
* ``StorageManagerFactory("DurableFS")``      — ``KVStorageManager`` subclass.

Object layout
-------------
``{root}/{global_index}@{field_name}`` — one object per (sample, field), which is the
same granularity TQ's KV key model uses. Field-per-object (rather than packing a whole
trajectory into one object) is deliberate: verl reads fields selectively (DAPO group
filtering and staleness paths fetch ``rewards`` alone), and packing measured 251x read
amplification on those paths for a 2.6x write win.

Correctness notes that cost real bugs to learn
----------------------------------------------
1. ``clear_data`` must NOT trust ``metadata.field_names``. TQ's controller computes a
   batch's field set as the *intersection* over its samples, so verl's
   ``_clear_groups`` (prompt key + trajectory keys in ONE ``kv_clear``) arrives with an
   EMPTY field set even though the trajectories have stored bytes. Returning early
   there silently leaks every consumed group's objects, and after a journal restore
   those samples become sampleable again (duplicate training). We therefore delete
   everything belonging to the cleared *indexes* — exact, because ``clear_samples``
   removes whole samples.
2. Deleting by index must not scan the store per index: a per-index ``glob`` is
   O(store) per sample and measured 81s to clear 2.5k keys in a 50k-object store.
   We delete ``{index} x {known field names}`` with delete-if-exists, keeping the field
   names in a tiny shared registry file so consumer-only processes know them too.
3. ``save_checkpoint`` writes *fence artifacts* (per-writer op-log offsets +
   ``fence.json``) instead of copying data. That rides verl's unmodified
   ``tq.save_checkpoint(...)`` call, and it is what lets ``fsck`` detect objects
   written or rewritten after a snapshot (TQ recycles freed sample indexes, so a
   stale index can otherwise serve another sample's bytes). The op log is mandatory:
   the removed ``oplog: false`` mode wrote a full manifest AFTER the controller
   pickle, which recorded gap overwrites as valid — ``write_manifest(full=True)``
   survives only as an offline audit tool recovery never consumes.
4. Store-internal paths are validated lstat-style at every read/write/list point:
   ``_oplog``/``_fields.d`` must be real non-symlink directories resolving under the
   store root, and op logs / field markers / canonical objects / the ``_store_id``
   identity stamp must be regular non-symlink files — anything else is refused
   (:class:`StoreIntegrityError`)
   instead of followed, because a link there redirects writes (op log, registry) or
   reads (``get`` deserializing foreign bytes under a trusted key, ``_store_id``
   letting an external file act as the store's lineage authority) outside the
   store. LIMITATION: the checks are check-then-use, not dirfd/``O_NOFOLLOW``
   plumbing, so a swap between the ``lstat`` and the operation is not closed — the
   store lives on trusted FSx/NFS paths and the threat model is misconfiguration,
   not an active local attacker racing the client.
5. The intent log is FLUSHED, not fsynced, before each publish. That ordering holds
   across a killed process and not across a lost node, where a durable rename can
   outlive a log record still sitting in a client cache. ``DurableFS.oplog_fsync:
   true`` adds one ``os.fsync`` per mutation batch and extends the ordering to a node
   crash; it is opt-in because the barrier costs shared-filesystem producer
   throughput (see :meth:`DurableFSClient._log_ops`).
"""

from __future__ import annotations

import inspect
import itertools
import json
import logging
import os
import stat
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
from transfer_queue.storage.clients.base import StorageClientFactory, StorageKVClient
from transfer_queue.storage.managers.base import KVStorageManager, StorageManagerFactory
from transfer_queue.utils import serial_utils as _su

logger = logging.getLogger(__name__)

BACKEND_NAME = "DurableFS"
CLIENT_NAME = "DurableFSClient"
MANIFEST_FILE = "store_manifest.tsv"
OPLOG_OFFSETS_FILE = "oplog_offsets.json"
FENCE_FILE = "fence.json"
FIELDS_REGISTRY = "_fields.d"
OPLOG_DIR = "_oplog"
OBJECT_SUFFIX = ".bin"
STORE_ID_FILE = "_store_id"


class StoreIntegrityError(RuntimeError):
    """A store-internal path is not the filesystem object this backend creates.

    The backend only ever creates real directories (``_oplog``, ``_fields.d``) and
    regular files (op logs, field markers, canonical objects, the ``_store_id``
    identity stamp) directly under the
    store root. A symlink — or any other node kind — at one of those names means the
    store was tampered with or misassembled (a copied tree, a stray link), and
    following it would redirect reads or writes OUTSIDE the store: a symlinked
    ``_oplog`` sends intent-log appends elsewhere (poisoning the only recovery fence
    source), a symlinked object makes ``get`` deserialize foreign bytes under a
    trusted key, and a symlinked ``_store_id`` lets an EXTERNAL file act as the
    store's lineage authority in front of destructive reconciliation.
    Every read/write/list point therefore validates lstat-style and
    raises this error instead of following (fsck FENCES such entries instead).

    RESIDUAL TOCTOU (documented limitation): validation is check-then-use, not
    dirfd/``O_NOFOLLOW``-plumbed, so a node could in principle be swapped between
    the ``lstat`` and the operation. The store lives on trusted FSx/NFS paths and
    the threat model is operator misconfiguration, not an active local attacker
    racing the client — the check closes the misconfiguration hole and makes the
    corruption loud.
    """


def _reject_non_regular(path: Path, what: str) -> None:
    """FAIL CLOSED unless ``path`` is absent or a regular, non-symlink file.

    Absence is fine — each caller has its own missing-file behavior (publish
    creates, reads raise ``FileNotFoundError``). Anything present that is not a
    plain regular file is refused instead of followed.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise StoreIntegrityError(
            f"{what} {path} exists but is not a regular non-symlink file "
            f"(mode {stat.filemode(st.st_mode)}); a link here would redirect the backend's "
            "reads/writes outside the store, so it is refused (fail closed)"
        )


def _require_internal_dir(path: Path, root: Path, what: str) -> None:
    """FAIL CLOSED: ``path`` must be a real (non-symlink) directory that stays under
    the resolved store root."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise StoreIntegrityError(f"{what} {path} does not exist") from None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise StoreIntegrityError(
            f"{what} {path} is not a real (non-symlink) directory "
            f"(mode {stat.filemode(st.st_mode)}); a link here would redirect every op-log/"
            "registry write outside the store, so it is refused (fail closed)"
        )
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise StoreIntegrityError(f"{what} {path} resolves to {resolved}, outside the store root {root} (fail closed)")


# ---------------------------------------------------------------- key encoding
# Keys and field names are user data but become file names and one-line op-log
# records, so a small reserved set must be escaped reversibly:
#   '%'  the escape character itself
#   '/'  path separator (a field named "a/b" must not create a subdirectory,
#        and "../x" must not escape the store root)
#   '\t' the full-manifest column separator
#   '\n' '\r'  op-log records are line-oriented
#   '\0' cannot appear in a POSIX path at all
# plus a *leading* '.' (keeps encoded names out of the hidden/dot namespace and
# makes traversal like ".." impossible even after '/' encoding). '@' stays raw:
# the object-name grammar is "{int_index}@{field}" and the FIRST '@' delimits,
# so '@' inside a field name is unambiguous. This is the on-disk format from the
# start (pre-release) — there is no legacy unencoded data to migrate.
#
# Other Unicode line separators (U+0085 NEL, U+2028 LS, U+2029 PS, \v, \f) are
# LEGAL in names and stay raw — they are ordinary filename bytes. The invariant
# that keeps line-oriented artifacts (op log, manifest) parseable is therefore
# one-sided: writers only ever terminate records with literal '\n' (and '\n'/'\r'
# cannot appear inside an encoded name), and READERS MUST SPLIT ON LITERAL '\n'
# ONLY. `str.splitlines()` also splits on the five separators above and would
# corrupt records for such names (a round-1..2 review finding).
_ENC_TABLE = {c: f"%{ord(c):02X}" for c in "%/\t\n\r\x00"}


def _enc(s: str) -> str:
    out = "".join(_ENC_TABLE.get(c, c) for c in s)
    if out.startswith("."):
        out = "%2E" + out[1:]
    return out


def _dec(s: str) -> str:
    parts = s.split("%")
    out = [parts[0]]
    for p in parts[1:]:
        out.append(chr(int(p[:2], 16)) + p[2:])
    return "".join(out)


def _enc_key(key: str) -> str:
    """Encode a KV key for use as a file name.

    Keys are ``{index}@{field}``; the components are encoded separately so that the
    leading-dot rule applies to the FIELD too (a field named ``.hidden`` must produce
    the same file name whether composed via ``_path`` or via the field registry).
    Decoding is ``_dec`` on the whole stem — '@' is never encoded, so per-component
    encode and whole-stem decode are inverses.
    """
    if "@" in key:
        idx, fname = key.split("@", 1)
        return f"{_enc(idx)}@{_enc(fname)}"
    return _enc(key)


def read_store_id(root: str | Path) -> str | None:
    """The store's identity stamp (or ``None`` for a store created before one).

    lstat-validated, FAIL CLOSED: the stamp is the store's LINEAGE AUTHORITY — the
    bootstrap/attach identity checks and recovery's destructive reconciliation all
    trust it — so a symlinked or otherwise non-regular ``_store_id`` raises
    :class:`StoreIntegrityError` instead of being read through (an external file,
    even one holding a perfectly valid UUID, must never get to act as this store's
    lineage). Absence stays ``None``, exactly as before.
    """
    path = Path(root) / STORE_ID_FILE
    _reject_non_regular(path, "store identity stamp")
    try:
        return path.read_text().strip() or None
    except FileNotFoundError:
        return None


def _ensure_store_id(root: Path) -> str:
    """Create-or-read the identity stamp. O_EXCL makes concurrent creators agree.

    The concurrent-create fallback is lstat-validated too: ``O_CREAT|O_EXCL`` fails
    on ANY pre-existing node — including a symlink, even a dangling one — and the
    fallback read must not then adopt an identity THROUGH that link
    (:class:`StoreIntegrityError` instead, same discipline as :func:`read_store_id`).
    """
    path = root / STORE_ID_FILE
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        _reject_non_regular(path, "store identity stamp")
        return path.read_text().strip()
    with os.fdopen(fd, "w") as f:
        new_id = uuid.uuid4().hex
        f.write(new_id)
    return new_id


# Per-process counters; useful for tests, harnesses and metrics exporters.
STATS: dict[str, int] = {
    "put_objects": 0,
    "put_bytes": 0,
    "get_objects": 0,
    "clear_calls": 0,
    "clear_objects_deleted": 0,
    "clear_without_field_schema": 0,
    "clear_index_scans": 0,
    "manifest_writes": 0,
    "manifest_rows": 0,
    "oplog_records": 0,
    "oplog_fsyncs": 0,
}


def _dumps(value: Any, serializer: str) -> bytes:
    if serializer == "torch":
        import io

        import torch

        bio = io.BytesIO()
        torch.save(value, bio)
        return bio.getvalue()
    frames = _su.encode(value)
    buf = bytearray(_su.calc_packed_size(frames))
    _su.pack_into(buf, frames)
    return bytes(buf)


def _loads(blob: bytes, serializer: str) -> Any:
    if serializer == "torch":
        import io

        import torch

        return torch.load(io.BytesIO(blob), weights_only=False)
    return _su.decode([bytes(f) for f in _su.unpack_from(blob)])


def _reject_oplog_disabled(conf_value) -> None:
    """The ``oplog: false`` tick mode is REMOVED (round-4 review).

    That mode wrote the full ``(inode, size)`` manifest from the storage hook, which
    runs AFTER the controller state was pickled — so a same-name overwrite landing in
    the gap was recorded as valid and a restore could serve the replacement bytes
    under the snapshotted key. There is no safe way to consume such a snapshot, so
    the mode does not exist: the op log is mandatory, and recovery refuses journals
    recorded with ``oplog: false`` (see ``fsck.check_journal_recoverable``). The full
    manifest remains available via ``write_manifest(full=True)`` as an explicitly
    OFFLINE AUDIT artifact that recovery never consumes.
    """
    if not bool(conf_value):
        raise ValueError(
            "DurableFS: 'oplog: false' is not supported. The full-manifest tick mode was "
            "removed because its manifest is written after the controller snapshot, so a "
            "same-name overwrite in the gap is recorded as valid and recovery could serve "
            "replacement bytes under an old key. Keep the op log enabled (the default); "
            "write_manifest(full=True) remains available as an offline audit tool only."
        )


@StorageBootstrapProvider.register_provider(BACKEND_NAME)
def bootstrap_durable_fs(conf) -> dict[str, Any]:
    """Provision (or attach to) the durable store directory.

    Returns the config dict handed to every ``DurableFSClient`` in the job.
    ``create: false`` (used by offline tools such as ``restore_tool``) attaches to an
    EXISTING store without writing anything — no ``mkdir`` and, critically, no
    ``_store_id`` stamping: stamping an unstamped directory would manufacture the
    very identity that the fail-closed store checks exist to demand.
    """
    backend_conf = conf.backend[BACKEND_NAME]
    _reject_oplog_disabled(backend_conf.get("oplog", True))
    root = Path(str(backend_conf.root))
    create = bool(backend_conf.get("create", True))
    if create:
        root.mkdir(parents=True, exist_ok=True)
        _ensure_store_id(root)  # bind journals/fence manifests to THIS store (fsck checks it)
    elif read_store_id(root) is None:
        raise FileNotFoundError(
            f"DurableFS: create=False requires an existing store with an identity stamp; "
            f"{root} has none (wrong --store path, or a store this backend never wrote)"
        )
    return {
        "root": str(root),
        "client_name": backend_conf.get("client_name", CLIENT_NAME),
        "serializer": backend_conf.get("serializer", "tq"),
        "create": create,
        # Opt-in node-crash barrier on the intent log (default OFF; see
        # ``DurableFSClient._log_ops``). Carried in the client config so every writer
        # in the job inherits it from the controller-stored bootstrap result.
        "oplog_fsync": bool(backend_conf.get("oplog_fsync", False)),
    }


def _close_quietly(fh) -> None:
    try:
        fh.close()
    except Exception:  # pragma: no cover
        pass


@StorageClientFactory.register(CLIENT_NAME)
class DurableFSClient(StorageKVClient):
    """File-per-(sample, field) durable KV client with atomic publish."""

    _tmp_counter = itertools.count()

    def __init__(self, config: dict):
        super().__init__(config)
        _reject_oplog_disabled(config.get("oplog", True))
        self.root = Path(str(config["root"]))
        self.serializer = config.get("serializer", "tq")
        # Opt-in: fsync the intent-log batch before the caller publishes the mutations
        # it describes. OFF by default because the ordering it strengthens already
        # holds against a killed process and a barrier costs measurable producer
        # throughput on a shared filesystem; ON when the crash model must include a
        # lost node (see :meth:`_log_ops`).
        self.oplog_fsync = bool(config.get("oplog_fsync", False))
        if bool(config.get("create", True)):
            self.root.mkdir(parents=True, exist_ok=True)
            self.store_id = _ensure_store_id(self.root)
        else:
            # Attach-only (offline tools): NEVER stamp — an unstamped store must stay
            # observably unstamped so the fail-closed identity checks can reject it.
            store_id = read_store_id(self.root)
            if store_id is None:
                raise FileNotFoundError(f"DurableFS: create=False and {self.root} carries no identity stamp")
            self.store_id = store_id
        # Per-writer random token, used in BOTH generated-name families (op-log file
        # and tmp publish names). PID+counter alone is unique per host only: on a
        # shared filesystem two writers on DIFFERENT hosts can carry the same PID and
        # the same counter value, which would interleave two op logs (corrupted
        # records) or share a tmp name (one writer publishing the other's bytes).
        self._writer_token = uuid.uuid4().hex
        self._fields_seen: set[str] = set()  # ENCODED field names (as stored on disk)
        self._fields_mtime: float | None = None
        # Incremental op log: one append-only file per writer process. Each line is
        # "P <object>" or "D <object>" (object = encoded file name). A journal tick only
        # has to record each log's length, and a restore only has to read the tails past
        # those lengths — so neither operation scales with the store. On FSx Lustre,
        # where a single `stat` costs ~0.3 ms, that is the difference between a 17 s
        # journal tick at 50k objects and a millisecond one.
        #
        # The op log is MANDATORY (the removed ``oplog: false`` mode is rejected in
        # ``__init__``): it is the only fence source recovery consumes.
        #
        # Records are INTENT-logged: the record is written and flushed BEFORE the
        # mutation is published (P before os.replace, D before unlink). A crash between
        # the two leaves a logged-but-unpublished record, which fsck resolves as
        # fence+DROP — safe, at most wasteful. The reverse order would leave a
        # published-but-unlogged mutation that op-log fencing could never see, which
        # under TQ's index reuse can expose another sample's bytes through a stale key.
        self._oplog_path: Path | None = None
        self._oplog_fh = None
        self._oplog_finalizer = None

    # ------------------------------------------------------------------ op log
    @property
    def oplog_enabled(self) -> bool:
        """Always ``True``: the op log is this backend's only recovery fence source.

        Kept as an attribute because ``JournalWriter`` checks it before pre-capturing
        offsets — a hypothetical KV client advertising ``oplog_enabled=False`` must
        fail the tick rather than publish a journal recovery cannot trust.
        """
        return True

    def _oplog(self):
        if self._oplog_fh is None:
            d = self.root / OPLOG_DIR
            d.mkdir(parents=True, exist_ok=True)
            # A symlinked _oplog would satisfy mkdir(exist_ok=True) yet send every
            # intent-log append — the only recovery fence source — outside the store.
            _require_internal_dir(d, self.root, "op-log directory")
            self._oplog_path = d / f"{os.getpid()}-{self._writer_token}.log"
            _reject_non_regular(self._oplog_path, "op log")
            self._oplog_fh = open(self._oplog_path, "a", buffering=1024 * 64, encoding="utf-8")
            # Close on GC (and detachably on close()): a leaked handle is a
            # ResourceWarning and, worse, unflushed records on interpreter exit.
            self._oplog_finalizer = weakref.finalize(self, _close_quietly, self._oplog_fh)
        return self._oplog_fh

    def close(self) -> None:
        """Flush and close the op-log handle (idempotent)."""
        if self._oplog_fh is not None:
            _close_quietly(self._oplog_fh)
            self._oplog_fh = None
        if self._oplog_finalizer is not None:
            self._oplog_finalizer.detach()
            self._oplog_finalizer = None

    def _log_ops(self, op: str, names: list[str]) -> None:
        """INTENT log: call before publishing the mutations named in ``names``.

        ``flush()`` moves the batch out of this process, which is what makes the
        record-before-mutation ordering hold across a killed PROCESS: the page cache
        survives it, so the record is visible to a later replay. It does not hold
        across a lost NODE. Buffered file data and a ``rename`` do not share a
        durability class on a shared filesystem — the server commits the namespace
        operation while the log bytes may still be in a client cache — so a power
        loss or kernel panic can leave a published object whose intent record is
        gone, and under TQ's index reuse a restore would then serve another sample's
        bytes under the snapshotted key.

        ``DurableFS.oplog_fsync: true`` closes that by making the batch durable
        before the caller publishes anything. It is one barrier per mutation BATCH
        rather than per object, so a whole ``kv_batch_put`` pays for one, but on a
        shared filesystem a barrier is expensive enough to be expected to move the
        producer rate, which is why it is opt-in rather than the default. That cost
        is not measured in this branch's artifacts: every producer benchmark here
        ran at the default of False.
        """
        fh = self._oplog()
        if fh is None or not names:
            return
        fh.write("".join(f"{op} {n}\n" for n in names))
        fh.flush()  # one flush per batch: every published object is covered by the log
        if self.oplog_fsync:
            os.fsync(fh.fileno())  # node-crash barrier: the record is on stable storage
            STATS["oplog_fsyncs"] += 1
        STATS["oplog_records"] += len(names)

    # ------------------------------------------------------------------ helpers
    def _path(self, key: str) -> Path:
        return self.root / f"{_enc_key(key)}{OBJECT_SUFFIX}"

    def _tmp_path(self, path: Path) -> Path:
        """A tmp name unique per (writer, call): two clients publishing the SAME key
        concurrently must never share a tmp file (one's rename would fail or publish
        the other's bytes). The writer token makes this hold ACROSS hosts too — on a
        shared filesystem PID+counter can collide between machines."""
        return path.parent / f"{path.name}.{os.getpid()}.{self._writer_token}.{next(self._tmp_counter)}.tmp"

    def _known_fields(self) -> list[str]:
        """Union of (encoded) field names ever written to this store.

        Kept as MARKER FILES in ``_fields.d/`` (one empty file per field name) so that
        (a) a process which only ever *consumes* can still delete a sample's objects
        without listing the store, and (b) concurrent writers cannot race: creating a
        marker is idempotent, unlike rewriting a shared registry file. (A shared
        ``_fields.txt`` + tmp-rename lost a race under 8 concurrent rollout workers —
        two writers shared the tmp name — and could also silently drop a field when
        two unions overlapped. Marker files have neither problem.)

        Each child must be a regular non-symlink file — exactly what
        ``_register_fields`` creates. Anything else (a symlink, a directory) proves
        the registry was tampered with or misassembled, and since these names drive
        whole-sample deletes it is refused (:class:`StoreIntegrityError`) rather
        than counted as a field marker: skipping it silently could just as well hide
        a REPLACED legitimate marker, leaking that field's objects past every clear.
        """
        d = self.root / FIELDS_REGISTRY
        try:
            mtime = os.lstat(d).st_mtime  # lstat: a symlinked registry must not be probed through
        except FileNotFoundError:
            return sorted(self._fields_seen)
        _require_internal_dir(d, self.root, "field registry")  # never list a redirected registry
        if mtime != self._fields_mtime:
            names: set[str] = set()
            with os.scandir(d) as entries:
                for entry in entries:
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        raise StoreIntegrityError(
                            f"field marker {entry.path} is not a regular non-symlink file; the field "
                            "registry only ever holds writer-created marker files, and its names drive "
                            "whole-sample deletes, so a tampered registry is refused (fail closed)"
                        )
                    names.add(entry.name)
            self._fields_seen |= names
            self._fields_mtime = mtime
        return sorted(self._fields_seen)

    def _register_fields(self, keys: list[str]) -> None:
        new = {_enc(k.split("@", 1)[1]) for k in keys if "@" in k} - self._fields_seen
        if not new:
            return
        d = self.root / FIELDS_REGISTRY
        d.mkdir(parents=True, exist_ok=True)
        _require_internal_dir(d, self.root, "field registry")  # a symlinked registry redirects marker writes
        for field in new:
            marker = d / field
            _reject_non_regular(marker, "field marker")  # touch(exist_ok=True) would utime THROUGH a link
            marker.touch(exist_ok=True)
        self._fields_seen |= new
        self._fields_mtime = None  # force re-read next time (another writer may add more)

    # --------------------------------------------------------- StorageKVClient
    def put(self, keys: list[str], values: list[Any]) -> list[Any] | None:
        self._register_fields(keys)
        paths = [self._path(key) for key in keys]
        # Intent log FIRST (one flushed batch), then publish: a crash in between leaves
        # log records without objects, which fsck resolves as a safe DROP.
        self._log_ops("P", [p.name for p in paths])
        for path, value in zip(paths, values, strict=True):
            # os.replace would silently REPLACE a symlink at the destination (rename
            # never follows the final component) — no escape, but a symlinked object
            # name is store corruption and must be loud, not papered over.
            _reject_non_regular(path, "canonical object")
            blob = _dumps(value, self.serializer)
            tmp = self._tmp_path(path)
            try:
                tmp.write_bytes(blob)
                os.replace(tmp, path)  # atomic publish: readers never see a torn object
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            STATS["put_objects"] += 1
            STATS["put_bytes"] += len(blob)
        return None

    def get(self, keys: list[str], shapes=None, dtypes=None, custom_backend_meta=None) -> list[Any]:
        out = []
        for key in keys:
            path = self._path(key)
            # A symlinked object would make this deserialize bytes from OUTSIDE the
            # store under a trusted key — refuse before any read (lstat, not open).
            _reject_non_regular(path, "canonical object")
            out.append(_loads(path.read_bytes(), self.serializer))
            STATS["get_objects"] += 1
        return out

    def clear(self, keys: list[str], custom_backend_meta=None) -> None:
        STATS["clear_calls"] += 1
        paths = [self._path(key) for key in keys]
        # Intent log the whole batch before the first unlink. Records for keys that
        # turn out not to exist are harmless on replay: the object is absent either
        # way, so fsck's DROP direction is already correct.
        self._log_ops("D", [p.name for p in paths])
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                continue  # delete-if-exists: mixed batches address keys that never existed
            STATS["clear_objects_deleted"] += 1

    # ------------------------------------------------- durable-backend extras
    def clear_by_index(self, global_indexes: list[int]) -> int:
        """Delete every object belonging to these samples (whole-sample clear).

        Fast path: ``{index} x {known fields}`` delete-if-exists, no listing.
        Fallback: a single ``os.scandir`` when the field registry is unavailable.
        """
        idx_set = {int(i) for i in global_indexes}
        deleted = 0
        fields = self._known_fields()  # encoded names, ready to compose file names
        if fields:
            names = [f"{idx}@{field}{OBJECT_SUFFIX}" for idx in idx_set for field in fields]
            self._log_ops("D", names)  # intent: before the unlinks (superset is safe)
            for name in names:
                try:
                    (self.root / name).unlink()
                except FileNotFoundError:
                    continue
                deleted += 1
                STATS["clear_objects_deleted"] += 1
            return deleted
        STATS["clear_index_scans"] += 1
        victims: list[str] = []
        with os.scandir(self.root) as entries:
            for entry in entries:
                name = entry.name
                if not name.endswith(OBJECT_SUFFIX) or "@" not in name:
                    continue
                try:
                    idx = int(name.split("@", 1)[0])
                except ValueError:
                    continue
                if idx in idx_set:
                    victims.append(name)
        self._log_ops("D", victims)
        for name in victims:
            try:
                (self.root / name).unlink()
            except FileNotFoundError:
                continue
            deleted += 1
            STATS["clear_objects_deleted"] += 1
        return deleted

    def oplog_offsets(self) -> dict[str, int]:
        """Current length of every writer's op log (the cheap fence token).

        Fail closed on a symlinked ``_oplog`` dir or log entry: offsets captured
        through a link would describe files OUTSIDE the store, and a journal built
        on them could never be replayed trustworthily. Raising here fails the
        journal tick (``JournalPrecaptureError``, no stamp published) or the
        checkpoint hook — both safe directions.
        """
        d = self.root / OPLOG_DIR
        try:
            os.lstat(d)
        except FileNotFoundError:
            return {}
        _require_internal_dir(d, self.root, "op-log directory")
        offsets = {}
        with os.scandir(d) as entries:
            for entry in entries:
                if not entry.name.endswith(".log"):
                    continue
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise StoreIntegrityError(
                        f"op log {entry.path} is not a regular non-symlink file; offsets captured "
                        "through a link would fence against bytes outside the store (fail closed)"
                    )
                offsets[entry.name] = entry.stat(follow_symlinks=False).st_size
        return offsets

    def write_manifest(self, checkpoint_dir: str | Path, full: bool = False) -> int:
        """Record what fsck needs to fence post-snapshot writes.

        Default: store each writer's op-log length — O(number of writer processes),
        independent of buffer size. Restore reads the tails past those offsets, which
        is exactly the delta since the snapshot. This is the ONLY fence artifact
        recovery consumes.

        Ordering caveat: this hook runs *inside* ``tq.save_checkpoint``, AFTER the
        controller state was pickled, so offsets captured here are slightly LATE
        relative to the metadata cut (a delete+rewrite pair landing in between could
        evade the fence).  ``JournalWriter.tick`` therefore pre-captures the offsets
        BEFORE calling ``tq.save_checkpoint`` and overwrites this file afterwards —
        early offsets only ever over-fence, which is the safe direction. Snapshots
        taken by verl's direct ``tq.save_checkpoint`` call (trainer checkpoints) keep
        the late capture; see the README's guarantees section.

        ``full=True`` additionally writes a per-object ``(inode, size)`` manifest.
        That is an explicitly OFFLINE AUDIT artifact: it costs O(store) ``stat``
        calls (17 s at 50k objects on FSx Lustre) and, being written after the
        controller pickle, it can record a same-name overwrite from the gap as valid
        — which is why recovery NEVER consumes it (the ``oplog: false`` tick mode
        built on it was removed; see ``_reject_oplog_disabled``).
        """
        d = Path(checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        rows = 0
        if full:
            lines = []
            with os.scandir(self.root) as entries:
                for entry in entries:
                    if not entry.name.endswith(OBJECT_SUFFIX) or "@" not in entry.name:
                        continue
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        continue  # not a writer-emitted object; the audit must not stat through links
                    st = entry.stat(follow_symlinks=False)
                    lines.append(f"{entry.name}\t{st.st_ino}\t{st.st_size}\t{st.st_mtime}")
            rows = len(lines)
            tmp = d / (MANIFEST_FILE + ".tmp")
            # UTF-8 + '\n'-joined records. Encoded names cannot contain '\t'/'\n'/'\r'
            # but MAY contain other Unicode line separators (U+0085/U+2028/...), so the
            # reader must split on literal '\n' only — see fsck._read_manifest.
            tmp.write_text("\n".join(lines), encoding="utf-8")
            os.replace(tmp, d / MANIFEST_FILE)
            STATS["manifest_rows"] = rows
        fh = self._oplog()
        if fh is not None:
            fh.flush()
        (d / OPLOG_OFFSETS_FILE).write_text(json.dumps(self.oplog_offsets()), encoding="utf-8")
        (d / FENCE_FILE).write_text(
            json.dumps(
                {
                    "fence_ts": time.time(),
                    "objects": rows,
                    "root": str(self.root),
                    "oplog": True,  # recovery refuses snapshots recorded with the (removed) oplog:false mode
                    "store_id": self.store_id,  # binds this snapshot to ITS store (fsck aborts on mismatch)
                }
            ),
            encoding="utf-8",
        )
        STATS["manifest_writes"] += 1
        return rows


#: Does the installed TQ's ``KVStorageManager.__init__`` take ``zmq_context``?
#:
#: It does at TQ@main (``storage/managers/base.py:510-538``, added with the shared-context
#: work) and did not in earlier releases. The probe keeps ONE constructor signature
#: working across both instead of pinning a floor here: verl gates its own TQ-checkpoint
#: support on a ``>= 0.1.9`` runtime probe (``trainer_base._tq_supports_checkpoint``), and
#: nothing in that floor promises this parameter. Read on every ``__init__`` call, not
#: baked in at class creation, which is what lets
#: ``tests/experimental/durable_buffer/test_tq_main_compat_on_cpu.py`` drive both branches
#: (including against a two-argument stand-in for the older parent).
_PARENT_ACCEPTS_ZMQ_CONTEXT = "zmq_context" in inspect.signature(KVStorageManager.__init__).parameters


@StorageManagerFactory.register(BACKEND_NAME)
class DurableFSManager(KVStorageManager):
    """KV storage manager whose data outlives the job."""

    def __init__(self, controller_info, config, zmq_context=None):
        """Accepts TQ@main's third constructor parameter and hands it to the parent.

        ``zmq_context`` is accepted for signature compatibility, not because this backend
        needs a caller's context. ``StorageManagerFactory.create`` passes it through
        ``_filter_supported_kwargs`` (``storage/managers/base.py:483-507``), which drops
        any keyword the constructor cannot accept AND logs a warning for each; taking the
        parameter is what silences that warning, and it also stops the day TQ makes the
        keyword load-bearing from turning into a silent behaviour change here.

        What the parent then does with it is the parent's business, and it is worth being
        precise because the obvious assumption is wrong: ``KVStorageManager.__init__``
        accepts ``zmq_context`` and DELIBERATELY IGNORES it, calling
        ``StorageManager.__init__(controller_info, config)`` with no context so the manager
        creates and owns its own (``storage/managers/base.py:514-536``, whose docstring
        says KV backends "keep an independent context rather than drawing on a caller's
        shared socket budget"). Forwarding therefore does NOT make this manager share the
        caller's context today. It means this class states no policy of its own: whichever
        way TQ decides for KV backends, this subclass follows without an edit.

        When the installed TQ predates the parameter the value is dropped, with a warning
        if it was not None, which is exactly what the factory did before.
        """
        config = dict(config or {})
        config.setdefault("client_name", CLIENT_NAME)
        if _PARENT_ACCEPTS_ZMQ_CONTEXT:
            super().__init__(controller_info, config, zmq_context=zmq_context)
            return
        if zmq_context is not None:
            import transfer_queue as _tq

            logger.warning(
                "durable_buffer: the installed TransferQueue (%s) has a KVStorageManager that does "
                "not accept zmq_context; dropping the caller's context, exactly as "
                "StorageManagerFactory did before the parameter existed. The manager creates and "
                "owns its own context.",
                getattr(_tq, "__version__", "unknown version"),
            )
        super().__init__(controller_info, config)

    async def clear_data(self, metadata) -> None:
        """Delete the storage bytes of the samples in ``metadata``.

        Unlike the base implementation this never raises on a field-less batch and
        never trusts the batch schema alone (see module docstring, note 1).
        """
        if metadata.size == 0:
            return None
        if metadata.field_names:
            keys = self._generate_keys(metadata.field_names, metadata.global_indexes)
            self.storage_client.clear(keys=keys)
        else:
            STATS["clear_without_field_schema"] += 1
        # Sweep the indexes: covers tag-only samples (nothing to delete) and
        # field-bearing samples whose fields vanished from the batch schema.
        self.storage_client.clear_by_index(list(metadata.global_indexes))
        return None

    async def save_checkpoint(self, checkpoint_dir: str) -> None:
        """No data copy — write the fence manifest that ``fsck`` consumes on restore."""
        self.storage_client.write_manifest(checkpoint_dir)
        return None

    async def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Nothing to load: the store *is* the data. Reconciliation lives in ``fsck``."""
        return None

    def close(self) -> None:
        try:
            self.storage_client.close()
        finally:
            super().close()


def backend_config(root: str | Path, serializer: str = "tq", create: bool = True, oplog_fsync: bool = False):
    """Build the ``transfer_queue`` config block for this backend (test/harness helper).

    ``create=False`` attaches to an existing, already-stamped store without writing
    anything (offline tools); there is no ``oplog`` knob — the op log is mandatory.
    ``oplog_fsync=True`` adds the node-crash barrier described in
    ``DurableFSClient._log_ops`` (default OFF).
    """
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "backend": {
                "storage_backend": BACKEND_NAME,
                BACKEND_NAME: {
                    "root": str(root),
                    "client_name": CLIENT_NAME,
                    "serializer": serializer,
                    "create": create,
                    "oplog_fsync": bool(oplog_fsync),
                },
            }
        }
    )
