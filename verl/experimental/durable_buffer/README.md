# Crash-durable replay buffer for verl trainer-V1

**Status**: working, tested on CPU (253 package tests + verl's own V1 suite), benchmarked.
**Scope**: `trainer.use_v1=true`. The durability guarantees cover `trainer_mode ∈
{colocate_async, separate_async}`; `sync` is supported for registration and configuration
but persists no rollout payloads into TransferQueue, so there is nothing for the journal to
recover there. Either way this is the TransferQueue data plane: it does not touch
`one_step_off_policy` or `fully_async_policy`, which use other queues.

## The problem

verl V1 already persists every rollout trajectory as it is produced — into
TransferQueue. But every shipped TQ storage backend lives and dies with the job
(`SimpleStorage` = in-memory Ray actors; Mooncake/YuanRong = external caches without a
recovery contract), so a trainer crash throws away **finished-but-untrained**
trajectories. verl also already contains the resume path
(`tq.save_checkpoint` / `tq.load_checkpoint` / `_reissue_inflight_prompts`), and it is
LIVE: TransferQueue is a core verl dependency installed from
`git+https://github.com/Ascend/TransferQueue.git@main` (`requirements.txt`,
`pyproject.toml`), and the capability gate is a runtime probe for `>= 0.1.9` plus both
methods being callable (`trainer_base._tq_supports_checkpoint`), not a pin. What is
missing is not the mechanism but the window: everything generated since the last
`trainer.save_freq` checkpoint still dies with the process, because the snapshot is
only as good as the backend it references.

Generation is the expensive half of RL: at 25 s/trajectory (measured on our 2-node
Qwen3.5-27B run) a buffer of 18 unconsumed groups × 8 samples is **an hour of generation work**
that today evaporates on any restart.

## What this package adds

| property | verl today | with this package |
|---|---|---|
| trajectory bytes survive process death | ❌ | ✅ written as produced to FSx/NFS |
| buffer metadata survives | only at `trainer.save_freq` (TQ ≥ 0.1.9) | ✅ every `journal_interval` seconds |
| consumed samples never retrain | ✅ | ✅ for early-offset (`JournalWriter`) restores — delete IS the consume marker; the trainer-checkpoint fallback keeps a ms-wide late-offset window (see Guarantees) |
| restart prompt accounting | ❌ re-serves every dispatch since the last checkpoint | ✅ journaled dataloader state shrinks it to two bounded one-tick windows: duplicate-dispatch and skip (see Guarantees) |
| complete groups whose prompt tag was lost | re-generated | ✅ rescued (`promote_complete_groups`) |
| verl core changes required | — | **none**: registration rides verl's existing `VERL_USE_EXTERNAL_MODULES` hook (`verl/__init__.py`, documented in `docs/extend_guide.rst`). Zero TQ core changes; everything else is config + a registered backend |

## Usage

1. Point TransferQueue at the durable backend (this is the only required change):

```yaml
transfer_queue:
  backend:
    storage_backend: DurableFS
    DurableFS:
      root: /shared/fs/runs/${trainer.experiment_name}/buffer
      serializer: tq          # tq (default, TQ's own codec) | torch
      oplog_fsync: false      # default. true = fsync the intent log before each
                              # publish, which extends the ordering guarantee from a
                              # killed process to a lost node (see Guarantees). One
                              # barrier per mutation batch, and it costs producer
                              # throughput on a shared filesystem, so it is opt-in.
```

2. Make sure the backend is importable in every process that talks to TQ (driver,
   TaskRunner, rollout/trainer workers). Set one env var before launch:

```bash
export VERL_USE_EXTERNAL_MODULES=verl.experimental.durable_buffer.register
# and forward it into Ray workers:
#   +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=verl.experimental.durable_buffer.register
```

The env var is honored by verl itself: `verl/__init__.py` imports the modules it
names in every process that imports verl (`import_utils.import_external_libs`,
documented in `docs/extend_guide.rst`), so this package adds NO verl-core change;
the `verl.plugins` entry-point group is the env-var-free alternative for a backend
shipped as its own package. Do NOT use Ray's job-level `worker_process_setup_hook`
for this: it broke per-actor GPU assignment in the cluster run (every FSDP rank
landed on one CUDA device).

3. Journal on a cadence — pick one:

```python
# (a) background thread in the driver (recommended: buffer semantics untouched)
from verl.experimental.durable_buffer import register
register.start_journal("/shared/fs/runs/<name>/journal",
                       interval_s=2.0, dataloader=trainer.train_dataloader)

# (b) custom-sampler subclass (config-only). Also runs a background journal thread
#     by DEFAULT; durable_buffer.background=false opts into poll-tick mode, whose
#     effective cadence is the training-step time, not interval_s.
#   trainer.v1.sampler.custom_sampler.path = verl/experimental/durable_buffer/buffer.py
#   trainer.v1.sampler.custom_sampler.name = DurableReplayBuffer
#   trainer.v1.sampler.sampler_kwargs.durable_buffer.journal_root = /.../journal
# NOTE: verl passes filter_groups_metric / sync_refill_failed_groups only to
# non-custom samplers; re-supply them via sampler_kwargs.durable_buffer.* if the
# recipe uses them (the subclass forwards them into the base constructor).
```

Pick one, and mean it: a ring root has exactly ONE writer, enforced by a lease, so
wiring both paths at one root leaves the second one running with journaling OFF rather
than interleaving two writers' snapshots (see Guarantees). How that is signalled depends
on which wiring was refused: the subclass reports `durable_buffer/journaling_enabled = 0`
in the trainer's metrics, while `start_journal` returns `None` and warns, because it is a
plain function with no metrics channel of its own. Journaling is what a refused lease
costs, never the run, because the same refusal happens to a restart inside the takeover
window after a hard kill. `start_journal` records the trainer's `global_steps` by
default, from the calling trainer or from an explicit
`trainer=`/`metadata_fn=`; slot ranking and the journal-versus-checkpoint freshness
comparison both prefer the step and fall back to timestamps, so a run that records no
step is reported at startup rather than left to a clock tie-break later.

4. On restart, reconcile before sampling:

```python
from verl.experimental.durable_buffer import recover
from verl.experimental.durable_buffer.journal import latest_journal

journal = latest_journal("/shared/fs/runs/<name>/journal")
if journal is not None:
    recover(tq, store_root, journal, partition="train",
            group_size=config.actor_rollout_ref.rollout.n,
            # OPERATOR ASSERTION: this run used a uniform rollout.n. Without it,
            # prompts lacking a persisted per-prompt count are never promoted.
            assume_uniform_group_size=True)
    # recover() restores the BUFFER only — apply the journaled dataloader position
    # separately, or every prompt dispatched since the trainer checkpoint is
    # re-served on resume:
    dl_state = journal / "dataloader_state.pt"
    if dl_state.exists():
        trainer.train_dataloader.load_state_dict(torch.load(dl_state, weights_only=False))
```

`recover()` = fail-closed preflights (store identity, fence artifacts) →
`tq.load_checkpoint` → `fsck` → optional group rescue. verl's own
`_reissue_inflight_prompts` then restarts the prompts that were still generating.
`restore_tool` (below) is the complete offline path: it rewrites both the
checkpoint's TQ snapshot AND its `data.pt` in one transaction. Run it between the
crash and the restart:

```bash
# preview the complete plan first: ZERO writes of any kind
python -m verl.experimental.durable_buffer.restore_tool \
    --store /shared/fs/runs/<name>/buffer \
    --ckpt-dir /shared/fs/runs/<name>/ckpts \
    --journal /shared/fs/runs/<name>/journal \
    --group-size 8 \
    --dry-run

# then execute exactly that plan (drop --dry-run)
python -m verl.experimental.durable_buffer.restore_tool \
    --store /shared/fs/runs/<name>/buffer \
    --ckpt-dir /shared/fs/runs/<name>/ckpts \
    --journal /shared/fs/runs/<name>/journal \
    --group-size 8
```

Flags (matching `python -m verl.experimental.durable_buffer.restore_tool --help`):

* `--store` (REQUIRED) — durable store root (the backend's `DurableFS.root`).
* `--ckpt-dir` (REQUIRED) — `trainer.default_local_dir` (the `global_step_*` parent).
* `--journal` — journal ring root (`JournalWriter` root); omit to reconcile from the
  checkpoint's own TQ snapshot (the no-journal fallback — see Guarantees).
* `--partition` — TQ partition to restore (default `train`).
* `--group-size` — rollout.n; enables the group-completeness rescue and is the
  OPERATOR ASSERTION that the run used a uniform rollout.n (see PROMOTE).
* `--serializer` — `tq` (default) or `torch`; must match the backend config.
* `--dry-run` — print the complete plan and exit; zero writes of any kind.
* `--journal-slot` — reconcile from THIS ring slot (a `journal-NNN` name) instead of
  the newest trustworthy one. Without it the ring is walked newest-first and a slot
  whose artifacts fail validation falls back to the next-older complete slot, every
  refusal reported under `source_slot_rejections`.
* `--force` — freshness override only: apply the journal even when the checkpoint
  is as fresh or freshness cannot be established.
* `--allow-live-controller` — proceed although a TransferQueue controller is reachable
  (DANGEROUS; see the exclusivity paragraph below, which also says what the probe cannot
  see). Use only for a leftover controller you have verified belongs to no live run. It
  is not the remedy when the probe merely could not decide: clearing a dead cluster's
  address pointer keeps the check, and this flag drops it.
* `--force-store-mismatch` — NARROW identity override: proceed when the journal's
  store identity is VALID but DIFFERENT from the store's (you vouch for a known
  foreign lineage — DANGEROUS). A missing/empty/invalid identity on either side
  always aborts and is never overridable.

Reconciliation locking is AUTOMATIC: the mutating phase runs under an exclusive
per-checkpoint lock (`reconcile.lock` in the step dir), and a concurrent restorer
fails fast with status `locked` and the holder's identity. There is deliberately NO
force-break flag — a lock left by a hard-killed run is removed manually after
verifying the recorded host/pid is dead; the tool never breaks a lock itself.

STORE EXCLUSIVITY comes first, before the preflight and in BOTH modes. Reconciliation
deletes store objects and rewrites controller metadata, so it is only correct against a
buffer whose writers and controller are gone. The tool probes for a reachable
`TransferQueueController` actor and refuses with status `live_controller_present` if it
finds one. `--dry-run` is not exempt: `tq.init` ignores its `create=False` config
whenever a controller is reachable (TQ returns from `init` at
`if _init_from_existing()`, before the config is used) and `recover(..., dry_run=True)`
still calls `tq.load_checkpoint`, so a "preview" against a live job would replace that
job's buffer index while leaving the store and checkpoint trees byte-identical, exactly
as advertised.

Actor names are scoped to a GCS, so where the probe runs decides what it can see, and
the tool runs it in two different places. A caller that already has Ray up (the
in-process `recover` path) is probed in its own Ray, which is where its TransferQueue
lives. A CLI invocation has no Ray, so the tool enumerates every Ray instance this host
can discover (`RAY_ADDRESS`, `/tmp/ray/ray_current_cluster`, and every local
`gcs_server` process), attaches to each in turn only long enough to probe it, detaches,
and refuses if any of them holds the controller. All of them and not one: Ray's own
resolution returns a single address and its last step picks an arbitrary local
`gcs_server`, so on a host running a second Ray instance the probe would ask a GCS that
is not the run's, find no controller there, and report absence with confidence rather
than doubt, which is the one shape of wrong answer that lets the reconcile delete objects
under a live job. The reconcile then runs in a PRIVATE Ray instance of the tool's own
(`address="local"`, which starts a new instance even when `RAY_ADDRESS` is set), so the
scratch TransferQueue can never bind a cluster controller. Probing after that isolation,
which is what an earlier version did, could not work by construction: the private
instance holds no actors, so the answer was "absent" whatever the cluster was doing.

Discovery reporting an address does NOT mean a cluster is there, and on the crash path
that difference is the whole story. `RAY_ADDRESS` outlives the head it names, and
`/tmp/ray/ray_current_cluster` is written by `ray start` and removed by nothing, so a
cluster that died leaves a pointer behind, and this tool runs precisely then. Every
discovered address is therefore pre-checked with one TCP connection (2 s): nothing
listening means no live cluster there, which is the same answer as no address at all, and
if that is true of all of them the tool proceeds to its own isolated Ray. `ray.init`
cannot be asked instead, because it retries the GCS connect 20 times at about 41 s each:
820 s measured against a dead pointer, ending in doubt, which fails closed into refusing
the very restore the crash made necessary. An address that accepts a connection but cannot
be joined is still doubt and still refused, now with a bounded attach (one retry, not
twenty) and a detail that says which pointer to clear; doubt about one instance outranks
the others that came back clean. Every result records what the probe did under
`controller_probe`, including which addresses were discovered and whether anything
answered at each, because "no controller found" and "no cluster found" are different
facts.

The refusal detail is written for the doubt cases separately: they do not report a
controller as reachable, they do not tell an operator to stop a job that a stale pointer
invented, and they name the cheap remedy (clear the stale `RAY_ADDRESS`, delete a dead
cluster's address file, which restores the check) before mentioning
`--allow-live-controller`, which removes it.

What this does NOT check is the store. A run whose cluster this host cannot discover is
not detected: a different head node with no pointer here leaves nothing to enumerate, and
so does a raylet this user cannot read in `ps`, which is where the local scan gets its
addresses. Nor is a cluster this host cannot reach: an address that does not answer a TCP
connection within the 2 s probe reads as no cluster, so the refusal does not apply and the
tool proceeds, whether the cluster behind that address is dead or merely unreachable from
here. And neither is any other writer publishing into `--store`; there is no store-level
writer lease. So the refusal is the controller half of the exclusivity precondition, and
the store half stays an ASSUMPTION: point the tool at a store nothing is publishing into.
In real mode it fences, which unlinks objects a live writer is still creating.

A refused source no longer blocks the tool's own repair. When an earlier run was killed
inside its two-rename publication swap, the checkpoint has no live `transfer_queue` dir,
only `transfer_queue.pre_reconcile`; restoring that is a rename of this tool's own
backup and depends on no journal artifact, so it runs whenever the checkpoint layout
validated, and the result is status `healed_only` (healed, nothing reconciled). That
matters because verl's resume does `if os.path.exists(tq_ckpt_path)` and skips a missing
TransferQueue dir WITHOUT a warning: the run would otherwise start with an empty
controller, every durable trajectory orphaned, and the only trace a status line in the
output of a tool the operator ran before restarting. For the same reason the exit code
is non-zero for any outcome that leaves the checkpoint without a live `transfer_queue`
dir (`checkpoint_tq_present: false` in the result), whatever the status says, and
`healed_only` is non-zero too because the refusal it reports is the operator's to
resolve.

Checkpoint/journal layout is validated FAIL CLOSED before any lock or mutation:
the selected step dir, its live/backup `transfer_queue` dirs, `data.pt` (+ backup),
marker/lock files, journal-ring slots and every recovery-critical artifact file
must be real (non-symlink) dirs/files resolving under their roots. A symlink
anywhere there is refused as a controlled `invalid_checkpoint_layout` /
`unrecoverable_journal` result (and hostile journal slots are simply never
selected) instead of being followed. The same lstat discipline covers the last
indirections: the `latest_checkpointed_iteration.txt` pointer must itself be a
regular non-symlink file (else it is ignored and the `global_step_*` scan
decides), the store's `_store_id` lineage stamp is never read through a link (a
redirected stamp aborts like a mismatch, never overridable), and completing an
interrupted publication first PROVES the marker's recorded source path —
validated UNCONDITIONALLY for every pending marker: malformed values (empty,
NUL-carrying, non-string, oversized) classify unprovable before any syscall, and
every component must be a real non-symlink directory under the `--journal` ring
or the step dir — before reading any identity/fence/dataloader artifact through
it (each of those artifacts is lstat-gated AGAIN at the read/copy itself); an
unprovable source is discarded with the checkpoint's data intact (an explicit
plan entry, identical in dry and real mode).

## Recovery contract

Let *T* be the last journal tick before the crash.

* trajectories finished **before T** → restored, sampled, trained. **Not regenerated.**
* trajectories finished **after T** → their objects are fenced and the prompts are
  re-issued (verl's existing refill), i.e. the loss window is the journal cadence.
* samples consumed (trained) **after T** → dropped by fsck; they never come back, so no
  duplicate training (journal restores are early-offset — see Guarantees for the
  trainer-checkpoint fallback's window).
* prompts still generating at T → re-issued from the payload verl already persists.
* a group fully written at T but whose prompt tag was not yet published → promoted to
  `finished` and trained instead of regenerated (per-session completeness check;
  requires the expected group size to be durably recorded on the prompt, or the
  operator's explicit uniform-n assertion — see PROMOTE below).

### Guarantees, exactly

What the package guarantees, and every residual window. Race directions are chosen so
that storage/metadata races degrade to *extra regeneration or a bounded duplicate
dispatch* (wasteful, benign) rather than *serving wrong bytes* — with ONE inline
exception: the "never wrong bytes" direction holds for early-offset (`JournalWriter`)
snapshots, while the trainer-checkpoint fallback keeps a delete-and-rewrite window as
wide as the controller pickle, tens of milliseconds at the sizes measured (its bullet below). Prompt accounting has exactly two
bounded windows (below). There is deliberately NO blanket "never loses /
never duplicates" claim anywhere in this package:

* **Never mislabeled data — for early-offset (JournalWriter) snapshots, against
  process-level failure.** Every store mutation is **intent-logged** (the op-log
  record is flushed BEFORE the object is published/unlinked), so a crash between the
  two leaves a logged-but-unpublished record that fsck resolves as fence+DROP. The
  reverse order would leave mutations the fence cannot see — under TQ's index reuse
  that serves another sample's bytes. A journal tick captures the op-log offsets
  BEFORE `tq.save_checkpoint` pickles the controller: mutations landing during the
  snapshot are replayed and fenced (over-fencing, safe), never missed. The
  guarantee is scoped twice over. It covers exactly those early-offset snapshots: the
  direct trainer-checkpoint fallback captures its offsets AFTER the controller pickle
  and retains a residual delete-and-rewrite window (see the trainer-checkpoint bullet
  below). And the record is FLUSHED, not fsynced, so the ordering holds across a
  killed process (which the kill drills exercise) and not across a lost node, where
  buffered log bytes and a server-committed `rename` do not share a durability class
  and a published object can outlive the record that was supposed to cover it.
  `DurableFS.oplog_fsync: true` adds one `os.fsync` per mutation batch and extends the
  ordering to a node crash; it is off by default because a barrier on a shared
  filesystem is expensive enough to be expected to move the producer rate, and that cost
  has not been measured here.
* **Snapshot/mutation straddles resolve to fence/drop.** TQ mutates STORAGE and
  CONTROLLER metadata in two steps (put: object published, then metadata; clear:
  object unlinked, then metadata), and a snapshot can land in between. fsck's
  membership rules make both directions converge: an object the snapshot does not
  expect is FENCED (put-straddle — regenerated), metadata whose object is gone is
  DROPPED (clear-straddle — otherwise the restored controller re-serves a key whose
  read can only fail). Wasteful, never wrong. Op-log mode does this with one
  names-only readdir — no per-object `stat`. A clear batch unlinks index-major, so a
  snapshot can also expect a MIX of gone and present objects; the DROP set is closed
  over trajectory groups (a group that lost one member loses the rest, and its prompt
  with it) so no restore publishes a group nothing will ever complete.
* **Trajectory loss window = journal cadence.** Anything finished after the last tick
  is regenerated via verl's stock re-issue.
* **Prompt accounting: two bounded windows, not exactness.** Within a tick the
  dataloader state is captured BEFORE the TQ cut (the reverse ordering would
  silently lose every prompt fetched during the snapshot). The two residual windows:
  (a) **bounded duplicate-dispatch window** — a prompt fetched after the loader
  capture whose group still lands in the restored buffer is re-served although its
  trajectories are already there; bounded by the dispatches completing between the
  loader capture and the TQ cut, i.e. within one tick. In the journal path the
  pre-captured offsets usually fence those groups (registered after the fence
  token), converting the would-be duplicate into a regeneration; checkpoint-sourced
  restores (late offsets, loader from `data.pt`) keep the duplicate direction.
  (b) **skip window** — a dispatch spanning BOTH cuts (prompt fetched before the
  loader capture, registered after the fence cut) is recorded as served yet absent
  from the buffer, so a restore skips it for the current epoch; bounded by the
  dispatches in flight at the tick instant (verl dispatches loader-fetch →
  `kv_batch_put` synchronously on the trainer thread, ≈ one), verl's group-refill
  keeps training batches full with other prompts, and the skipped prompt returns
  next epoch. Neither window is detectable package-side (the prompt exists in
  neither the store nor the snapshot, and `StatefulDataLoader` state is opaque); a
  dispatch WAL in verl or a controller-side WAL in TQ would close both.
* **Fence-source honesty: op-log or abort.** Op-log replay is the ONLY
  recovery-grade fence source, and every artifact it depends on is validated
  fail-closed: the offsets fence must be a strict mapping of log name → nonnegative
  integer, the logs must not have shrunk or vanished, their bytes must be valid
  UTF-8, and every record must parse AND name a canonical encoded store-object
  basename (record names feed `unlink` calls under the store root, so a traversal
  or absolute-path record — `P ../x`, `P /etc/x` — aborts instead of deleting
  outside the store). The log/offset INFRASTRUCTURE must also be exactly what the
  writers create — real (non-symlink) dirs and regular files; a symlinked
  `_oplog`, log file or snapshot artifact aborts the same way, and the store-side
  writers refuse symlinked internals outright (`StoreIntegrityError`) while fsck
  FENCES canonical-named non-regular entries (the link, never its target).
  ANY violation — as well as a snapshot recorded with
  `oplog: false` (the REMOVED full-manifest tick mode: its manifest was written
  AFTER the controller pickle, so a same-name overwrite in the gap was recorded as
  valid) — ABORTS recovery before `tq.load_checkpoint`, with no automatic
  degradation; an untrusted fence artifact is never read as "nothing changed". A
  failed offsets pre-capture fails the journal tick itself (no stamp file), so
  `latest_journal` skips that slot and recovery uses the previous good snapshot.
  `write_manifest(full=True)` survives only as an offline audit artifact recovery
  never consumes; `mode="mtime"` exists as an explicitly-labeled legacy flow.
* **One journal writer per ring root, and slots that prove it.** A ring root is
  leased (`journal_writer.lock`, `O_CREAT|O_EXCL`, holder identity + heartbeat), so a
  second `JournalWriter` fails at construction instead of interleaving: two writers
  both start at `journal-000`, and the artifact writes that follow
  `tq.save_checkpoint` resolve their paths at write time, so one writer's offsets and
  stamp can land on the controller pickle the other just published. Such a slot passes
  every completeness gate while its fence is captured LATER than its controller state,
  which is the one ordering the design exists to prevent. Independently of the lease,
  each tick writes ONE `fence_ts` and its writer id into both TQ's `metadata.json` and
  `journal_stamp.json`, and a slot whose two sides disagree is REFUSED at read time. A
  lease whose holder has stopped ticking is taken over with a warning, because a
  crashed driver must not leave a ring root unusable on the restart path this package
  serves; the read-time cross-check is what covers that takeover. Inside the takeover
  window (`max(30s, interval_s * 5)`) a restart is still refused the lease, and both
  wiring sites treat that the way `tick` treats every other failure: they warn and run
  with JOURNALING OFF. The signal differs by site, because only one of them emits
  metrics: the `DurableReplayBuffer` subclass reports
  `durable_buffer/journaling_enabled = 0` alongside its sampling metrics, while
  `start_journal` returns `None` and leaves the warning as the only trace. A refused
  lease costs journaling, never the resumed run.
* **The controller pickle's schema is pinned.** A journal artifact is TransferQueue's
  own pickle of its controller objects, which carries live class references and no
  schema version, and verl installs TransferQueue from `@main`, so a container rebuild
  between the crash and the restore could previously make a journal unreadable or,
  worse, readable with drifted semantics. Each tick stamps `transfer_queue.__version__`
  and a fingerprint of the attributes the snapshot ITSELF carries, read back from the
  controller pickle the tick just wrote and compared at restore against the attributes
  `load_journal` reads (`fsck.SNAPSHOT_ATTRIBUTES`). Deriving the recorded value from the
  real snapshot is what makes a TransferQueue-side rename detectable: the renamed
  attribute drops out of the fingerprint, so the mismatch is refused fail-closed at
  restore with a message naming both versions and both field sets. A bare version
  difference with a matching fingerprint warns and proceeds, since every field being read
  is present and unmoved. A schema-pin test takes a real snapshot and asserts those
  attributes exist, so an upstream rename also fails in CI rather than at 3am on a
  restore, and `load_journal` reads all five by plain attribute access, so a rename that
  somehow reached the parse is an unrecoverable-journal refusal and never a silent
  default. The cross-version evidence in this package covers stored object BYTES; this
  covers the metadata plane.
* **The whole ring is reachable, and a mid-record fence does not abort.** Recovery
  walks complete slots newest-first (ranked by a persisted monotonic tick sequence,
  falling back to the stamp timestamp) rather than being limited to the newest one, so
  one poisoned slot no longer makes the journal unusable: the restore tool falls back
  to the next-older complete slot and reports which slot it used and why the newer ones
  were refused (`--journal-slot` pins the choice instead). An offsets fence that lands
  inside a record — a file size sampled while writers append from other nodes — is
  aligned DOWN to the previous boundary and the straddling record is replayed as
  post-cut, which over-fences; a record still unparseable after alignment keeps its
  abort.
* **Trainer-checkpoint snapshots** (verl's direct `tq.save_checkpoint` at
  `save_freq`) capture their op-log offsets *inside* the storage-manager hook, i.e.
  slightly AFTER the controller pickle — the unsafe direction, kept only because verl
  calls TQ directly there. `restore_tool` therefore prefers a journal whenever one is
  as fresh; reconciling against a checkpoint (the no-journal fallback) carries a
  window as wide as the controller pickle in which a delete+rewrite pair can evade the fence — that
  fallback is explicitly OUTSIDE the "never mislabeled data" guarantee above, which
  covers early-offset `JournalWriter` snapshots only.
* **Store identity, fail closed.** Journals and fence manifests are stamped with the
  store's identity (`_store_id`, created at bootstrap; it is *logical lineage* — a
  copied store keeps it). `fsck`/`recover`/`restore_tool` verify it BEFORE any
  mutation, including `tq.load_checkpoint`, and treat a missing/empty/invalid
  identity on either side as a mismatch. `--force-store-mismatch` overrides ONLY a
  mismatch between two VALID identities (you vouch for a known foreign lineage);
  missing/empty/invalid identities are never overridable — an unstamped store is
  indistinguishable from a wrong `--store` path. This is distinct from the
  freshness `--force`. The restore tool's publication transaction is additionally
  BOUND to the lineage: the marker and the published TQ dir record the store's
  `_store_id`, and a half-finished transaction left behind by a deleted store is
  discarded (with a report) rather than completed against a store recreated at the
  same path.
* **Clocks are not trusted.** Fencing uses op-log replay (order, not time) or
  `(inode, size)` write identity; the wall-clock `mtime` mode exists only as an
  EXPLICITLY-requested legacy flow (never auto-selected) and conservatively
  over-fences by a margin.

### The fsck rules, and why each exists

| rule | what it does | the failure it prevents |
|---|---|---|
| DROP | clear metadata for samples whose expected objects are gone — in EVERY partition of the snapshot — plus the rest of any trajectory group that lost a member, and then that group's prompt when it is terminal (`finished`/`failure`; an in-flight prompt stays for re-issue, which clears its old trajectories itself) | consumed samples resurrecting and being trained twice; restored metadata pointing at deleted objects in a partition you did not ask to restore (also resolves the clear-side snapshot straddle); a group left holding some of its trajectories, which no re-issue and no promotion will ever complete (a clear batch unlinks index-major, so a snapshot can land mid-loop) |
| FENCE (write identity) | delete objects unknown to the journal **or rewritten since it** | TQ recycles freed sample indexes, so a stale index can hold *another sample's bytes* — silent training on mislabeled data (membership fencing also resolves the put-side snapshot straddle) |
| store-wide fencing | fence against the union of all partitions in the snapshot | a `train` restore deleting `val`'s objects |
| store identity check | abort before ANY mutation (incl. `load_checkpoint`) unless both sides carry the same valid `_store_id` — missing/empty/invalid fails closed and is never forceable (`--force-store-mismatch` covers only two valid, different identities) | destructively "reconciling" a store against another run's journal, or against a wrong `--store` path |
| terminal-only prompt drop | never drop `pending`/`running` prompts | losing in-flight work that `_reissue_inflight_prompts` would have restarted |
| PROMOTE | rescue groups where every session `0..n-1` has a successful trajectory; `n` must be DURABLY KNOWN: the prompt's own persisted `__rollout_n__` when present, else the caller's rollout.n ONLY under the explicit `assume_uniform_group_size` operator assertion (restore_tool's `--group-size` IS that assertion) — prompts with neither are left for re-issue, as is any prompt whose record is unreadable or whose observed sessions reach past the assumed `n` | regenerating a group whose generation already completed; counting KEYS would let several outputs of ONE session fake completeness; verl does NOT persist batch-level `__rollout_n__` overrides (and sync mode persists no payload at all), so an absent field proves nothing — promoting at an assumed size would train a partial group as "complete" |

The fence source is **op-log replay** (order-based write identity, O(changes since
the snapshot) plus one names-only readdir for membership — no per-object `stat`),
and it is the ONLY source recovery consumes: untrusted artifacts abort (see the
guarantees above). The full `(inode, size)` manifest (`write_manifest(full=True)`)
is an offline AUDIT artifact — clock-free, but written after the controller pickle,
which is exactly why the removed `oplog: false` mode that consumed it was unsound;
on S3 the audit role would be played by `ETag`/`VersionId`. A wall-clock mode
(`mode="mtime"`) exists for explicitly-labeled legacy flows (a store with no usable
fence artifacts at all); it is never auto-selected and conservatively re-generates
up to one margin of just-written trajectories.

## Measured results

Three layers of evidence: a real kill drill on an 8×B200 cluster node (an
internal-cluster measurement), a deterministic recovery benchmark, and cost microbenchmarks
(both produced by the harnesses in `experiments/durable_buffer/`; raw outputs and
summaries in `experiments/durable_buffer/results/`). The benchmark numbers are
re-runnable on any CPU box; the kill-drill, FSx-Lustre, and NVMe rows are cluster
measurements from the validation runs.

### Cluster kill drill (8×B200, Qwen2.5-0.5B GSM8K, V1 `colocate_async`, `rollout.n=4`, response length 512, `save_freq=5`, 2026-08-23)

`kill -9` of the whole process tree at step 7 (last checkpoint: step 5), then resume of
the same experiment name. Both arms ran to step 14 and exited 0:

| resume arm | re-issued prompt groups | objects regenerated to finish |
|---|---|---|
| stock verl (no reconciliation) | **32**, every in-flight group | 48 192 |
| `restore_tool` + stock verl | **0** | 44 864 |

Δ = **3 328 objects = 128 trajectories × 26 stored fields**, which is the 32 in-flight
groups at `rollout.n=4`: exactly the generation the stock resume repeated and the
reconciled resume did not. The zero is a property of the cut rather than of
reconciliation itself: the stock arm resumes from the step-5 checkpoint, where those 32
groups were in flight, while the reconciler restores the journal's step-7 cut, where
they had already resolved. A journal cut that still holds prompts in flight re-issues
them, unless the optional PROMOTE rule proves their groups already complete. Reconciliation itself took **4.889 s** over 6 972 op-log
records, with one names-only readdir of 4 022 names (0.024 s) and zero per-object `stat`
calls; the whole tool invocation was 19.07 s, the remaining ~14 s being Ray and
TransferQueue bring-up, which is worth keeping separate from the reconciliation cost. The
resumed dataloader position came from the journal (no duplicate dispatches in this drill).
On the same filesystem the journal tick measured **58 to 163 ms** across 5 000 to 50 000
live objects (`costfsx_journal_*`). One thing the drill did NOT show: `trajectory_spans`
on the first resumed step read the same 1.0 on both arms, and the reconciler reported no
promoted trajectories, so the reconciled arm's advantage here is the prompt and object
accounting above and nothing is claimed about the age of the data in that batch. On this
deliberately small/fast workload the recovered work is ~2 minutes of generation; the same
mechanism at the 27B or agentic scale is what the benchmark below quantifies.

An earlier validation run (2026-07) at a heavier generation shape (`rollout.n=8`, response
length 1024) produced the same accounting: stock resume re-issued 29 groups; the
reconciled resume, restoring a journal cut past those groups, re-issued 0 and regenerated **6 032 fewer objects = 232 trajectories =
the same 29 groups × 8**. Those two figures are from that July run and have not been
re-measured on this branch; the light shape above is what this branch's artifacts carry.
The recovered work scales with group size × per-trajectory generation cost, which is what
the presets below quantify for expensive workloads.

### Crash cost — preset `qwen27b` (8 groups/step, `rollout.n=8`, 100 s/step, 25 s/trajectory, `save_freq=5`)

| arm | kill@step | untrained@kill | recovered | traj lost | **gen-s wasted** | dup traj |
|---|---|---|---|---|---|---|
| `none` (stock in-memory buffer) | 6 | 12 | 0 | 96 | **2400** | 0 |
| `ckpt` (journal only at `save_freq`) | 6 | 12 | 6 | 48 | **1200** | 0 |
| `journal` (2 s cadence) | 6 | 12 | 12 | 0 | **0** | 48 |
| `journal_dl` (+ dataloader state) | 6 | 12 | 12 | 0 | **0** | **0** |
| `none` | 8 | 16 | 0 | 128 | 3200 | 16 |
| `ckpt` | 8 | 16 | 0 | 128 | 3200 | 16 |
| `journal` | 8 | 16 | 16 | 0 | 0 | 144 |
| `journal_dl` | 8 | 16 | 16 | 0 | 0 | **0** |
| `none` | 10 | 20 | 0 | 160 | 4000 | 0 |
| `ckpt` | 10 | 20 | 20 | 0 | 0 | 0 |
| `journal` / `journal_dl` | 10 | 20 | 20 | 0 | 0 | 0 |

Reading: the journal arm recovers **everything** at every kill point. Checkpoint-granular
durability (what queue checkpointing alone buys) grades with distance from the boundary — everything
when the crash lands on a checkpoint step, half the untrained work one step past it
(kill@6: 6 of 12 groups), and **nothing** once the buffer
has turned over since the last one (kill@8). Journaling the dataloader state is what
removed the duplicate work in these runs — this benchmark's kills land between
dispatches, so both prompt-accounting windows (see Guarantees) are empty here.

### Crash cost — preset `agentic` (300 s/trajectory, `rollout.n=8`)

| arm | kill@7 gen-s wasted | kill@9 gen-s wasted |
|---|---|---|
| `none` | 16 800 | 21 600 |
| `ckpt` | 16 800 | 21 600 |
| `journal` / `journal_dl` | **0** | **0** |

≈ **4.7–6.0 hours of generation occupancy recovered per crash** in the long-rollout regime
where multi-minute trajectories make every lost rollout expensive; the unit sums
concurrent generation work, so this is work bought back, not elapsed machine time.

### Cost of durability

| producer path (1 process) | in-memory | durable, local disk | durable, FSx Lustre (cluster) |
|---|---|---|---|
| trajectories/s | ~17 200–18 800 | ~2 800–2 900 | ~86 |
| ms per trajectory | 0.05–0.06 | 0.34–0.36 | 11.5 |

The cost is per-object metadata operations, not bytes (within 8% at resp_len 512 versus
2048). Even the FSx number is ≥100× a real rollout fleet's per-process production rate
(generation takes seconds-to-minutes per trajectory), and writes happen inside the
agent-loop workers, off the training critical path — the cluster run's step time was
unaffected. Inside the recovery benchmark the end-to-end put latency including TQ RPC
runs 0.92–0.95 ms on the in-memory arm and 1.06–1.39 ms on the durable arms; the
~0.25 ms gap of means is the durable write cost seen from inside the loop (the overhead
benchmark's direct added-cost figure is 0.28–0.30 ms per trajectory).

| live objects | journal tick s, `costfsx_journal_*` | full manifest s, `costfsx_fence_*` | manifest / tick, within the fence harness |
|---|---|---|---|
| 5 000 | 0.058 | 0.114 | 1.2× |
| 20 000 | 0.084 | 8.12 | 86.8× |
| 50 000 | 0.163 | 20.8 | 144.5× |

Each column comes from one harness, because the two measure different things and their
ticks differ: `costfsx_journal_*` times a whole journal tick in isolation (0.0584, 0.0842,
0.1634 s, the 58 to 163 ms the RFC quotes), while `costfsx_fence_*` times a tick and a
full manifest back to back in the same process so the ratio is internally consistent
(`manifest_over_tick` in those artifacts, against fence-harness ticks of 0.0948, 0.0936
and 0.1439 s). Reading a ratio off the journal ticks instead would divide two harnesses
and land somewhere neither artifact says.

A tick has two costs, and they scale differently: the storage-side **offset capture is
O(writer processes)** (one `stat` per op log, flat in store size, which is what the op
log buys over the 145×-slower full manifest at 50 000 objects), while the **controller
pickle is O(live metadata)** (1.49 MB at 50k objects, the dominant term at scale,
≈3.0–3.6 µs/object on local disk: 0.018 s at 5k to 0.15 s at 50k, `cost`). FSx charges
~0.41 ms per metadata operation above about 20 000 objects, which is why the op log
exists. The full `(inode,size)` manifest stays available (`write_manifest(full=True)`) for
offline audits.

| restore (fsck) | store objects | mode | fsck total | notes |
|---|---|---|---|---|
| | 50 000 (5 % churn) | op-log | 4.4 s (FSx, cluster, `costfsx`) / 0.59 s (local disk, `cost`) | O(churn) log replay + one names-only readdir; zero per-object stats |
| | 50 000 (5 % churn) | full scan | 23.2 s (FSx, cluster, earlier validation) | fallback when logs are unusable |

Restore is dominated by process/controller bring-up, not by reconciliation: 2.7–2.8 s to
re-attach on local disk when Ray is already up, negligible next to model load. The cluster drill splits
the same way: 4.889 s of reconciliation inside a 19.07 s `restore_tool` invocation, the
rest being Ray and TransferQueue bring-up. First `ReplayBufferAsync.sample` after
recovery: **~5 ms** with no rollout workers alive.

## Testing

```bash
# package suite (253 tests, no GPU): backend semantics, crash recovery, integration,
# restore tool, the round-2 hardening regressions (intent logging, tick coherence,
# store identity, per-session promotion, publication crash safety, hostile key names),
# the round-3 regressions (snapshot straddles, identity preflight, publication
# transaction, freshness matrix, cross-host name tokens, per-prompt __rollout_n__
# promotion, custom-sampler kwarg forwarding), the round-4 regressions (oplog-mode
# removal + gap-replacement refusal, strict offsets validation, pre-capture failure,
# zero-write identity preflight and dry-run, transaction/artifact binding with
# slot-reuse faults, fail-closed promotion, prompt-accounting characterization) and
# the round-5 regressions (fail-closed op-log record names incl. path traversal,
# plan-then-execute preflight equivalence, store-lineage-bound transactions, the
# exclusive per-checkpoint restore lock, narrowed force-store-mismatch, strict
# __rollout_n__ scalars) and the round-6 regressions (checkpoint selection contained
# in --ckpt-dir against hostile pointers/symlinks, strict fence.json recovery-mode
# stamp, fail-closed reconcile.state markers with digest-required completion,
# writer-canonical object-name encodings, full-plan dry/real schema equality) and
# the round-7 regressions (store-internal symlinks refused or fenced with outside
# sentinels byte-identical, checkpoint/journal containment below global_step_N,
# invalid-UTF-8 fence/marker artifacts failing closed, the pinned no-preflight
# result schema) and the round-8 regressions (the _store_id lineage authority and
# _fields.d children lstat-gated with VALID external sentinels, marker-source
# provenance before transaction completion — symlink-swapped and escaping sources
# decline with dry == real, the remaining direct readers: pointer file, exported
# load_journal, fence probe, offline audit manifest) and the round-9 regressions
# (marker-source validation made UNCONDITIONAL — empty/NUL/non-string/oversized
# sources classify unprovable with dry == real and byte-identical checkpoints;
# independent, mutation-tested guards for a symlinked identity stamp and a
# symlinked dataloader_state.pt at completion time) and the round-10 regressions
# (the opt-in oplog_fsync barrier landing once per batch before the first publish,
# the restore tool's refusal of a reachable TransferQueue controller in both modes
# with its own private Ray, the split restore gate — a refused journal still heals
# an interrupted swap as healed_only, and any outcome without a live transfer_queue
# dir exits non-zero — the one-writer ring lease with heartbeat takeover, the
# one-tick fence/writer pair refused on mismatch, the controller pickle parsed in
# recover()'s preflight so a corrupt one never reaches load_checkpoint, the
# controller-schema fingerprint pinned against a real snapshot and refused on drift,
# mid-record offsets aligned down instead of aborting, older-slot fallback and
# --journal-slot selection, tick-sequence slot ordering across a writer restart, and
# the trainer-step default plus the unwired-dataloader signal) and the round-11
# regressions (the exclusivity probe running BEFORE any isolation — Ray's own
# discovery pinned against the running cluster, a fresh-interpreter CLI invocation
# refusing the controller it previously could not see and detaching again, the
# private instance asserted for real with a cluster and RAY_ADDRESS both in reach,
# and doubt in discovery counting as presence; a held ring lease degrading both
# wiring sites to journaling-off, with the journaling_enabled metric on the subclass
# and a None return plus one warning on the register wiring, while JournalWriter
# itself still raises; and the schema fingerprint read off the snapshot the tick
# wrote, so a TransferQueue-side rename moves it and is refused, with custom_meta
# parsed like the other four) and the round-12 regressions (a discovered cluster
# address nothing answers at reaching a verdict in milliseconds instead of 820 s,
# the TCP pre-check's three answers, the bounded attach, and the refusal detail
# branching so a doubt case names the pointer to clear instead of reporting a
# controller as reachable) and the round-13 regressions (the probe enumerating every
# locally discoverable Ray instance instead of the one address Ray's resolution
# returns, so a stale RAY_ADDRESS no longer hides a controller running one local
# gcs_server away; each instance detached before the next is joined, the first
# sighting ending the search and naming its cluster, doubt about one instance
# outranking the others that came back clean, and a broken enumeration reading as
# doubt rather than as nothing to probe) and the TransferQueue@main
# compatibility pins (journal-ring selection against TQ#155's .tmp/.old siblings,
# TQ#141 mark_clearing's invisibility to the journal parser, the DROP disposition
# of a mid-clear snapshot and the group-atomic drop when a snapshot lands inside the
# unlink loop, the rollout-skip annotation firing exactly once on the
# sample override, separate_async's wait_for_sampleable / get_sampleable_count
# reaching the journal poll tick, the restore tool's handling of
# transfer_queue.old / .tmp siblings including the publication-swap window, and
# DurableFSManager's zmq_context parameter on both sides of the version probe)
pytest tests/experimental/durable_buffer -q

# Cross-version store compatibility (0.1.9-written store read by 0.1.10.dev0 and the
# reverse) is scripted evidence rather than a suite test, because it needs a second
# interpreter: see tests/experimental/durable_buffer/CROSS_VERSION_NOTES.md.

# verl's OWN V1 suites, but running on the durable backend
pytest tests/trainer/ppo/v1 -q -p tests.experimental.durable_buffer.durable_plugin
#   -> 153 passed, 2 failed (vs 155 passed in-memory)
```

The two failures are `test_save_load_roundtrip_restores_prompts` and
`test_save_load_then_reissue_only_inflight`. Both simulate process death by wiping the
partition and then expect `load_checkpoint` to bring the *data* back. That works only
for a checkpoint that **copies** data (in-memory backends); an external store's
checkpoint **references** data, so wiping the partition deletes what the checkpoint
points at. verl production never calls `clear_partition`, and the package's own tests
simulate death the honest way (`tq.close()` + `tq.init()`, store survives). A
backend-agnostic version of those two tests is a good upstream contribution.

## Limitations / next steps

* **TQ 0.1.9 floor** (`_tq_supports_checkpoint`), already satisfied: verl installs
  TransferQueue from `@main` as a core dependency and gates on a runtime version probe
  rather than a pin. (Upstream's `verl/utils/transferqueue_utils.py`, untouched by this
  package, still carries a stale `pip install TransferQueue==0.1.8` string in an
  ImportError message — the only survivor of the old pin.)
* **Registration must reach every process.** `VERL_USE_EXTERNAL_MODULES` is read at `import verl` time, so it must be set in the driver's environment and forwarded to Ray workers through `runtime_env` (Usage step 2). The `verl.plugins` entry-point group removes even the env var for a backend shipped as its own package.
* **S3 tier** not implemented here. The client contract is small (`put`/`get`/`clear`/
  `clear_by_index`/`write_manifest`); `ETag` replaces the inode. Note the validation
  cluster's pods had no S3 egress, so an S3 tier could only be developed and tested
  off-cluster until that changes.
* **Tick cost at very large buffers** is the controller pickle, O(live metadata)
  (≈1.5 MB @ 50k objects, growing linearly), not the storage fence (O(writers)
  offsets). At ≥10⁵ live objects either stretch the cadence or pursue an incremental
  controller snapshot upstream.
* **Residual windows** (see "Guarantees, exactly"): trajectories — the journal
  cadence; prompts — two bounded one-tick windows (duplicate-dispatch and skip;
  neither detectable package-side); checkpoint-fallback restores — a
  fence window as wide as the controller pickle, inherent to verl's direct
  `tq.save_checkpoint` call order. A controller-side write-ahead log inside TQ would close all of them; that
  belongs upstream.
* **The op log is mandatory.** The former `oplog: false` tick mode is REMOVED:
  it wrote its full manifest from the storage hook, after the controller pickle, so
  a same-name overwrite in that gap was recorded as valid — unsound to recover from.
  The config is rejected at bootstrap, and recovery refuses snapshots recorded with
  it. `write_manifest(full=True)` remains an offline audit tool only.
* **Symlink validation is check-then-use (residual TOCTOU).** Store internals,
  checkpoint layout and journal artifacts are validated lstat-style at every
  read/write/list point (refuse or fence, never follow), but the validation is not
  dirfd/`O_NOFOLLOW`-plumbed, so a node could in principle be swapped between the
  check and the operation. The store and checkpoint live on trusted FSx/NFS paths;
  the threat model these checks close is operator misconfiguration (stray links,
  mis-assembled trees), not an active local attacker racing the process.
* **PROMOTE requires a durably-known group size.** Per-sample `__rollout_n__`
  overrides (tensor / `NonTensorStack` columns) are persisted with the prompt
  payload and honored exactly. Everything else — a batch-level `NonTensorData`
  override (consumed by the agent loop, never persisted) or sync mode's tag-only
  prompts — leaves the true count durably unknowable, and such prompts are NEVER
  promoted unless the operator explicitly asserts a uniform rollout.n
  (`assume_uniform_group_size=True`; restore_tool's `--group-size` carries that
  assertion). Even then, observable evidence (any session index ≥ the assumed `n`)
  disables promotion for that prompt.
* **Results provenance**: the CPU benchmark JSONs/summaries under
  `experiments/durable_buffer/results/` (qwen27b / agentic / cadence / cost) are
  regenerated by the harnesses in this tree — the recovery benchmark's journal arms
  drive the real `JournalWriter` (offset pre-capture, tick locking, ring publication)
  and kill-state metrics are captured before reconciliation — and every artifact
  embeds its provenance: commit, `git describe --dirty`, a sha256 content digest of
  the harness code itself (`experiments/durable_buffer/*.py` — identifies the exact
  harness even on a dirty tree), python/torch/TQ versions, the full invocation, and
  a timestamp.
  Kill-drill, FSx-Lustre and NVMe numbers are cluster measurements from this design's
  earlier internal validation runs, not reproducible from this tree, and are labeled
  as such where cited.
* **Lifecycle decoupling** (buffer outliving the job) additionally needs
  `lifetime="detached"` TQ actors and a shared Ray cluster. The durable store already
  makes the buffer *state* outlive the job, which covers most of the value.
