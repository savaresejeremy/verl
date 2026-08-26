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
"""Crash-recovery benchmark for verl's V1 replay buffer.

Answers, with numbers, the two questions that decide whether the durable buffer is
worth shipping:

  1. **What does a crash cost today, and what does it cost with durability?**
     Metric: finished-but-untrained trajectories that must be regenerated after the
     restart, converted to trajectory-generation-seconds with the per-trajectory generation cost that the preset models.
  2. **What does durability cost while nothing is failing?**
     Metric: producer-side put throughput/latency versus the stock in-memory backend,
     plus the journal tick cost.

Arms (each runs in its own subprocess because a process hosts one TQ controller):

  ``none``        stock in-memory backend, no TQ checkpoint            (today's default)
  ``ckpt``        durable store, journal only at trainer checkpoints   (TQ 0.1.9 as-is)
  ``journal``     durable store, journal every ``--journal-interval``  (this package)
  ``journal_dl``  ``journal`` + journaled dataloader state             (v1.1: zero dup
                  dispatches UNDER THIS DETERMINISTIC SCHEDULE — kills land between
                  dispatches, so both documented prompt-accounting windows are empty
                  here; real workloads keep the two bounded one-tick windows)

Usage::

    python experiments/durable_buffer/bench_recovery.py --all --out results/
    python experiments/durable_buffer/bench_recovery.py --arm journal --kill-at 7

The workload is a deterministic model of verl's V1 async loop (prompt dispatch ->
trajectories written as produced -> trainer consumes a batch per step), driven by a
virtual clock so runs are reproducible and fast; the TransferQueue, the storage
backends, the journal, fsck and ``ReplayBufferAsync`` are all real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

ARMS = ("none", "ckpt", "journal", "journal_dl")
DEFAULT_OUT = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------- workload
class Workload:
    """Deterministic model of a verl V1 async training loop."""

    def __init__(self, args):
        self.groups_per_step = args.groups_per_step
        self.group_size = args.group_size
        self.resp_len = args.resp_len
        self.steps = args.steps
        self.kill_at = args.kill_at
        self.ckpt_every = args.ckpt_every
        self.journal_interval = args.journal_interval
        self.step_seconds = args.step_seconds  # virtual wall-clock per training step
        self.gen_seconds_per_traj = args.gen_seconds_per_traj
        self.inflight_per_step = args.inflight_per_step


def fields_for(n: int, value: int, resp_len: int) -> TensorDict:
    return TensorDict(
        {
            "response_ids": torch.full((n, resp_len), value % 1000, dtype=torch.long),
            "rewards": torch.full((n,), float(value % 7)),
        },
        batch_size=[n],
    )


def _provenance() -> dict:
    """Embedded in every artifact: exactly what produced these numbers.

    ``commit`` alone cannot identify a dirty tree, so ``git_describe`` carries
    ``--dirty`` and ``harness_digest_sha256`` fingerprints the actual harness code
    (every ``*.py`` under ``experiments/durable_buffer``) independently of git state.
    """
    import platform

    import transfer_queue

    repo = Path(__file__).resolve().parents[2]
    harness_dir = Path(__file__).resolve().parent

    def _git(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            out = ""
        return out or "unknown"

    hasher = hashlib.sha256()
    for path in sorted(harness_dir.glob("*.py")):
        hasher.update(path.name.encode())
        hasher.update(hashlib.sha256(path.read_bytes()).digest())
    try:
        dirty_raw = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:  # noqa: BLE001
        dirty_raw = ""
    dirty_paths = [line[3:] for line in dirty_raw.splitlines() if line.strip()]
    return {
        "commit": _git("rev-parse", "HEAD"),
        "git_describe": _git("describe", "--always", "--dirty", "--abbrev=12"),
        # a '-dirty' describe is self-diagnosing: these are the paths that differed
        # from HEAD at generation time (a multi-artifact run dirties its OWN results
        # dir as it goes; the harness digest below pins the actual harness code)
        "dirty_paths": dirty_paths[:40] + ([f"... {len(dirty_paths) - 40} more"] if len(dirty_paths) > 40 else []),
        "harness_digest_sha256": hasher.hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transfer_queue": getattr(transfer_queue, "__version__", "unknown"),
        "invocation": " ".join(sys.argv),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_arm(args) -> dict:
    """Produce -> crash -> recover for one arm; returns the metric dict."""
    import transfer_queue as tq

    wl = Workload(args)
    store = Path(args.store)
    journal_root = Path(args.journal_root)
    ckpt_state = Path(args.journal_root) / "trainer_data.pt"
    for p in (store, journal_root):
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)

    durable = args.arm != "none"
    if durable:
        import verl.experimental.durable_buffer.register  # noqa: F401
        from verl.experimental.durable_buffer.fs_kv import STATS, backend_config

        conf = backend_config(store)
    else:
        STATS = {}
        conf = None
    tq.init(conf)

    partition = "train"
    virtual_t = 0.0
    last_journal_t = -1e9
    produced: dict[str, int] = {}  # uid -> step produced
    trained: set[str] = set()
    inflight: set[str] = set()
    put_latencies: list[float] = []
    journal_ticks: list[float] = []
    journal_dir = None  # the ckpt arm's last trainer-checkpoint snapshot
    current_step = {"step": 0}

    # The journal arms drive the REAL JournalWriter (deterministic, virtual-clock
    # triggered ticks) so the benchmark exercises offset pre-capture, tick locking
    # and ring publication — not a hand-rolled approximation of them. The ckpt arm
    # keeps verl's own direct tq.save_checkpoint call: that IS what a trainer
    # checkpoint does (and it is the arm's whole point).
    jw = None
    if args.arm in ("journal", "journal_dl"):
        from verl.experimental.durable_buffer.journal import JournalWriter

        jw = JournalWriter(
            journal_root,
            interval_s=0.0,
            ring=3,
            dataloader_fn=(lambda: {"next_group": len(produced)}) if args.arm == "journal_dl" else None,
            metadata_fn=lambda: {"global_steps": current_step["step"]},
        )

    def journal_tick(step: int) -> None:
        nonlocal journal_dir
        current_step["step"] = step
        if jw is not None:
            info = jw.tick()
            if info is None:
                raise SystemExit(f"arm {args.arm}: journal tick unexpectedly skipped at step {step}")
            journal_ticks.append(info["tick_s"])
            return
        # ckpt arm: verl-stock trainer-checkpoint snapshot (direct TQ call)
        slot = len(journal_ticks) % 3
        target = journal_root / f"journal-{slot:03d}"
        t0 = time.perf_counter()
        tq.save_checkpoint(str(target), metadata={"global_steps": step, "fence_ts": time.time()})
        journal_ticks.append(time.perf_counter() - t0)
        journal_dir = target

    for step in range(1, wl.steps + 1):
        # ---- dispatch prompts + write trajectories as they finish.
        # The virtual clock advances *within* the step so that a journal cadence finer
        # than a training step is modelled honestly (2s journal vs a 100s step).
        per_group_seconds = wl.step_seconds / max(wl.groups_per_step, 1)
        for g in range(wl.groups_per_step):
            # production shape: an underscore-free uid (verl uses uuid4) and trajectory
            # keys {uid}_{session}_{index} — fsck/promotion parse exactly this grammar.
            uid = f"g{step:03d}x{g:02d}"
            tq.kv_put(
                key=uid,
                partition_id=partition,
                tag={"is_prompt": True, "status": "running", "global_steps": step},
            )
            if g < wl.groups_per_step - wl.inflight_per_step:
                keys = [f"{uid}_{s}_0" for s in range(wl.group_size)]
                t0 = time.perf_counter()
                tq.kv_batch_put(
                    keys=keys,
                    partition_id=partition,
                    fields=fields_for(len(keys), step * 10 + g, wl.resp_len),
                    tags=[{"status": "success", "global_steps": step}] * len(keys),
                )
                put_latencies.append((time.perf_counter() - t0) / len(keys))
                tq.kv_put(
                    key=uid,
                    partition_id=partition,
                    tag={"is_prompt": True, "status": "finished", "global_steps": step},
                )
                produced[uid] = step
            else:
                inflight.add(uid)
            virtual_t += per_group_seconds
            if args.arm in ("journal", "journal_dl") and virtual_t - last_journal_t >= wl.journal_interval:
                journal_tick(step)
                last_journal_t = virtual_t

        # ---- trainer consumes one batch of the oldest finished groups
        ready = [u for u in sorted(produced) if u not in trained]
        for uid in ready[: args.train_groups_per_step]:
            tq.kv_clear(
                keys=[uid] + [f"{uid}_{s}_0" for s in range(wl.group_size)],
                partition_id=partition,
            )
            trained.add(uid)

        # ---- journal policy: the "ckpt" arm only snapshots at trainer checkpoints
        if args.arm == "ckpt" and step % wl.ckpt_every == 0:
            journal_tick(step)
            last_journal_t = virtual_t
        if step % wl.ckpt_every == 0:
            torch.save({"next_group": len(produced)}, ckpt_state)  # trainer checkpoint (weights+loader)

        if step == wl.kill_at:
            break

    produced_untrained = {u for u in produced if u not in trained}
    has_snapshot = journal_dir is not None or (jw is not None and journal_ticks)
    time_since_journal = virtual_t - last_journal_t if has_snapshot else float("inf")
    # kill-state metrics: measured AT the kill, before recovery fences anything
    store_objects_at_kill = len(list(store.glob("*.bin"))) if durable else 0
    store_bytes_at_kill = sum(p.stat().st_size for p in store.glob("*.bin")) if durable else 0

    # ------------------------------------------------------------------ CRASH
    tq.close()
    t_restart0 = time.perf_counter()
    tq.init(conf)
    t_init = time.perf_counter() - t_restart0

    recovery_dir = None
    if durable:
        if jw is not None:
            from verl.experimental.durable_buffer.journal import latest_journal

            recovery_dir = latest_journal(journal_root)  # newest complete ring slot
        else:
            recovery_dir = journal_dir

    recovery: dict = {}
    if durable and recovery_dir is not None:
        from verl.experimental.durable_buffer import recover

        t0 = time.perf_counter()
        # assume_uniform_group_size mirrors restore_tool's --group-size semantics: the
        # benchmark's workload is uniform rollout.n by construction, so the operator
        # assertion is true here. (This benchmark's prompts are tag-only; no per-prompt
        # __rollout_n__ is persisted, exactly like verl sync mode.)
        recovery = recover(tq, store, recovery_dir, partition, group_size=wl.group_size, assume_uniform_group_size=True)
        t_recover = time.perf_counter() - t0
    else:
        t_recover = 0.0  # nothing to recover: the in-memory buffer died with the actors

    live = tq.kv_list().get(partition, {})

    def _traj_uid(key: str) -> str | None:
        parts = key.rsplit("_", 2)
        if len(parts) != 3:
            return None
        try:
            int(parts[1]), int(parts[2])
        except ValueError:
            return None
        return parts[0]

    survivor_groups = {u for u in map(_traj_uid, live) if u is not None}
    survivors = survivor_groups & produced_untrained
    lost = produced_untrained - survivors
    reissued = {k for k, tag in live.items() if tag.get("is_prompt") and tag.get("status") in ("pending", "running")}

    # duplicate accounting: the restored trainer loader position comes from the trainer
    # checkpoint (arms none/ckpt/journal) or from the journal (arm journal_dl)
    state_path = None
    if args.arm == "journal_dl" and recovery_dir is not None and (recovery_dir / "dataloader_state.pt").exists():
        state_path = recovery_dir / "dataloader_state.pt"
    elif ckpt_state.exists():
        state_path = ckpt_state
    loader_pos = torch.load(state_path, weights_only=False)["next_group"] if state_path else 0
    produced_order = [u for u, _ in sorted(produced.items(), key=lambda kv: kv[0])]
    will_reserve = set(produced_order[loader_pos:])
    duplicates = (survivors | trained) & will_reserve

    # first sample after recovery: the recovery only counts if the buffer immediately
    # serves TRAINABLE data — a nonempty batch of the expected groups whose bytes
    # decode back to what the pre-crash producer wrote.
    t_sample = None
    if survivors:
        from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

        rb = ReplayBufferAsync(
            trainer_mode="separate_async",
            trainer_config=OmegaConf.create({}),
            max_off_policy_threshold=10**6,
            max_off_policy_strategy="drop",
            sampler_kwargs=OmegaConf.create({}),
            poll_interval=0.05,
            refill_fn=None,
        )
        want_groups = min(len(survivors), 8)
        t0 = time.perf_counter()
        batch, _metrics = rb.sample(global_steps=wl.kill_at + 1, partition_id=partition, batch_size=want_groups)
        t_sample = time.perf_counter() - t0
        if batch is None or not getattr(batch, "keys", None):
            raise SystemExit(f"arm {args.arm}: recovered buffer served an EMPTY batch ({len(survivors)} survivors)")
        sampled_groups = {u for u in map(_traj_uid, batch.keys) if u is not None}
        if len(sampled_groups) != want_groups or not sampled_groups <= survivors:
            raise SystemExit(
                f"arm {args.arm}: sampled groups {sorted(sampled_groups)} do not match "
                f"the expected {want_groups} recovered groups"
            )
        # byte validation: rewards encode (step*10 + g) % 7 by construction
        check = tq.kv_batch_get(keys=list(batch.keys), partition_id=partition, select_fields=["rewards"])
        rewards = check["rewards"].reshape(-1).tolist()
        for key, reward in zip(batch.keys, rewards, strict=True):
            uid = _traj_uid(key)
            step_of, g_of = int(uid[1:4]), int(uid[5:7])  # g{step:03d}x{g:02d}
            expected = float((step_of * 10 + g_of) % 7)
            if float(reward) != expected:
                raise SystemExit(f"arm {args.arm}: recovered bytes wrong for {key}: {reward} != {expected}")

    traj_per_group = wl.group_size
    result = {
        "arm": args.arm,
        "steps_run": min(wl.kill_at, wl.steps),
        "kill_at_step": wl.kill_at,
        "groups_produced": len(produced),
        "groups_trained": len(trained),
        "groups_untrained_at_kill": len(produced_untrained),
        "groups_recovered": len(survivors),
        "groups_lost": len(lost),
        "trajectories_lost": len(lost) * traj_per_group,
        "generation_seconds_wasted": round(len(lost) * traj_per_group * wl.gen_seconds_per_traj, 1),
        "inflight_reissued": len(reissued),
        "duplicate_groups": len(duplicates),
        "duplicate_trajectories": len(duplicates) * traj_per_group,
        "virtual_seconds_since_journal_at_kill": (
            None if time_since_journal == float("inf") else round(time_since_journal, 1)
        ),
        "rto_init_s": round(t_init, 3),
        "rto_recover_s": round(t_recover, 3),
        "rto_first_sample_s": None if t_sample is None else round(t_sample, 3),
        "put_latency_ms_mean": round(statistics.mean(put_latencies) * 1e3, 3) if put_latencies else None,
        "put_latency_ms_p95": round(sorted(put_latencies)[int(0.95 * len(put_latencies)) - 1] * 1e3, 3)
        if len(put_latencies) > 20
        else None,
        "journal_ticks": len(journal_ticks),
        "journal_tick_ms_mean": round(statistics.mean(journal_ticks) * 1e3, 2) if journal_ticks else None,
        "store_objects_at_kill": store_objects_at_kill,
        "store_bytes_at_kill": store_bytes_at_kill,
        "backend_stats": {k: v for k, v in (STATS or {}).items()},
        "fsck": {k: v for k, v in recovery.items() if k not in ("promotion",)},
        "promotion": recovery.get("promotion"),
    }
    tq.close()
    return result


PRESETS = {
    # name: (groups_per_step, group_size, resp_len, train_groups_per_step, inflight_per_step,
    #        step_seconds, gen_seconds_per_traj, ckpt_every, journal_interval)
    # A small dense model on one node: short generations, frequent steps.
    "small": (8, 8, 512, 4, 2, 20.0, 2.0, 10, 2.0),
    # Shaped like our measured 2-node Qwen3.5-27B disaggregated run: ~100s steps,
    # 25s per trajectory of generation, rollout.n=8, checkpoint every 5 steps.
    "qwen27b": (8, 8, 512, 4, 2, 100.0, 25.0, 5, 2.0),
    # Long agentic rollouts: multi-minute trajectories make every lost rollout
    # expensive and restarts are frequent at fleet scale.
    "agentic": (6, 8, 1024, 3, 2, 600.0, 300.0, 5, 2.0),
}


def apply_preset(args) -> None:
    if not args.preset:
        return
    (
        args.groups_per_step,
        args.group_size,
        args.resp_len,
        args.train_groups_per_step,
        args.inflight_per_step,
        args.step_seconds,
        args.gen_seconds_per_traj,
        args.ckpt_every,
        args.journal_interval,
    ) = PRESETS[args.preset]


# ------------------------------------------------------------------------ driver
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--all", action="store_true", help="run every arm in a subprocess and print a table")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--kill-at", type=int, default=10)
    p.add_argument("--groups-per-step", type=int, default=8)
    p.add_argument("--inflight-per-step", type=int, default=2, help="groups still generating when the step ends")
    p.add_argument("--train-groups-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8, help="rollout.n")
    p.add_argument("--resp-len", type=int, default=512)
    p.add_argument("--ckpt-every", type=int, default=5, help="trainer.save_freq (steps)")
    p.add_argument("--journal-interval", type=float, default=2.0, help="seconds of *virtual* training time")
    p.add_argument("--step-seconds", type=float, default=100.0, help="virtual wall-clock per training step")
    p.add_argument("--gen-seconds-per-traj", type=float, default=25.0, help="measured generation cost per trajectory")
    p.add_argument("--store", default="/tmp/db_bench_store")
    p.add_argument("--journal-root", default="/tmp/db_bench_journal")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--preset", choices=sorted(PRESETS), help="workload shape (overrides the knobs it owns)")
    p.add_argument("--sweep-kill", default="", help="comma-separated kill steps; runs every arm at each")
    p.add_argument("--sweep-journal", default="", help="comma-separated journal intervals (s) to compare")
    p.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    args = p.parse_args(argv)
    apply_preset(args)
    return args


def _fmt_table(rows: list[dict], with_kill: bool = False, with_interval: bool = False) -> str:
    cols = [
        ("arm", "arm"),
        ("groups_untrained_at_kill", "untrained@kill"),
        ("groups_recovered", "recovered"),
        ("groups_lost", "lost"),
        ("trajectories_lost", "traj lost"),
        ("generation_seconds_wasted", "gen-s wasted"),
        ("duplicate_trajectories", "dup traj"),
        ("rto_recover_s", "recover s"),
        ("rto_first_sample_s", "1st sample s"),
        ("put_latency_ms_mean", "put ms"),
        ("journal_tick_ms_mean", "tick ms"),
    ]
    if with_kill:
        cols.insert(1, ("kill_at_step", "kill@step"))
    if with_interval:
        cols.insert(1, ("journal_interval_s", "journal s"))
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k)) for k, _ in cols) + " |")
    return "\n".join(out)


def _run_subprocess_arm(args, arm: str, kill_at: int, outdir: Path, tag: str) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm, "--tag", tag]
    skip = {"arm", "all", "tag", "out", "preset", "sweep_kill", "kill_at"}
    for key, value in vars(args).items():
        if key in skip:
            continue
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    cmd += ["--kill-at", str(kill_at), "--out", str(outdir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    path = outdir / f"{tag}_{arm}.json"
    if proc.returncode != 0 or not path.exists():
        print(proc.stdout[-2000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"arm {arm} (kill@{kill_at}) failed with rc={proc.returncode}")
    return json.loads(path.read_text())["result"]


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.arm and not args.all:
        result = run_arm(args)
        payload = {"config": vars(args), "provenance": _provenance(), "result": result}
        path = outdir / f"{args.tag}_{args.arm}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str))
        return 0

    rows = []
    kills = [int(k) for k in args.sweep_kill.split(",") if k.strip()] or [args.kill_at]
    intervals = [float(x) for x in args.sweep_journal.split(",") if x.strip()]
    if intervals:
        # Loss-window curve: how much rollout does the journal cadence protect?
        for kill_at in kills:
            for arm in ("none", "ckpt"):
                tag = f"{args.tag}_k{kill_at}_ref"
                print(f"\n=== arm {arm} (kill@{kill_at}, reference) ===", flush=True)
                r = _run_subprocess_arm(args, arm, kill_at, outdir, tag)
                r["journal_interval_s"] = None
                rows.append(r)
            for interval in intervals:
                args.journal_interval = interval
                for arm in ("journal", "journal_dl"):
                    tag = f"{args.tag}_k{kill_at}_j{interval:g}"
                    print(f"\n=== arm {arm} (kill@{kill_at}, journal every {interval:g}s) ===", flush=True)
                    r = _run_subprocess_arm(args, arm, kill_at, outdir, tag)
                    r["journal_interval_s"] = interval
                    rows.append(r)
    else:
        for kill_at in kills:
            for arm in ARMS:
                tag = f"{args.tag}_k{kill_at}"
                print(f"\n=== arm {arm} (kill@step {kill_at}) ===", flush=True)
                result = _run_subprocess_arm(args, arm, kill_at, outdir, tag)
                rows.append(result)
                print(
                    f"lost={result['trajectories_lost']} traj, "
                    f"gen-s wasted={result['generation_seconds_wasted']}, dup={result['duplicate_trajectories']}"
                )

    table = _fmt_table(rows, with_kill=len(kills) > 1, with_interval=bool(intervals))
    summary = outdir / f"{args.tag}_summary.md"
    prov = _provenance()
    saved = {}
    for kill_at in kills:
        per = {r["arm"]: r for r in rows if r["kill_at_step"] == kill_at}
        if "none" in per and "journal" in per:
            saved[kill_at] = per["none"]["generation_seconds_wasted"] - per["journal"]["generation_seconds_wasted"]
    header = (
        f"# Durable replay buffer — crash-recovery benchmark ({args.tag})\n\n"
        f"provenance: commit `{prov['commit']}` (`{prov['git_describe']}`) | harness sha256 "
        f"`{prov['harness_digest_sha256'][:16]}` | python {prov['python']} | torch {prov['torch']} | "
        f"transfer_queue {prov['transfer_queue']} | {prov['generated_at']}\n"
        f"invocation: `{prov['invocation']}`\n\n"
        f"preset: `{args.preset or 'custom'}` | {args.groups_per_step} groups/step "
        f"(rollout.n={args.group_size}, {args.inflight_per_step} still generating at step end), "
        f"{args.train_groups_per_step} groups trained/step, {args.step_seconds:.0f}s/step, "
        f"trainer.save_freq={args.ckpt_every} steps, journal every {args.journal_interval}s, "
        f"generation {args.gen_seconds_per_traj}s/trajectory\n\n"
        f"arms: `none` = stock in-memory buffer · `ckpt` = durable store journaled only at "
        f"trainer checkpoints · `journal` = durable store + {args.journal_interval}s journal "
        f"(real JournalWriter ticks) · `journal_dl` = journal + dataloader state\n\n"
    )
    body = table + "\n"
    if saved:
        body += (
            "\nTrajectory-generation-seconds recovered by the journal arm vs the stock buffer: "
            + ", ".join(f"kill@{k}: **{v:.0f}s**" for k, v in sorted(saved.items()))
            + "\n"
        )
    summary.write_text(header + body)
    print("\n" + table)
    if saved:
        print(
            "Generation-seconds saved (journal vs none): "
            + ", ".join(f"kill@{k}={v:.0f}" for k, v in sorted(saved.items()))
        )
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
