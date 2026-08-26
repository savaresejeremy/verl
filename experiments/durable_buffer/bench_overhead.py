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
"""What durability *costs*: producer throughput, journal ticks, and fsck at scale.

Three measurements, each in its own subprocess (a process hosts one TQ controller):

``produce``  write N trajectories through TQ and report throughput/latency, for the
             stock in-memory backend and for the durable backend. The interesting
             number is the delta: rollout generation takes seconds per trajectory, so
             any per-put cost below a millisecond is free.
``journal``  metadata-only snapshot cost as a function of live buffer size, which sets
             the affordable journal cadence (and hence the loss window).
``fsck``     restore-time reconciliation cost as a function of store size, plus the
             components (list / fence / drop).

Usage::

    python experiments/durable_buffer/bench_overhead.py --all
    python experiments/durable_buffer/bench_overhead.py --phase journal --objects 100000
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
from tensordict import TensorDict

DEFAULT_OUT = Path(__file__).resolve().parent / "results"
BATCH = 512


def _provenance() -> dict:
    """Embedded in every artifact: exactly what produced these numbers.

    ``commit`` alone cannot identify a dirty tree, so ``git_describe`` carries
    ``--dirty`` and ``harness_digest_sha256`` fingerprints the actual harness code
    (every ``*.py`` under ``experiments/durable_buffer``) independently of git state.
    """
    import platform

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
        import transfer_queue

        tq_version = getattr(transfer_queue, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        tq_version = "unknown"
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
        "transfer_queue": tq_version,
        "invocation": " ".join(sys.argv),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _fields(n: int, resp_len: int, value: int = 1) -> TensorDict:
    return TensorDict(
        {
            "response_ids": torch.full((n, resp_len), value, dtype=torch.long),
            "rewards": torch.full((n,), float(value)),
        },
        batch_size=[n],
    )


def _init(backend: str, store: Path):
    import transfer_queue as tq

    if backend == "durable":
        import verl.experimental.durable_buffer.register  # noqa: F401
        from verl.experimental.durable_buffer.fs_kv import backend_config

        shutil.rmtree(store, ignore_errors=True)
        tq.init(backend_config(store))
    else:
        tq.init(None)
    return tq


def phase_produce(args) -> dict:
    """Producer-side cost: trajectories/s and per-trajectory put latency."""
    tq = _init(args.backend, Path(args.store))
    partition = "bench"
    n_traj = args.trajectories
    keys = [f"t{i}" for i in range(n_traj)]
    latencies = []
    t_start = time.perf_counter()
    for off in range(0, n_traj, BATCH):
        chunk = keys[off : off + BATCH]
        t0 = time.perf_counter()
        tq.kv_batch_put(
            keys=chunk,
            partition_id=partition,
            fields=_fields(len(chunk), args.resp_len),
            tags=[{"status": "success", "global_steps": 1}] * len(chunk),
        )
        latencies.append((time.perf_counter() - t0) / len(chunk))
    total = time.perf_counter() - t_start

    # read-back of one batch, the shape verl's DAPO/staleness paths use
    t0 = time.perf_counter()
    tq.kv_batch_get(keys=keys[:BATCH], partition_id=partition, select_fields=["rewards"])
    t_read_rewards = time.perf_counter() - t0

    store_objects = len(list(Path(args.store).glob("*.bin"))) if args.backend == "durable" else 0
    store_bytes = sum(p.stat().st_size for p in Path(args.store).glob("*.bin")) if args.backend == "durable" else 0
    out = {
        "phase": "produce",
        "backend": args.backend,
        "trajectories": n_traj,
        "resp_len": args.resp_len,
        "seconds": round(total, 3),
        "traj_per_s": round(n_traj / total, 1),
        "put_ms_per_traj_mean": round(statistics.mean(latencies) * 1e3, 3),
        "read_rewards_batch_ms": round(t_read_rewards * 1e3, 2),
        "store_objects": store_objects,
        "store_MB": round(store_bytes / 1e6, 1),
    }
    tq.close()
    return out


def phase_journal(args) -> dict:
    """Journal-tick cost vs live buffer size (the knob that sets the loss window)."""
    from verl.experimental.durable_buffer.journal import JournalWriter

    tq = _init("durable", Path(args.store))
    partition = "bench"
    per_traj_objects = 2
    n_traj = max(args.objects // per_traj_objects, 1)
    keys = [f"t{i}" for i in range(n_traj)]
    t0 = time.perf_counter()
    for off in range(0, n_traj, BATCH):
        chunk = keys[off : off + BATCH]
        tq.kv_batch_put(
            keys=chunk,
            partition_id=partition,
            fields=_fields(len(chunk), args.resp_len),
            tags=[{"status": "success", "global_steps": 1}] * len(chunk),
        )
    seed_s = time.perf_counter() - t0

    journal_root = Path(args.journal_root)
    shutil.rmtree(journal_root, ignore_errors=True)
    jw = JournalWriter(journal_root, interval_s=0.0, ring=3)
    ticks = [jw.tick()["tick_s"] for _ in range(args.repeat)]
    latest = sorted(journal_root.glob("journal-*"))[-1]
    pkl_bytes = (latest / "controller_state.pkl").stat().st_size
    manifest = latest / "store_manifest.tsv"
    out = {
        "phase": "journal",
        "objects": len(list(Path(args.store).glob("*.bin"))),
        "seed_traj_per_s": round(n_traj / seed_s, 1),
        "tick_s_mean": round(statistics.mean(ticks), 4),
        "tick_s_min": round(min(ticks), 4),
        "controller_pkl_MB": round(pkl_bytes / 1e6, 2),
        "manifest_MB": round(manifest.stat().st_size / 1e6, 2) if manifest.exists() else None,
        "us_per_object": round(statistics.mean(ticks) / max(len(list(Path(args.store).glob("*.bin"))), 1) * 1e6, 2),
    }
    tq.close()
    return out


def phase_fsck(args) -> dict:
    """Restore cost: load journal + reconcile a store of a given size."""
    from verl.experimental.durable_buffer import recover
    from verl.experimental.durable_buffer.journal import JournalWriter

    tq = _init("durable", Path(args.store))
    partition = "bench"
    per_traj_objects = 2
    n_traj = max(args.objects // per_traj_objects, 1)
    group = 4
    n_groups = max(n_traj // group, 1)
    for off in range(0, n_groups, BATCH // group):
        uids = [f"g{i}" for i in range(off, min(off + BATCH // group, n_groups))]
        tq.kv_batch_put(
            keys=uids,
            partition_id=partition,
            tags=[{"is_prompt": True, "status": "finished", "global_steps": 1}] * len(uids),
        )
        keys = [f"{u}_{s}_0" for u in uids for s in range(group)]
        tq.kv_batch_put(
            keys=keys,
            partition_id=partition,
            fields=_fields(len(keys), args.resp_len),
            tags=[{"status": "success", "global_steps": 1}] * len(keys),
        )

    journal_root = Path(args.journal_root)
    shutil.rmtree(journal_root, ignore_errors=True)
    jw = JournalWriter(journal_root, interval_s=0.0, ring=2)
    tick = jw.tick()
    time.sleep(0.05)

    # post-journal churn: some groups produced, some consumed -> real work for fsck
    churn = max(n_groups // 20, 1)
    post = [f"p{i}" for i in range(churn)]
    tq.kv_batch_put(
        keys=[f"{u}_{s}_0" for u in post for s in range(group)],
        partition_id=partition,
        fields=_fields(len(post) * group, args.resp_len, value=9),
        tags=[{"status": "success", "global_steps": 2}] * (len(post) * group),
    )
    consumed = [f"g{i}" for i in range(churn)]
    tq.kv_clear(
        keys=consumed + [f"{u}_{s}_0" for u in consumed for s in range(group)],
        partition_id=partition,
    )

    # crash + restore
    tq.close()
    t0 = time.perf_counter()
    tq = _init_existing_store(args)
    t_init = time.perf_counter() - t0
    # assume_uniform_group_size mirrors restore_tool's --group-size assertion; this
    # workload is uniform-n by construction (tag-only prompts, like verl sync mode).
    report = recover(tq, args.store, tick["dir"], partition, group_size=group, assume_uniform_group_size=True)
    churn = report["fenced_objects"] + report["dropped_samples"]
    out = {
        "phase": "fsck",
        "mode": report["mode"],
        "store_objects": args.objects,
        "objects_listed": report["objects_listed"],
        "rto_init_s": round(t_init, 2),
        "load_journal_s": report["t_load_checkpoint_s"],
        "fsck_total_s": report["t_total_s"],
        "fsck_list_s": report["t_list_s"],
        "fsck_fence_s": report["t_fence_s"],
        "fsck_drop_s": report["t_drop_s"],
        "fenced_objects": report["fenced_objects"],
        "dropped_samples": report["dropped_samples"],
        "ms_per_churned_record": round(report["t_total_s"] / max(churn, 1) * 1e3, 3),
    }
    tq.close()
    return out


def _init_existing_store(args):
    """Re-init TQ against an EXISTING durable store (no wipe)."""
    import transfer_queue as tq

    import verl.experimental.durable_buffer.register  # noqa: F401
    from verl.experimental.durable_buffer.fs_kv import backend_config

    tq.init(backend_config(args.store))
    return tq


PHASES = {"produce": phase_produce, "journal": phase_journal, "fsck": phase_fsck}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=sorted(PHASES))
    p.add_argument("--all", action="store_true")
    p.add_argument("--backend", choices=("durable", "memory"), default="durable")
    p.add_argument("--trajectories", type=int, default=4000)
    p.add_argument("--objects", type=int, default=20000)
    p.add_argument("--resp-len", type=int, default=512)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--store", default="/tmp/db_overhead_store")
    p.add_argument("--journal-root", default="/tmp/db_overhead_journal")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    return p.parse_args(argv)


def _sub(args, extra: list[str], tag: str) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--tag", tag, "--out", args.out] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    path = Path(args.out) / f"{tag}.json"
    if proc.returncode != 0 or not path.exists():
        print(proc.stdout[-1500:])
        print(proc.stderr[-2500:], file=sys.stderr)
        raise SystemExit(f"phase {extra} failed rc={proc.returncode}")
    return json.loads(path.read_text())


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.phase and not args.all:
        result = PHASES[args.phase](args)
        result["provenance"] = _provenance()
        (outdir / f"{args.tag}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    prov = _provenance()
    lines: list[str] = [
        f"# Durable replay buffer — cost of durability ({args.tag})\n",
        f"provenance: commit `{prov['commit']}` (`{prov['git_describe']}`) | harness sha256 "
        f"`{prov['harness_digest_sha256'][:16]}` | python {prov['python']} | torch {prov['torch']} | "
        f"transfer_queue {prov['transfer_queue']} | {prov['generated_at']}",
        f"invocation: `{prov['invocation']}`\n",
    ]

    # 1. producer cost, durable vs in-memory
    prod_rows = []
    for backend in ("memory", "durable"):
        for resp in (512, 2048):
            r = _sub(
                args,
                [
                    "--phase",
                    "produce",
                    "--backend",
                    backend,
                    "--trajectories",
                    str(args.trajectories),
                    "--resp-len",
                    str(resp),
                    "--store",
                    args.store,
                ],
                f"{args.tag}_produce_{backend}_{resp}",
            )
            prod_rows.append(r)
            print(f"produce {backend} resp={resp}: {r['traj_per_s']} traj/s, {r['put_ms_per_traj_mean']} ms/traj")
    lines.append("## Producer path (write-as-produced)\n")
    lines.append(
        "| backend | resp_len | trajectories | traj/s | ms per trajectory | read rewards batch (ms) | store MB |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in prod_rows:
        lines.append(
            f"| {r['backend']} | {r['resp_len']} | {r['trajectories']} | {r['traj_per_s']} | "
            f"{r['put_ms_per_traj_mean']} | {r['read_rewards_batch_ms']} | {r['store_MB']} |"
        )

    # 2. journal cost vs buffer size
    j_rows = []
    for objects in (5000, 20000, 50000):
        r = _sub(
            args,
            [
                "--phase",
                "journal",
                "--objects",
                str(objects),
                "--resp-len",
                "256",
                "--store",
                args.store,
                "--journal-root",
                args.journal_root,
                "--repeat",
                "3",
            ],
            f"{args.tag}_journal_{objects}",
        )
        j_rows.append(r)
        print(f"journal {objects} objects: tick {r['tick_s_mean']}s ({r['us_per_object']} us/object)")
    lines.append("\n## Journal tick (metadata-only snapshot + fence manifest)\n")
    lines.append(
        "| live objects | tick s (mean) | controller pkl MB | manifest MB | us/object "
        "| affordable cadence @5% overhead |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in j_rows:
        lines.append(
            f"| {r['objects']} | {r['tick_s_mean']} | {r['controller_pkl_MB']} | {r['manifest_MB']} | "
            f"{r['us_per_object']} | {round(r['tick_s_mean'] / 0.05, 1)}s |"
        )

    # 3. fsck cost vs store size
    f_rows = []
    for objects in (5000, 20000, 50000):
        r = _sub(
            args,
            [
                "--phase",
                "fsck",
                "--objects",
                str(objects),
                "--resp-len",
                "256",
                "--store",
                args.store,
                "--journal-root",
                args.journal_root,
            ],
            f"{args.tag}_fsck_{objects}",
        )
        f_rows.append(r)
        print(
            f"fsck {objects} objects [{r['mode']}]: {r['fsck_total_s']}s "
            f"({r['ms_per_churned_record']} ms/churned record)"
        )
    lines.append("\n## Restore (fsck) cost\n")
    lines.append(
        "| store objects | mode | load journal s | fsck total s | list | fence | drop "
        "| fenced | dropped | ms/churned record |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in f_rows:
        lines.append(
            f"| {r['store_objects']} | {r['mode']} | {r['load_journal_s']} | {r['fsck_total_s']} "
            f"| {r['fsck_list_s']} | "
            f"{r['fsck_fence_s']} | {r['fsck_drop_s']} | {r['fenced_objects']} | {r['dropped_samples']} | "
            f"{r['ms_per_churned_record']} |"
        )

    summary = outdir / f"{args.tag}_overhead.md"
    summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
