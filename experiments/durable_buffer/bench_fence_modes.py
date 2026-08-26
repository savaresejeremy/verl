#!/usr/bin/env python3
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
"""Snapshot fence cost: per-writer op-log offsets against a per-object manifest.

`bench_overhead.py --phase journal` times a whole journal tick, whose cost is dominated
by the controller pickle. This script isolates the *storage-side fence* instead, which is
the part that differs between the shipped design and the design it replaced:

  * `write_manifest(full=False)`: flush each writer's op log and record its length.
    O(writer processes), independent of store size. This is what a tick does.
  * `write_manifest(full=True)`: additionally `scandir` the store and `lstat` every
    object into an `(inode, size, mtime)` manifest. O(store), and the reason the
    manifest-based tick mode was removed rather than kept as an option.

The gap between the two on a shared filesystem is the measurement behind the RFC's
"16.8 s at 50,000 objects against 0.15 s" claim, and dividing the full-manifest time by
its row count gives the per-metadata-operation price for that filesystem.

Run it where the store lives:

    python3 experiments/durable_buffer/bench_fence_modes.py \\
        --objects 5000,20000,50000 --store /shared/fs/runs/<run>/fence_store \\
        --journal-root /shared/fs/runs/<run>/fence_journal --tag costfsx --out <dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_overhead import BATCH, _fields, _init, _provenance  # noqa: E402

MANIFEST_FILE = "store_manifest.tsv"


def seed(tq, partition: str, n_objects: int, resp_len: int) -> int:
    """Write n_objects/2 trajectories of 2 fields each, matching phase_journal's shape."""
    per_traj = 2
    n_traj = max(n_objects // per_traj, 1)
    keys = [f"t{i}" for i in range(n_traj)]
    for off in range(0, n_traj, BATCH):
        chunk = keys[off : off + BATCH]
        tq.kv_batch_put(
            keys=chunk,
            partition_id=partition,
            fields=_fields(len(chunk), resp_len),
            tags=[{"status": "success", "global_steps": 1}] * len(chunk),
        )
    return n_traj


def attach_client(root: Path):
    """Attach to the seeded store WITHOUT writing to it (create=False never stamps)."""
    from verl.experimental.durable_buffer.fs_kv import DurableFSClient

    return DurableFSClient({"root": str(root), "create": False})


def measure(objects: int, store: Path, journal_root: Path, resp_len: int, repeat: int) -> dict:
    from verl.experimental.durable_buffer.journal import JournalWriter

    tq = _init("durable", store)  # wipes and recreates the store
    n_traj = seed(tq, "bench", objects, resp_len)
    on_disk = len(list(store.glob("*.bin")))

    # (1) a whole journal tick, op-log fence mode: what the shipped code does.
    import shutil

    shutil.rmtree(journal_root, ignore_errors=True)
    jw = JournalWriter(journal_root, interval_s=0.0, ring=3)
    tick_s = [jw.tick()["tick_s"] for _ in range(repeat)]

    # (2) and (3): the storage-side fence alone, both modes, via the real client.
    client = attach_client(store)
    offsets_s, full_s, rows = [], [], None
    for i in range(repeat):
        d = journal_root / f"_fence_offsets_{i}"
        t0 = time.perf_counter()
        client.write_manifest(d, full=False)
        offsets_s.append(time.perf_counter() - t0)
    for i in range(repeat):
        d = journal_root / f"_fence_full_{i}"
        t0 = time.perf_counter()
        rows = client.write_manifest(d, full=True)
        full_s.append(time.perf_counter() - t0)
    manifest_bytes = (journal_root / f"_fence_full_{repeat - 1}" / MANIFEST_FILE).stat().st_size

    tq.close()
    mean_full = statistics.mean(full_s)
    mean_off = statistics.mean(offsets_s)
    return {
        "objects_requested": objects,
        "objects_on_disk": on_disk,
        "trajectories_seeded": n_traj,
        "resp_len": resp_len,
        "repeat": repeat,
        "tick_s_mean": round(statistics.mean(tick_s), 4),
        "tick_s_min": round(min(tick_s), 4),
        "fence_offsets_s_mean": round(mean_off, 4),
        "fence_offsets_s_min": round(min(offsets_s), 4),
        "manifest_full_s_mean": round(mean_full, 4),
        "manifest_full_s_min": round(min(full_s), 4),
        "manifest_rows": rows,
        "manifest_MB": round(manifest_bytes / 1e6, 3),
        "manifest_over_offsets": round(mean_full / mean_off, 1) if mean_off else None,
        "manifest_over_tick": round(mean_full / statistics.mean(tick_s), 1) if tick_s else None,
        "us_per_manifest_row": round(mean_full / max(rows or 1, 1) * 1e6, 1),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--objects", default="5000,20000,50000")
    p.add_argument("--resp-len", type=int, default=256)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--store", required=True)
    p.add_argument("--journal-root", required=True)
    p.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for objects in [int(x) for x in a.objects.split(",")]:
        r = measure(objects, Path(a.store), Path(a.journal_root), a.resp_len, a.repeat)
        r["provenance"] = _provenance()
        rows.append(r)
        (outdir / f"{a.tag}_fence_{objects}.json").write_text(json.dumps(r, indent=2))
        print(
            f"fence {objects}: tick {r['tick_s_mean']}s | offsets {r['fence_offsets_s_mean']}s | "
            f"full manifest {r['manifest_full_s_mean']}s ({r['manifest_rows']} rows, "
            f"{r['us_per_manifest_row']} us/row) | full/offsets {r['manifest_over_offsets']}x",
            flush=True,
        )

    lines = [
        f"# Snapshot fence cost: op-log offsets vs per-object manifest ({a.tag})\n",
        f"provenance: commit `{rows[0]['provenance']['commit']}` | python "
        f"{rows[0]['provenance']['python']} | transfer_queue {rows[0]['provenance']['transfer_queue']} | "
        f"{rows[0]['provenance']['generated_at']}",
        f"invocation: `{rows[0]['provenance']['invocation']}`\n",
        "| live objects | whole tick s | fence, op-log offsets s | fence, full manifest s | "
        "manifest rows | manifest MB | full/offsets | us per manifest row |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['objects_on_disk']} | {r['tick_s_mean']} | {r['fence_offsets_s_mean']} | "
            f"{r['manifest_full_s_mean']} | {r['manifest_rows']} | {r['manifest_MB']} | "
            f"{r['manifest_over_offsets']}x | {r['us_per_manifest_row']} |"
        )
    (outdir / f"{a.tag}_fence_modes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
