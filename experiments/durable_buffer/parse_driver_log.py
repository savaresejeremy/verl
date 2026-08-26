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
#!/usr/bin/env python3
"""Parse a cluster run's driver.log into the per-step JSON artifact used by the
durable-buffer result set.

verl's console logger emits one line per step in `concat_dict_to_str` form:

    step:7 - critic/rewards/mean:0.0234 - durable_buffer/journal_ticks:124 - ...

so a step line is a `-`-separated list of `key:value` pairs led by `step:N`. This
reads those lines and keeps a chosen subset of keys, plus the `[DIAG]` banners
(one per job attempt, which is how a preemption-and-restart shows up in the log).

Two derived series are added because the journal counters are cumulative and the
per-tick cost is what the analysis needs:

    ticks_delta     ticks added since the previous step line
    tick_window_s   mean per-tick cost over just that step's ticks, recovered as
                    (mean_i*ticks_i - mean_{i-1}*ticks_{i-1}) / (ticks_i - ticks_{i-1})

`journal_last_s` is the single most recent tick and `tick_window_s` is the mean
over the step, so a step whose window mean sits under `interval_s` while its
`journal_last_s` spikes had one slow tick rather than uniformly slow ticks.

Usage:
    parse_driver_log.py <driver.log> <run-name> [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Keys kept in the per-step series. Anything absent from a line is emitted as
# null rather than dropped, so every record has the same shape.
KEYS = [
    "durable_buffer/journal_ticks",
    "durable_buffer/journal_mean_s",
    "durable_buffer/journal_last_s",
    "durable_buffer/journal_failures",
    "critic/rewards/mean",
    "critic/score/mean",
    "timing_s/step",
    "perf/throughput",
    "training/off_policy/trajectory_spans/mean",
    "training/off_policy/trajectory_staleness/mean",
]

STEP_RE = re.compile(r"step:(\d+) -")

# Some metrics reach the console logger as numpy scalars, whose repr survives the
# f-string: `trajectory_spans/mean:np.float64(1.0)`, `.../max:np.int64(1)`. A parser
# that only tries float() drops those keys and writes null, which reads as "the run
# did not emit this metric" instead of "the parser could not read it". Unwrap the
# repr, and record whatever still fails to convert so a gap is visible in the
# artifact rather than silent.
REPR_RE = re.compile(r"^(?:np|numpy)\.\w+\(([^)]*)\)$|^tensor\(([^)]*)\)$")


def to_number(raw: str) -> float | None:
    s = raw.strip()
    m = REPR_RE.match(s)
    if m:
        s = next(g for g in m.groups() if g is not None)
    try:
        return float(s)
    except ValueError:
        return None


def parse_step_line(line: str) -> tuple[dict, list[str]] | None:
    m = STEP_RE.search(line)
    if not m:
        return None
    rec: dict = {"step": int(m.group(1))}
    # split on " - " so a value containing a colon (a path, a timestamp) survives
    fields = line[m.start() :].split(" - ")
    seen: dict[str, float] = {}
    unparsed: list[str] = []
    for f in fields:
        k, sep, v = f.partition(":")
        if not sep:
            continue
        k = k.strip()
        n = to_number(v)
        if n is None:
            if k in KEYS:
                unparsed.append(f"{k}={v.strip()[:40]}")
            continue
        seen[k] = n
    for k in KEYS:
        rec[k] = seen.get(k)
    return rec, unparsed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    series: list[dict] = []
    diag: list[str] = []
    reissued: list[str] = []
    unparsed: list[str] = []
    n_step_lines = 0
    with open(args.log, errors="replace") as fh:
        for line in fh:
            if "[DIAG]" in line:
                diag.append(line.strip()[:400])
            if "Re-issued" in line:
                reissued.append(line.strip()[:400])
            parsed = parse_step_line(line)
            if parsed is None:
                continue
            rec, bad = parsed
            n_step_lines += 1
            series.append(rec)
            unparsed.extend(bad)

    # cumulative -> per-step tick cost
    prev_ticks = prev_mean = None
    for rec in series:
        ticks, mean = rec["durable_buffer/journal_ticks"], rec["durable_buffer/journal_mean_s"]
        rec["ticks_delta"] = None
        rec["tick_window_s"] = None
        if ticks is not None and mean is not None:
            if prev_ticks is not None and ticks > prev_ticks:
                d = ticks - prev_ticks
                rec["ticks_delta"] = d
                rec["tick_window_s"] = round((mean * ticks - prev_mean * prev_ticks) / d, 4)
            elif prev_ticks is None:
                # first line of an attempt: the cumulative mean IS the window mean
                rec["ticks_delta"] = ticks
                rec["tick_window_s"] = round(mean, 4)
            # a ticks count that went DOWN means a new attempt (counter restarted)
            if prev_ticks is not None and ticks < prev_ticks:
                rec["ticks_delta"] = ticks
                rec["tick_window_s"] = round(mean, 4)
            prev_ticks, prev_mean = ticks, mean

    steps = [r["step"] for r in series]
    out = {
        "run": args.run,
        "n_step_lines": n_step_lines,
        "steps_seen": steps,
        "max_step": max(steps) if steps else None,
        "n_attempts": len([d for d in diag if "host=" in d]),
        "unparsed_values": sorted(set(unparsed)),
        "diag_lines": diag,
        "reissued_lines": reissued,
        "series": series,
    }
    text = json.dumps(out, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}: {len(series)} step records, max_step={out['max_step']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
