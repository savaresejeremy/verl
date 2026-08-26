# Durable replay buffer — cost of durability (cost)

provenance: commit `dabcf10bad7a627a42aabbc01826bdb090704bf7` (`dabcf10bad7a`) | harness sha256 `096603b11df39665` | python 3.12.13 | torch 2.13.0+cpu | transfer_queue 0.1.10.dev0 | 2026-08-26T21:29:09Z
invocation: `experiments/durable_buffer/bench_overhead.py --all --tag cost --out experiments/durable_buffer/results`

## Producer path (write-as-produced)

| backend | resp_len | trajectories | traj/s | ms per trajectory | read rewards batch (ms) | store MB |
|---|---|---|---|---|---|---|
| memory | 512 | 4000 | 17236.1 | 0.058 | 17.11 | 0.0 |
| memory | 2048 | 4000 | 18761.2 | 0.053 | 16.87 | 0.0 |
| durable | 512 | 4000 | 2936.4 | 0.34 | 36.63 | 16.8 |
| durable | 2048 | 4000 | 2796.0 | 0.357 | 36.55 | 66.0 |

## Journal tick (metadata-only snapshot + fence manifest)

| live objects | tick s (mean) | controller pkl MB | manifest MB | us/object | affordable cadence @5% overhead |
|---|---|---|---|---|---|
| 5000 | 0.0182 | 0.15 | None | 3.64 | 0.4s |
| 20000 | 0.0594 | 0.59 | None | 2.97 | 1.2s |
| 50000 | 0.1522 | 1.49 | None | 3.04 | 3.0s |

## Restore (fsck) cost

| store objects | mode | load journal s | fsck total s | list | fence | drop | fenced | dropped | ms/churned record |
|---|---|---|---|---|---|---|---|---|---|
| 5000 | oplog | 0.0071 | 0.0643 | 0.0264 | 0.0059 | 0.032 | 248 | 124 | 0.173 |
| 20000 | oplog | 0.0177 | 0.23 | 0.0982 | 0.0253 | 0.1065 | 1000 | 500 | 0.153 |
| 50000 | oplog | 0.039 | 0.5912 | 0.2575 | 0.0669 | 0.2669 | 2496 | 1248 | 0.158 |
