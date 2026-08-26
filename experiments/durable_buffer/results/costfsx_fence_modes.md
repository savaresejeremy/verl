# Snapshot fence cost: op-log offsets vs per-object manifest (costfsx)

provenance: commit `6cae67a7c9e17f0861a9bc63b826b17f90540d67` | python 3.12.3 | transfer_queue 0.1.10.dev0 | 2026-08-23T06:37:46Z
invocation: `experiments/durable_buffer/bench_fence_modes.py --objects 20000 --tag costfsx --out /shared/fs/durable_buffer/bench_shared_fs/results --store /shared/fs/durable_buffer/bench_shared_fs/fence_store --journal-root /shared/fs/durable_buffer/bench_shared_fs/fence_journal`

| live objects | whole tick s | fence, op-log offsets s | fence, full manifest s | manifest rows | manifest MB | full/offsets | us per manifest row |
|---|---|---|---|---|---|---|---|
| 20000 | 0.0936 | 0.0162 | 8.1234 | 20000 | 1.128 | 500.8x | 406.2 |
