# Durable replay buffer — crash-recovery benchmark (cadence)

provenance: commit `dabcf10bad7a627a42aabbc01826bdb090704bf7` (`dabcf10bad7a`) | harness sha256 `096603b11df39665` | python 3.12.13 | torch 2.13.0+cpu | transfer_queue 0.1.10.dev0 | 2026-08-26T21:49:58Z
invocation: `experiments/durable_buffer/bench_recovery.py --all --preset qwen27b --kill-at 8 --sweep-journal 2,10,30,60 --tag cadence --out experiments/durable_buffer/results`

preset: `qwen27b` | 8 groups/step (rollout.n=8, 2 still generating at step end), 4 groups trained/step, 100s/step, trainer.save_freq=5 steps, journal every 60.0s, generation 25.0s/trajectory

arms: `none` = stock in-memory buffer · `ckpt` = durable store journaled only at trainer checkpoints · `journal` = durable store + 60.0s journal (real JournalWriter ticks) · `journal_dl` = journal + dataloader state

| arm | journal s | untrained@kill | recovered | lost | traj lost | gen-s wasted | dup traj | recover s | 1st sample s | put ms | tick ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | None | 16 | 0 | 16 | 128 | 3200.0 | 16 | 0.0 | None | 0.923 | None |
| ckpt | None | 16 | 0 | 16 | 128 | 3200.0 | 16 | 0.028 | None | 1.128 | 2.76 |
| journal | 2.0 | 16 | 16 | 0 | 0 | 0.0 | 144 | 0.019 | 0.005 | 1.143 | 3.0 |
| journal_dl | 2.0 | 16 | 16 | 0 | 0 | 0.0 | 0 | 0.019 | 0.005 | 1.238 | 3.16 |
| journal | 10.0 | 16 | 16 | 0 | 0 | 0.0 | 144 | 0.019 | 0.005 | 1.114 | 2.99 |
| journal_dl | 10.0 | 16 | 16 | 0 | 0 | 0.0 | 0 | 0.019 | 0.005 | 1.166 | 3.21 |
| journal | 30.0 | 16 | 16 | 0 | 0 | 0.0 | 144 | 0.019 | 0.005 | 1.151 | 3.1 |
| journal_dl | 30.0 | 16 | 16 | 0 | 0 | 0.0 | 0 | 0.019 | 0.006 | 1.157 | 3.29 |
| journal | 60.0 | 16 | 15 | 1 | 8 | 200.0 | 136 | 0.021 | 0.005 | 1.137 | 3.1 |
| journal_dl | 60.0 | 16 | 15 | 1 | 8 | 200.0 | 0 | 0.02 | 0.005 | 1.074 | 3.32 |

Trajectory-generation-seconds recovered by the journal arm vs the stock buffer: kill@8: **3000s**
