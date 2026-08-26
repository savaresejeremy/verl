# Durable replay buffer — crash-recovery benchmark (qwen27b)

provenance: commit `dabcf10bad7a627a42aabbc01826bdb090704bf7` (`dabcf10bad7a`) | harness sha256 `096603b11df39665` | python 3.12.13 | torch 2.13.0+cpu | transfer_queue 0.1.10.dev0 | 2026-08-26T21:40:35Z
invocation: `experiments/durable_buffer/bench_recovery.py --all --preset qwen27b --sweep-kill 6,8,10 --tag qwen27b --out experiments/durable_buffer/results`

preset: `qwen27b` | 8 groups/step (rollout.n=8, 2 still generating at step end), 4 groups trained/step, 100s/step, trainer.save_freq=5 steps, journal every 2.0s, generation 25.0s/trajectory

arms: `none` = stock in-memory buffer · `ckpt` = durable store journaled only at trainer checkpoints · `journal` = durable store + 2.0s journal (real JournalWriter ticks) · `journal_dl` = journal + dataloader state

| arm | kill@step | untrained@kill | recovered | lost | traj lost | gen-s wasted | dup traj | recover s | 1st sample s | put ms | tick ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 6 | 12 | 0 | 12 | 96 | 2400.0 | 0 | 0.0 | None | 0.94 | None |
| ckpt | 6 | 12 | 6 | 6 | 48 | 1200.0 | 0 | 0.021 | 0.005 | 1.348 | 2.94 |
| journal | 6 | 12 | 12 | 0 | 0 | 0.0 | 48 | 0.019 | 0.005 | 1.152 | 3.0 |
| journal_dl | 6 | 12 | 12 | 0 | 0 | 0.0 | 0 | 0.019 | 0.005 | 1.189 | 3.08 |
| none | 8 | 16 | 0 | 16 | 128 | 3200.0 | 16 | 0.0 | None | 0.938 | None |
| ckpt | 8 | 16 | 0 | 16 | 128 | 3200.0 | 16 | 0.028 | None | 1.122 | 2.7 |
| journal | 8 | 16 | 16 | 0 | 0 | 0.0 | 144 | 0.019 | 0.005 | 1.15 | 2.98 |
| journal_dl | 8 | 16 | 16 | 0 | 0 | 0.0 | 0 | 0.02 | 0.005 | 1.055 | 3.19 |
| none | 10 | 20 | 0 | 20 | 160 | 4000.0 | 0 | 0.0 | None | 0.926 | None |
| ckpt | 10 | 20 | 20 | 0 | 0 | 0.0 | 0 | 0.009 | 0.006 | 1.117 | 2.97 |
| journal | 10 | 20 | 20 | 0 | 0 | 0.0 | 0 | 0.02 | 0.006 | 1.075 | 3.07 |
| journal_dl | 10 | 20 | 20 | 0 | 0 | 0.0 | 0 | 0.02 | 0.005 | 1.177 | 3.33 |

Trajectory-generation-seconds recovered by the journal arm vs the stock buffer: kill@6: **2400s**, kill@8: **3200s**, kill@10: **4000s**
