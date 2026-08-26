# Durable replay buffer — crash-recovery benchmark (agentic)

provenance: commit `dabcf10bad7a627a42aabbc01826bdb090704bf7` (`dabcf10bad7a`) | harness sha256 `096603b11df39665` | python 3.12.13 | torch 2.13.0+cpu | transfer_queue 0.1.10.dev0 | 2026-08-26T21:44:44Z
invocation: `experiments/durable_buffer/bench_recovery.py --all --preset agentic --sweep-kill 7,9 --tag agentic --out experiments/durable_buffer/results`

preset: `agentic` | 6 groups/step (rollout.n=8, 2 still generating at step end), 3 groups trained/step, 600s/step, trainer.save_freq=5 steps, journal every 2.0s, generation 300.0s/trajectory

arms: `none` = stock in-memory buffer · `ckpt` = durable store journaled only at trainer checkpoints · `journal` = durable store + 2.0s journal (real JournalWriter ticks) · `journal_dl` = journal + dataloader state

| arm | kill@step | untrained@kill | recovered | lost | traj lost | gen-s wasted | dup traj | recover s | 1st sample s | put ms | tick ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 7 | 7 | 0 | 7 | 56 | 16800.0 | 8 | 0.0 | None | 0.93 | None |
| ckpt | 7 | 7 | 0 | 7 | 56 | 16800.0 | 8 | 0.021 | None | 1.296 | 2.86 |
| journal | 7 | 7 | 7 | 0 | 0 | 0.0 | 64 | 0.017 | 0.005 | 1.287 | 2.8 |
| journal_dl | 7 | 7 | 7 | 0 | 0 | 0.0 | 0 | 0.017 | 0.005 | 1.39 | 3.01 |
| none | 9 | 9 | 0 | 9 | 72 | 21600.0 | 56 | 0.0 | None | 0.946 | None |
| ckpt | 9 | 9 | 0 | 9 | 72 | 21600.0 | 56 | 0.023 | None | 1.216 | 2.5 |
| journal | 9 | 9 | 9 | 0 | 0 | 0.0 | 128 | 0.018 | 0.005 | 1.171 | 2.83 |
| journal_dl | 9 | 9 | 9 | 0 | 0 | 0.0 | 0 | 0.018 | 0.005 | 1.291 | 2.97 |

Trajectory-generation-seconds recovered by the journal arm vs the stock buffer: kill@7: **16800s**, kill@9: **21600s**
