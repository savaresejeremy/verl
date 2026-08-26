#!/bin/bash
# Resumable driver for the shared-filesystem cost measurements.
#
# This queue preempts LOW-priority jobs every few minutes, and the whole
# bench_overhead.py --all run needs ~40. Each phase below is a separate short
# invocation writing its own JSON, and the driver skips any phase whose JSON already
# exists, so re-running it after a preemption resumes rather than restarts.
#
# Phases are ordered by how load-bearing they are for RFC section 11.4, so an
# interrupted sequence still yields the most important numbers. A PID file (not a
# process-name grep) is what the external keepalive uses to test liveness: a pattern
# like `pgrep -f run_shared_fs_costs.sh` also matches the kubectl exec command line
# that performs the check, which silently reports a dead driver as alive.
set -uo pipefail
BASE=/shared/fs/durable_buffer
SHAREDFS=$BASE/bench_shared_fs
OUT=$SHAREDFS/results
export PYTHONPATH=$BASE/tq-main:$BASE/verl-head
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/shared/fs/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VERL_USE_EXTERNAL_MODULES=verl.experimental.durable_buffer.register
mkdir -p "$OUT"
cd "$BASE/verl-head" || exit 1
LOG=$SHAREDFS/driver.log
echo $$ > "$SHAREDFS/driver.pid"
trap 'rm -f "$SHAREDFS/driver.pid"' EXIT
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

run_phase() {  # $1=tag  $2..=args
  local tag=$1; shift
  if [ -s "$OUT/$tag.json" ]; then say "SKIP $tag"; return 0; fi
  say "RUN  $tag"
  timeout 1500 python3 experiments/durable_buffer/bench_overhead.py \
      --tag "$tag" --out "$OUT" --store "$SHAREDFS/store" --journal-root "$SHAREDFS/journal" "$@" \
      >> "$LOG" 2>&1
  local rc=$?
  if [ -s "$OUT/$tag.json" ]; then say "OK   $tag"; else say "FAIL $tag rc=$rc"; fi
}

run_fence() {  # $1=objects
  local n=$1
  if [ -s "$OUT/costfsx_fence_$n.json" ]; then say "SKIP fence_$n"; return 0; fi
  say "RUN  fence_$n"
  timeout 1500 python3 experiments/durable_buffer/bench_fence_modes.py \
      --objects "$n" --tag costfsx --out "$OUT" \
      --store "$SHAREDFS/fence_store" --journal-root "$SHAREDFS/fence_journal" >> "$LOG" 2>&1
  local rc=$?
  if [ -s "$OUT/costfsx_fence_$n.json" ]; then say "OK   fence_$n"; else say "FAIL fence_$n rc=$rc"; fi
}

say "=== driver start on $(hostname) pid=$$ ==="

# 1. producer rate: E-B4's shared-filesystem figure, and its in-memory control.
run_phase costfsx_produce_durable_512  --phase produce --backend durable --trajectories 4000 --resp-len 512
run_phase costfsx_produce_durable_2048 --phase produce --backend durable --trajectories 4000 --resp-len 2048
run_phase costfsx_produce_memory_512   --phase produce --backend memory  --trajectories 4000 --resp-len 512
run_phase costfsx_produce_memory_2048  --phase produce --backend memory  --trajectories 4000 --resp-len 2048

# 2. finish the journal-tick series; 5k and 20k are already in hand, 50k is the one
#    the RFC quotes against the full manifest.
run_phase costfsx_journal_50000 --phase journal --objects 50000 --resp-len 256 --repeat 3

# 3. the fence comparison. 5000 first because it is cheap to seed and establishes the
#    ratio at all; then 50000, which is the size the RFC's 16.8 s figure refers to.
run_fence 5000
run_fence 50000
run_fence 20000

# 4. restore-time reconciliation on this filesystem: E-B14.
run_phase costfsx_fsck_50000 --phase fsck --objects 50000 --resp-len 256
run_phase costfsx_fsck_5000  --phase fsck --objects 5000  --resp-len 256
run_phase costfsx_fsck_20000 --phase fsck --objects 20000 --resp-len 256

say "=== driver finished; artifacts: $(ls $OUT | tr '\n' ' ') ==="
