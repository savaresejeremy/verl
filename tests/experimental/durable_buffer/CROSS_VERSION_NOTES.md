# Cross-version store compatibility: TransferQueue 0.1.9 and 0.1.10.dev0

Scripted evidence, not a permanent test. This is deliberately not in the suite: it needs a second interpreter with a different TransferQueue installed, which a single pytest process cannot provide. Re-run it by hand when either version moves.

Measured 2026-08-21 on verl `890dfc3` + the durable_buffer port, against TransferQueue 0.1.9 (PyPI) and 0.1.10.dev0 (a local checkout of Ascend/TransferQueue `main` @ `2247b95`).

## Result

Both directions are clean, in both the byte sense and the recovery sense.

| writer | reader | object bytes identical | journal selectable | fsck | live keys | payloads |
|---|---|---|---|---|---|---|
| 0.1.9 | 0.1.10.dev0 | yes, all 10 objects by sha256 | yes, the writer's own slot | 0 fenced, 0 dropped | identical set | identical values |
| 0.1.10.dev0 | 0.1.9 | yes, all 10 objects by sha256 | yes, the writer's own slot | 0 fenced, 0 dropped | identical set | identical values |

Two further observations from the same runs:

- The two versions produce **byte-identical objects for identical inputs**: every one of the 10 object sha256s matches across the 0.1.9-written and 0.1.10.dev0-written stores. TQ#154 changed `pack_into` to copy through a memoryview instead of `torch.frombuffer`, which changed how the bytes get there but not what they are.
- Object bytes are unchanged **after** the read path as well as before it, so recovery and `kv_batch_get` do not rewrite what they read.

## Two version-dependent behaviours found while doing this

Both are handled in the package rather than assumed away, and both are worth knowing before citing the compatibility result.

**`KVStorageManager.__init__` does not take `zmq_context` at 0.1.9.** Measured signature at 0.1.9 is `(self, controller_info, config)`; at 0.1.10.dev0 it is `(self, controller_info, config, zmq_context=None)`. `StorageManagerFactory._filter_supported_kwargs` does not exist at 0.1.9 either, so there is no kwarg filter to soften a mismatch. `DurableFSManager` therefore probes the parent signature at import (`_PARENT_ACCEPTS_ZMQ_CONTEXT` in `fs_kv.py`) instead of matching one version's shape. Verified under both: `False` at 0.1.9, `True` at 0.1.10.dev0, and the manager constructs and serves data either way.

**The `.old` checkpoint sibling is a 0.1.10.dev0-only artifact.** `tq.save_checkpoint` at 0.1.9 writes a `.tmp` and renames it into place, with no `.old` step; the rename-aside-to-`.old` protocol and the `load_checkpoint` roll-forward from it are TQ#155, after 0.1.9. So the journal-ring selection defect this port fixed cannot occur on a 0.1.9 runtime, and a store written under 0.1.9 will never carry `.old` siblings of its own. Checked by reading `inspect.getsource(tq.save_checkpoint)` under each version.

## Exact commands

Placeholders: `$REPO` is this verl checkout's root; `$MAIN_VENV` is a venv carrying verl's requirements with TransferQueue installed from main.

Throwaway environment (removed afterwards):

```bash
python3.12 -m venv /tmp/vrb-tq019-venv --system-site-packages
/tmp/vrb-tq019-venv/bin/pip install "TransferQueue==0.1.9"
/tmp/vrb-tq019-venv/bin/python -c "import transfer_queue as tq; print(tq.__version__)"   # 0.1.9
```

`PYTHONPATH` pins the checkout's verl, because the 0.1.9 venv inherits system site-packages which carry an unrelated global verl. Confirm before trusting a run:

```bash
PYTHONPATH="$REPO" /tmp/vrb-tq019-venv/bin/python \
  -c "import verl; print(verl.__file__)"
# $REPO/verl/__init__.py
```

The two scripts used are `/tmp/vrb_xver_write.py` (writes two finished groups plus their prompts, ticks one journal slot, prints a digest: object names, per-object sha256, the reward values put in, the live key set) and `/tmp/vrb_xver_read.py` (re-reads the store, checks the digest, selects the journal, runs `fsck.recover`, reads the payloads back, re-checks the digest). They are reproduced at the bottom of this file so the run does not depend on `/tmp` surviving.

Forward direction, 0.1.9 writes and 0.1.10.dev0 reads:

```bash
cd "$REPO"
rm -rf /tmp/vrb_xver_store /tmp/vrb_xver_journal
PYTHONPATH="$REPO" /tmp/vrb-tq019-venv/bin/python \
  /tmp/vrb_xver_write.py /tmp/vrb_xver_store /tmp/vrb_xver_journal \
  2>/dev/null | sed -n '/^{/,$p' > /tmp/vrb_xver_written_019.json
"$MAIN_VENV"/bin/python /tmp/vrb_xver_read.py \
  /tmp/vrb_xver_store /tmp/vrb_xver_journal /tmp/vrb_xver_written_019.json
```

Reverse direction, 0.1.10.dev0 writes and 0.1.9 reads:

```bash
cd "$REPO"
rm -rf /tmp/vrb_xver_store_r /tmp/vrb_xver_journal_r
"$MAIN_VENV"/bin/python /tmp/vrb_xver_write.py /tmp/vrb_xver_store_r /tmp/vrb_xver_journal_r \
  2>/dev/null | sed -n '/^{/,$p' > /tmp/vrb_xver_written_0110.json
PYTHONPATH="$REPO" /tmp/vrb-tq019-venv/bin/python \
  /tmp/vrb_xver_read.py /tmp/vrb_xver_store_r /tmp/vrb_xver_journal_r /tmp/vrb_xver_written_0110.json
```

Both reader invocations printed `"ALL_OK": true` and exited 0.

Teardown:

```bash
rm -rf /tmp/vrb-tq019-venv /tmp/vrb_xver_store /tmp/vrb_xver_journal /tmp/vrb_xver_store_r /tmp/vrb_xver_journal_r
```

## What was written

Store root, 10 objects plus the three backend-internal entries:

```
0@prompt_ids.bin  1@response_ids.bin  1@rewards.bin  2@response_ids.bin  2@rewards.bin
3@prompt_ids.bin  4@response_ids.bin  4@rewards.bin  5@response_ids.bin  5@rewards.bin
_fields.d  _oplog  _store_id
```

Journal slot, both versions:

```
controller_state.pkl  fence.json  journal_stamp.json  metadata.json  oplog_offsets.json
```

## What this does not cover

- One partition, 6 keys, 10 objects, `serializer="tq"`. Not a scale or a format matrix.
- Nothing crashes mid-write in either direction, so this says nothing about a store left inconsistent by 0.1.9 and recovered by 0.1.10.dev0. The in-suite crash tests cover that within one version.
- 0.1.9 is compared against 0.1.10.**dev0** at one commit. `0.1.10` final may differ, and `uv.lock` in this checkout pins yet another TQ (`0.1.9.dev0` at rev `434f8c4`), so "the installed TQ" is not one thing across reproduction paths.
- The controller pickle is compared only through what `load_journal` reads back out of it, not field by field.

## The scripts

`vrb_xver_write.py`:

```python
"""Write a durable store + journal, then print a digest of what was written.

Run under EITHER TransferQueue version. Argument 1 is the store root, argument 2 the
journal ring root. The digest lets the reader side prove it got the same bytes back.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

import transfer_queue as tq
import verl.experimental.durable_buffer.register  # noqa: F401  registers the backend
from verl.experimental.durable_buffer.fs_kv import backend_config
from verl.experimental.durable_buffer.journal import JournalWriter

PARTITION = "train"
GROUPS = (("xv1", 41), ("xv2", 42))
SESSIONS = 2


def main() -> int:
    store, journal_root = Path(sys.argv[1]), Path(sys.argv[2])
    tq.init(backend_config(store))
    try:
        for uid, value in GROUPS:
            tq.kv_put(
                key=uid,
                partition_id=PARTITION,
                tag={"is_prompt": True, "status": "running", "global_steps": 1},
                fields={"prompt_ids": torch.arange(4).unsqueeze(0)},
            )
            keys = [f"{uid}_{s}_0" for s in range(SESSIONS)]
            tq.kv_batch_put(
                keys=keys,
                partition_id=PARTITION,
                fields=TensorDict(
                    {
                        "response_ids": torch.full((len(keys), 8), value, dtype=torch.long),
                        "rewards": torch.full((len(keys),), float(value)),
                    },
                    batch_size=[len(keys)],
                ),
                tags=[{"status": "success", "global_steps": 1}] * len(keys),
            )
            tq.kv_put(
                key=uid,
                partition_id=PARTITION,
                tag={"is_prompt": True, "status": "finished", "global_steps": 1},
            )

        jw = JournalWriter(journal_root, interval_s=0.0, ring=2, metadata_fn=lambda: {"global_steps": 1})
        info = jw.tick()
        assert info is not None, "journal tick failed"

        objects = sorted(p.name for p in store.glob("*.bin"))
        digest = {
            "tq_version": tq.__version__,
            "journal_dir": info["dir"],
            "objects": objects,
            "object_sha256": {p: hashlib.sha256((store / p).read_bytes()).hexdigest() for p in objects},
            "rewards": {f"{uid}_{s}_0": float(value) for uid, value in GROUPS for s in range(SESSIONS)},
            "live_keys": sorted(tq.kv_list().get(PARTITION, {})),
        }
        print(json.dumps(digest, indent=2))
    finally:
        tq.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`vrb_xver_read.py`:

```python
"""Recover a store+journal written by the OTHER TransferQueue version and verify it.

Argument 1 store root, argument 2 journal ring root, argument 3 the writer's digest
JSON. Checks: object bytes unchanged by the read path, fsck clean (nothing fenced,
nothing dropped), the same live keys visible after recovery, and the tensor payloads
deserialize to the values the writer put in.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import transfer_queue as tq
import verl.experimental.durable_buffer.register  # noqa: F401  registers the backend
from verl.experimental.durable_buffer.fsck import recover
from verl.experimental.durable_buffer.fs_kv import backend_config
from verl.experimental.durable_buffer.journal import latest_journal

PARTITION = "train"


def main() -> int:
    store, journal_root = Path(sys.argv[1]), Path(sys.argv[2])
    written = json.loads(Path(sys.argv[3]).read_text())
    out: dict = {"reader_tq_version": tq.__version__, "writer_tq_version": written["tq_version"]}

    # 1. Byte identity BEFORE any TQ involvement: the store is plain files.
    now = sorted(p.name for p in store.glob("*.bin"))
    out["objects_match"] = now == written["objects"]
    out["object_sha256_match"] = all(
        hashlib.sha256((store / name).read_bytes()).hexdigest() == written["object_sha256"][name] for name in now
    )
    out["objects_present"] = len(now)

    # 2. The journal the other version wrote must still be selectable and recoverable.
    journal = latest_journal(journal_root)
    out["journal_selected"] = str(journal) if journal is not None else None
    out["journal_is_writers_slot"] = journal is not None and str(journal) == written["journal_dir"]

    tq.init(backend_config(store, create=False))
    try:
        report = recover(tq, store, journal, PARTITION, group_size=2)
        out["fsck"] = {"fenced_objects": report["fenced_objects"], "dropped_samples": report["dropped_samples"]}
        out["fsck_clean"] = report["fenced_objects"] == 0 and report["dropped_samples"] == 0
        live = sorted(tq.kv_list().get(PARTITION, {}))
        out["live_keys"] = live
        out["live_keys_match"] = live == written["live_keys"]

        # 3. The payloads must deserialize to the writer's values, through the reader's
        #    serial_utils (TQ#154 changed pack_into internals).
        got = {}
        for key, expected in written["rewards"].items():
            data = tq.kv_batch_get(keys=[key], partition_id=PARTITION, select_fields=["rewards"])
            rewards = data["rewards"]
            value = float(rewards.reshape(-1)[0]) if hasattr(rewards, "reshape") else float(rewards[0])
            got[key] = value
        out["rewards_read"] = got
        out["rewards_match"] = got == {k: float(v) for k, v in written["rewards"].items()}

        # 4. Byte identity AFTER the read path too: reads must not rewrite objects.
        out["object_sha256_match_after_read"] = all(
            hashlib.sha256((store / name).read_bytes()).hexdigest() == written["object_sha256"][name]
            for name in sorted(p.name for p in store.glob("*.bin"))
        )
    finally:
        tq.close()

    out["ALL_OK"] = all(
        out[k]
        for k in (
            "objects_match",
            "object_sha256_match",
            "journal_is_writers_slot",
            "fsck_clean",
            "live_keys_match",
            "rewards_match",
            "object_sha256_match_after_read",
        )
    )
    print(json.dumps(out, indent=2))
    return 0 if out["ALL_OK"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```
