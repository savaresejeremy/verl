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
"""Helpers shared by the durable-buffer CPU tests (mimic verl's V1 write patterns)."""

from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict


def make_fields(n: int, value: int, resp_len: int = 8) -> TensorDict:
    return TensorDict(
        {
            "response_ids": torch.full((n, resp_len), value, dtype=torch.long),
            "rewards": torch.full((n,), float(value)),
        },
        batch_size=[n],
    )


def submit_prompt(tq, partition_id: str, uid: str, status: str = "running", global_steps: int = 1, payload=True):
    """Mimic verl's ``_submit_batch_to_rollout``: prompt key + (async) persisted payload."""
    kwargs = {}
    if payload:
        kwargs["fields"] = {"prompt_ids": torch.arange(4).unsqueeze(0)}
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": status, "global_steps": global_steps},
        **kwargs,
    )


def finish_group(tq, partition_id: str, uid: str, n: int, value: int, global_steps: int = 1, status: str = "success"):
    """Mimic ``_agent_loop_postprocess``: trajectories first, terminal prompt tag later."""
    keys = [f"{uid}_{s}_0" for s in range(n)]
    tq.kv_batch_put(
        keys=keys,
        partition_id=partition_id,
        fields=make_fields(len(keys), value),
        tags=[{"status": status, "global_steps": global_steps}] * len(keys),
    )
    return keys


def tag_prompt_finished(tq, partition_id: str, uid: str, global_steps: int = 1):
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": "finished", "global_steps": global_steps},
    )


def consume_group(tq, partition_id: str, uid: str, n: int):
    """Mimic ``ReplayBuffer._clear_groups``: prompt key + trajectory keys in ONE call."""
    tq.kv_clear(keys=[uid] + [f"{uid}_{s}_0" for s in range(n)], partition_id=partition_id)


def objects(root: str | Path) -> list[str]:
    return sorted(p.name for p in Path(root).glob("*.bin"))


def indexes_of(tq, partition_id: str, keys: list[str]) -> list[int]:
    meta = tq.get_client().kv_retrieve_meta(keys=keys, partition_id=partition_id, create=False)
    return list(meta.global_indexes)


def crash_and_restart(tq_conf):
    """Kill the TransferQueue actors and bring a FRESH controller up (the store survives)."""
    import transfer_queue as tq

    tq.close()
    tq.init(tq_conf)
    return tq
