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
"""pytest plugin: run verl's own V1 suites against the durable backend.

verl's V1 tests call a bare ``tq.init()``, which selects the in-memory backend. Loading
this plugin patches ``transfer_queue.init`` so those same tests exercise the durable
storage path instead — a much stronger statement than any bespoke probe, because it is
verl's full staleness / DAPO / failure-refill / reissue matrix.

    TQ_DURABLE_TEST_STORE=/tmp/store \
    pytest tests/trainer/ppo/v1 -p tests.experimental.durable_buffer.durable_plugin

It also prints storage hygiene at the end of the session: with correct clear semantics
the store must be EMPTY once every sample has been consumed. (verl's suite does not
assert this today, which is exactly why a leaking ``clear_data`` can pass CI.)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import transfer_queue as tq

import verl.experimental.durable_buffer.register  # noqa: F401  registers the backend
from verl.experimental.durable_buffer.fs_kv import STATS, backend_config

STORE = Path(os.environ.get("TQ_DURABLE_TEST_STORE", "/tmp/verl_v1_durable_store"))
shutil.rmtree(STORE, ignore_errors=True)
STORE.mkdir(parents=True, exist_ok=True)

_ORIG_INIT = tq.init


def _patched_init(conf=None):
    return _ORIG_INIT(backend_config(STORE) if conf is None else conf)


tq.init = _patched_init
print(f"[durable_plugin] tq.init patched -> DurableFS root={STORE}", flush=True)


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    objects = list(STORE.glob("*.bin"))
    print(
        f"[durable_plugin] store after suite: {len(objects)} objects, "
        f"{sum(p.stat().st_size for p in objects) / 1e6:.2f} MB; stats={STATS}",
        flush=True,
    )
