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
"""Fixtures for the durable-buffer CPU tests.

No GPU and no cluster: a real TransferQueue instance with this package's durable
backend. Process death is simulated with ``tq.close()`` + ``tq.init()``, which gives a
genuinely fresh controller while the durable store survives — the honest analogue of a
trainer crash. (verl's own V1 checkpoint tests simulate death by wiping the partition,
which on an external store deletes the very data the checkpoint refers to.)
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

import verl.experimental.durable_buffer.register  # noqa: F401  registers the backend
from verl.experimental.durable_buffer.fs_kv import backend_config


@pytest.fixture(scope="session")
def store_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("durable_store")
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def tq_conf(store_root):
    return backend_config(store_root)


@pytest.fixture(scope="session")
def tq_session(tq_conf):
    import transfer_queue as tq

    tq.init(tq_conf)
    yield tq
    try:
        tq.close()
    except Exception:  # pragma: no cover
        pass


@pytest.fixture
def partition_id() -> str:
    return f"test-{uuid.uuid4().hex}"
