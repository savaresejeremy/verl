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
"""Regressions for the round-12 code review of the round-11 exclusivity probe.

Two defects, both on the CLI path the round-11 mechanism made reachable:

1.  **A discovered address is not a live cluster.** ``_live_controller`` handed whatever
    Ray's discovery named straight to ``ray.init``, which retries the GCS connect twenty
    times at about 41 s each: 820 s measured against an address nothing answers at,
    ending in "unknown", which fails closed into refusing the restore. And the trigger is
    the state this tool exists for, not an exotic one: ``RAY_ADDRESS`` outlives the head
    it names and ``/tmp/ray/ray_current_cluster`` is written by ``ray start`` and removed
    by nothing, so a cluster that crashed leaves a pointer behind. A TCP pre-check now
    decides that case in milliseconds and reads it as no cluster, the attach that follows
    a live listener is bounded, every result records what the probe did, and the refusal
    detail branches so a doubt case names the pointer to clear instead of reporting a
    controller as reachable and telling the operator to stop a job that may not exist.
2.  **Only one wiring site can report the journaling metric.** The README credited both
    with ``durable_buffer/journaling_enabled = 0`` on a refused ring lease, but
    ``register.start_journal`` is a plain function with no metrics channel: it signals by
    returning ``None`` and by one warning, and the metric comes from the sampler
    subclass, which is the only part of this package verl merges metrics from.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from helpers import finish_group
from test_round6_on_cpu import _make_ckpt, _tree_bytes

from verl.experimental.durable_buffer import restore_tool as restore_tool_mod
from verl.experimental.durable_buffer.journal import JournalWriter
from verl.experimental.durable_buffer.restore_tool import (
    ATTACH_GCS_CONNECT_ATTEMPTS,
    PROBE_ADDRESS_IS_DEAD,
    PROBE_CLUSTER_UNJOINABLE,
    PROBE_IN_CALLER_RAY,
    restore,
)

# --------------------------------------------------------------------------------------
# 1. a stale cluster pointer costs milliseconds, not fourteen minutes
# --------------------------------------------------------------------------------------


def test_the_liveness_precheck_answers_yes_no_and_dont_know():
    """Three answers, because two of them are not the same "no".

    A listener means something is there (not that it is a healthy GCS, which is the
    bounded attach's problem). A refused connection means the address is a pointer to a
    cluster that is gone. An address this cannot parse is not judged at all: ``None``
    leaves the decision to the attach rather than inventing a verdict from a bad parse.
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    address = f"127.0.0.1:{listener.getsockname()[1]}"
    try:
        assert restore_tool_mod._address_answers(address) is True
    finally:
        listener.close()
    assert restore_tool_mod._address_answers(address) is False, "the port is closed again"
    assert restore_tool_mod._address_answers("no-colon-at-all") is None
    assert restore_tool_mod._address_answers("10.0.0.1:auto") is None, "a non-numeric port is unparseable"
    assert restore_tool_mod._address_answers(":6379") is None, "a missing host is unparseable"


def test_the_attach_bounds_rays_gcs_retries_and_puts_them_back():
    """Twenty attempts at 41 s each is where the 820 s came from. One retry is enough
    once a TCP connection has already proved something is listening, and the tool must not
    leave Ray's constant narrowed afterwards: the caller's own Ray, brought up later for
    the reconcile, needs the stock value.
    """
    from ray._private import ray_constants

    stock = ray_constants.NUM_REDIS_GET_RETRIES
    with restore_tool_mod._bounded_gcs_connect():
        assert ray_constants.NUM_REDIS_GET_RETRIES == ATTACH_GCS_CONNECT_ATTEMPTS
        assert ATTACH_GCS_CONNECT_ATTEMPTS < stock, "the bound must be tighter than Ray's default"
    assert ray_constants.NUM_REDIS_GET_RETRIES == stock
    try:
        with restore_tool_mod._bounded_gcs_connect():
            raise RuntimeError("the attach failed")
    except RuntimeError:
        pass
    assert ray_constants.NUM_REDIS_GET_RETRIES == stock, "a failed attach must restore the constant too"


_DEAD_POINTER_PROBE = """
import json
import os
import time
from pathlib import Path

import ray
from ray._private import services
from verl.experimental.durable_buffer import restore_tool as rt

# The scenario is "the only thing discoverable is a stale pointer", and the parent process
# that launched this child is running a Ray of its own, which the enumeration would
# otherwise find on this same box. Emptying the gcs_server scan leaves RAY_ADDRESS as the
# single source and keeps everything under test real: the enumeration, the TCP pre-check
# and the attach all run unpatched.
services.find_gcs_addresses = lambda: frozenset()

out = {"initialized_before": ray.is_initialized()}
record = {}
started = time.monotonic()
out["found"] = rt._live_controller(probe_record=record)
out["probe_s"] = round(time.monotonic() - started, 2)
out["record"] = record

started = time.monotonic()
result = rt.restore(
    store=Path(os.environ["PROBE_STORE"]),
    journal_root=Path(os.environ["PROBE_JOURNAL"]),
    ckpt_dir=Path(os.environ["PROBE_CKPT"]),
    partition=os.environ["PROBE_PARTITION"],
    dry_run=True,
    journal_slot="journal-999",
)
out["restore_s"] = round(time.monotonic() - started, 2)
out["status"] = result["status"]
out["controller_probe"] = result["controller_probe"]
out["initialized_after"] = ray.is_initialized()
print("RESULT " + json.dumps(out, default=str))
"""

#: The verdict budget for a dead pointer. Generous next to the milliseconds a refused
#: connection costs, and two orders of magnitude under the 820 s this replaces, so a
#: failure means the pre-check stopped covering the ``ray.init`` path rather than that a
#: loaded box was slow.
DEAD_POINTER_BUDGET_S = 15.0


def test_a_dead_cluster_pointer_reaches_a_verdict_in_seconds(tq_session, store_root, partition_id, tmp_path):
    """The defect, measured the way it was found: a fresh interpreter, no Ray, and
    ``RAY_ADDRESS`` pointing at a port nothing serves.

    Ray canonicalizes that to the node's own address and hands it back as a discovery
    result, so the tool used to spend 820 s in ``ray.init`` and then refuse with
    "unknown". Both verdicts here, the probe on its own and a whole ``restore``, must land
    inside the budget, and the child must end with no Ray of any kind: joining the dead
    address is exactly what must not have been attempted. The status is
    ``journal_slot_not_found`` because a slot that does not exist is the cheapest way to
    reach a real report through the probe without reconciling anything.
    """
    finish_group(tq_session, partition_id, "dead12", n=2, value=7)
    ckpt_dir, step_dir = _make_ckpt(tq_session, tmp_path, step=5)
    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)

    env = dict(os.environ)
    env.update(
        RAY_ADDRESS="127.0.0.1:1",
        PROBE_STORE=str(store_root),
        PROBE_JOURNAL=str(journal_root),
        PROBE_CKPT=str(ckpt_dir),
        PROBE_PARTITION=partition_id,
    )
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", _DEAD_POINTER_PROBE], capture_output=True, text=True, env=env, timeout=180
    )
    child_s = time.monotonic() - started
    assert done.returncode == 0, f"probe failed:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    line = next((ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, f"probe printed no RESULT:\n{done.stdout[-3000:]}"
    got = json.loads(line[len("RESULT ") :])

    assert got["initialized_before"] is False, "the point of the subprocess is to have no Ray"
    assert got["found"] is None, f"a pointer nothing answers at is not a controller: {got['found']}"
    assert got["probe_s"] < DEAD_POINTER_BUDGET_S, f"the probe took {got['probe_s']}s (820 s was the defect)"
    assert got["restore_s"] < DEAD_POINTER_BUDGET_S, f"the restore took {got['restore_s']}s"
    assert got["record"]["outcome"] == PROBE_ADDRESS_IS_DEAD, got["record"]
    assert any(address.endswith(":1") for address in got["record"]["discovered"]), got["record"]
    assert set(got["record"]["answered"].values()) == {False}, got["record"]
    assert got["status"] == "journal_slot_not_found", got
    assert got["controller_probe"] == got["record"], "the report must carry the probe's own account"
    assert got["initialized_after"] is False, "the dead address must never have been joined"
    assert child_s < 120, f"the whole invocation took {child_s:.0f}s"
    assert _tree_bytes(store_root) == before_store
    assert _tree_bytes(ckpt_dir) == before_ckpt
    assert not (step_dir / "reconcile.lock").exists(), "a dry run writes nothing, lock included"


def test_a_cluster_that_cannot_be_joined_names_the_pointer_to_clear(
    tq_session, store_root, partition_id, tmp_path, monkeypatch
):
    """Doubt is still refused, but it must not be reported as a sighting.

    The old detail said a controller "is reachable" and told the operator to stop the job
    for every case including this one, where nothing was ever reached: on the crash path
    that sends someone hunting a job a stale pointer invented, with the DANGEROUS override
    as the only escape the report names. The remedy that keeps the check comes first.
    """
    import ray

    finish_group(tq_session, partition_id, "join12", n=2, value=8)
    ckpt_dir, step_dir = _make_ckpt(tq_session, tmp_path, step=5)
    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)
    address = "10.255.255.1:6379"

    def unjoinable(**kwargs):
        raise ConnectionError(f"Failed to connect to Ray cluster at {address}")

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(restore_tool_mod, "_discoverable_cluster_addresses", lambda: (address,))
    monkeypatch.setattr(restore_tool_mod, "_address_answers", lambda _address: True)
    monkeypatch.setattr(ray, "init", unjoinable)

    out = restore(store=store_root, journal_root=None, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)

    assert out["status"] == "live_controller_present", out
    assert out["plan_available"] is False, "the refusal precedes the preflight"
    assert out["controller_probe"]["outcome"] == PROBE_CLUSTER_UNJOINABLE, out["controller_probe"]
    assert out["controller_probe"]["discovered"] == [address]
    assert out["controller_probe"]["answered"] == {address: True}, "something listened; joining is what failed"
    detail = out["detail"]
    assert "could not be ruled out" in detail, detail
    assert "a TransferQueue controller is reachable" not in detail, detail
    assert "Stop the job" not in detail, detail
    assert address in detail and "RAY_ADDRESS" in detail and "/tmp/ray/ray_current_cluster" in detail, detail
    assert detail.index("RAY_ADDRESS") < detail.index("--allow-live-controller"), (
        f"the safe remedy must be offered before the DANGEROUS override: {detail}"
    )
    assert _tree_bytes(store_root) == before_store
    assert _tree_bytes(ckpt_dir) == before_ckpt
    assert not (step_dir / "reconcile.lock").exists()


def test_a_controller_the_probe_actually_saw_still_says_stop_the_job(tq_session, store_root, partition_id, tmp_path):
    """The negative control for the branch: the session's controller is really running in
    this process's Ray, so this is a sighting, and a sighting keeps the instruction that
    fits it. Also the place the ``controller_probe`` record is pinned for the in-process
    path, where there is no address to discover at all."""
    finish_group(tq_session, partition_id, "seen12", n=2, value=9)
    ckpt_dir, _ = _make_ckpt(tq_session, tmp_path, step=5)

    out = restore(store=store_root, journal_root=None, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)

    assert out["status"] == "live_controller_present", out
    assert out["controller"]["reachable"] is True, out["controller"]
    assert out["controller_probe"] == {"outcome": PROBE_IN_CALLER_RAY}, out["controller_probe"]
    detail = out["detail"]
    assert "a TransferQueue controller is reachable" in detail, detail
    assert "Stop the job" in detail, detail
    assert "could not be ruled out" not in detail, detail


def test_an_unresponsive_controller_is_a_sighting_and_an_undecided_probe_is_not():
    """The classification the two detail texts hang off, tested directly.

    A handle that was obtained and then did not answer is a live job (round 10's rule: a
    wedged controller is the partial-failure case the refusal exists for), so it keeps the
    sighting text. A probe that could not get an answer out of the actor name at all, and
    any shape this code does not recognise, is doubt: refused just the same, described
    honestly.
    """
    actor = {"actor": "TransferQueueController", "namespace": "transfer_queue"}
    wedged = actor | {"reachable": "unresponsive (GetTimeoutError: timed out)"}
    undecided = actor | {"reachable": "unknown (the controller exposes no get_config)"}

    assert restore_tool_mod._presence_is_a_sighting(wedged) is True
    assert restore_tool_mod._presence_is_a_sighting(undecided) is False
    assert restore_tool_mod._presence_is_a_sighting(actor) is False, "an unrecognised shape must not over-claim"

    probe = {"outcome": PROBE_IN_CALLER_RAY}
    wedged_detail = restore_tool_mod._live_controller_detail(wedged, probe)
    assert "a TransferQueue controller is reachable" in wedged_detail
    assert "Stop the job" in wedged_detail
    undecided_detail = restore_tool_mod._live_controller_detail(undecided, probe)
    assert "could not be ruled out" in undecided_detail
    assert "Stop the job" not in undecided_detail
    assert "ray list actors" in undecided_detail, "the remedy for actor-level doubt is on that cluster"


def test_a_result_decided_before_the_probe_says_so_instead_of_omitting_the_key(tmp_path):
    """``controller_probe`` joins the ONE stable result schema, so it is present and
    ``None`` for the statuses settled before the probe runs rather than missing from the
    JSON an automated wrapper parses."""
    out = restore(store=tmp_path / "store", journal_root=None, ckpt_dir=tmp_path / "empty", dry_run=True)
    assert out["status"] == "no_checkpoint", out
    assert "controller_probe" in out and out["controller_probe"] is None, out


# --------------------------------------------------------------------------------------
# 2. the journaling signal differs by wiring site
# --------------------------------------------------------------------------------------


def test_only_the_subclass_wiring_can_report_the_journaling_metric(tmp_path, caplog):
    """The README credited both wirings with ``durable_buffer/journaling_enabled = 0`` on
    a refused lease. An operator on the recommended (register) wiring who watches that key
    sees it absent, not zero: nothing in that path reports metrics, and with the lease
    refused there is not even a writer to ask for a summary. The warning and the ``None``
    are the signal there; the metric belongs to the sampler subclass, which is the only
    place verl merges this package's metrics from (pinned in round 11)."""
    from verl.experimental.durable_buffer import register

    root = tmp_path / "ring"
    holder = JournalWriter(root, interval_s=2.0, ring=2)
    try:
        with caplog.at_level(logging.WARNING):
            assert register.start_journal(str(root), interval_s=2.0) is None
        assert register.journal() is None, "no writer, so no summary() to report a metric from"
        assert "JOURNALING IS OFF" in caplog.text
    finally:
        register.stop_journal()
        holder.stop()

    package = Path(restore_tool_mod.__file__).parent
    assert ".summary()" not in (package / "register.py").read_text(), (
        "register.py grew a metrics path; the README says it has none"
    )
    assert ".summary()" in (package / "buffer.py").read_text(), (
        "the subclass stopped collecting the writer's summary, which is where the metric comes from"
    )
