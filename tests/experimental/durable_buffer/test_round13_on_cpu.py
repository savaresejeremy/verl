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
"""Regressions for the round-13 code review of the round-12 exclusivity probe.

One defect, in the direction that matters. The probe resolved ONE cluster address
(``canonicalize_bootstrap_address("auto")``), whose last resolution step picks an
arbitrary local ``gcs_server`` process and whose first step is ``RAY_ADDRESS``. On a host
running a second Ray instance, or carrying a pointer to a cluster that is gone, the probe
therefore asked a GCS that was not the run's, found no ``TransferQueueController`` there,
and returned the intended-path answer: absent. Not doubt, which is refused, but a
confident absence, which lets the reconcile proceed and delete store objects out from
under a live job. The tool now enumerates every locally discoverable instance
(``RAY_ADDRESS``, ``/tmp/ray/ray_current_cluster``, and every local ``gcs_server``),
probes each in turn, and refuses if any of them holds the controller. What is left is the
boundary the RFC and the README both state: a cluster this host cannot discover or reach.

The enumeration also has to leave Ray's ``auto`` alone. It is not an address but a request
to run that same single-address resolution, which raises when there is nothing to resolve,
so passing it through turned ``auto`` exported on a host whose cluster is gone into a
discovery failure, and a failure to decide is refused. That is the post-crash input the
tool exists to serve, so it is enumerated as no cluster and proceeds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from helpers import finish_group
from test_round6_on_cpu import _make_ckpt, _tree_bytes

from verl.experimental.durable_buffer import restore_tool as restore_tool_mod
from verl.experimental.durable_buffer.restore_tool import (
    PROBE_ADDRESS_IS_DEAD,
    PROBE_CLUSTER_UNJOINABLE,
    PROBE_DISCOVERY_FAILED,
    PROBE_IN_DISCOVERED_CLUSTER,
    PROBE_NO_CLUSTER,
)

#: A port nothing serves, used as the stale pointer. Port 1 is privileged and unused, so a
#: connection to it is refused rather than accepted by something unrelated.
DEAD_POINTER = "127.0.0.1:1"


def test_discovery_enumerates_every_local_instance_not_just_the_first(tq_session, monkeypatch):
    """The mechanism of the defect, pinned from both sides.

    Ray's own resolution returns a single address and stops at the first source that has
    one, so with ``RAY_ADDRESS`` set it never looks at the local ``gcs_server`` the
    session's controller actually lives in. The tool's enumeration takes the union, so the
    session's cluster is in the list even when a pointer to somewhere else is also set.
    Membership rather than equality, because a concurrent suite on the same box adds
    addresses and takes none away.
    """
    import ray
    from ray._private import services

    monkeypatch.setenv("RAY_ADDRESS", DEAD_POINTER)
    mine = ray.get_runtime_context().gcs_address
    decoy = services.canonicalize_bootstrap_address(DEAD_POINTER)

    single = services.canonicalize_bootstrap_address("auto")
    assert single == decoy, (
        f"Ray resolved {single!r} rather than the pointer; the premise of this test is that one "
        "address hides the others"
    )
    assert single != mine, "the decoy must not be the session's own cluster"

    found = restore_tool_mod._discoverable_cluster_addresses()
    assert decoy in found, f"the pointer Ray would have used is missing from {found}"
    assert mine in found, f"the cluster holding the controller is missing from {found}"
    assert len(found) == len(set(found)), f"the same cluster must not be probed twice: {found}"


def test_nothing_discoverable_is_still_an_empty_enumeration(monkeypatch):
    """The common case, and the one that must not raise. With no address file entry and no
    local ``gcs_server``, the enumeration is empty and the caller proceeds to its own
    isolated Ray. Run with ``RAY_ADDRESS=auto``, which is the input Ray's single-address
    helper raises ``ConnectionError`` on: the enumeration must not need that helper, and it
    does not, because ``auto`` is a request to resolve the same two sources this function
    reads directly and is skipped like ``local``. Asserted for an unset pointer too, which
    is the same answer by a shorter route.
    """
    from ray._private import services

    monkeypatch.setattr(services, "find_bootstrap_address", lambda _temp_dir: None)
    monkeypatch.setattr(services, "find_gcs_addresses", lambda: frozenset())

    monkeypatch.setenv("RAY_ADDRESS", "auto")
    assert restore_tool_mod._discoverable_cluster_addresses() == ()

    monkeypatch.delenv("RAY_ADDRESS")
    assert restore_tool_mod._discoverable_cluster_addresses() == ()


def test_auto_with_nothing_discoverable_proceeds_instead_of_refusing(monkeypatch):
    """``RAY_ADDRESS=auto`` with the cluster gone, which is the shape the tool exists for.

    ``auto`` is Ray's own idiom for "attach to the local cluster" and is routinely exported
    on a head node, so it outlives the crash that took the cluster with it. Resolving it
    re-enters Ray's resolution, which raises ``ConnectionError`` when there is nothing to
    find, and a broken enumeration reads as doubt: the operator would be told to repair
    Ray's discovery before restoring the checkpoint the crash made necessary. Nothing
    discoverable is the answer no cluster, and no cluster is a proceed.

    Asserted at :func:`_live_controller`, which is the decision point ``restore`` branches
    on (a non-``None`` return is the refusal), so no Ray is started here and ``ray.init``
    is forbidden outright.
    """
    import ray
    from ray._private import services

    def forbidden(*args, **kwargs):  # pragma: no cover - reached only on a regression
        raise AssertionError("nothing was discovered, so there is nothing to join")

    monkeypatch.setenv("RAY_ADDRESS", "auto")
    monkeypatch.setattr(services, "find_bootstrap_address", lambda _temp_dir: None)
    monkeypatch.setattr(services, "find_gcs_addresses", lambda: frozenset())
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", forbidden)

    record: dict = {}
    assert restore_tool_mod._live_controller(probe_record=record) is None, (
        "an exported `auto` and nothing to resolve it to is no cluster, which is a proceed"
    )
    assert record["outcome"] == PROBE_NO_CLUSTER, record
    assert record["discovered"] == [], record


def test_a_broken_discovery_names_the_pointer_to_clear(monkeypatch):
    """The refusal for a discovery that broke has to name an action that can be taken.

    It named only ``ray status``, which after a total crash keeps failing, so the report
    pointed at a state that never arrives. The action that does clear it, dropping the
    pointer discovery reads, was printed only on the branch that had an address to name.
    """
    import ray

    def broken():
        raise restore_tool_mod._ClusterDiscoveryFailed("RuntimeError: find_gcs_addresses moved")

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(restore_tool_mod, "_discoverable_cluster_addresses", broken)

    record: dict = {}
    found = restore_tool_mod._live_controller(probe_record=record)
    assert found is not None, "a discovery that cannot be trusted is doubt, and doubt is presence"
    assert record["outcome"] == PROBE_DISCOVERY_FAILED, record

    detail = restore_tool_mod._live_controller_detail(found, record)
    assert "unset a stale RAY_ADDRESS" in detail, detail
    assert "delete the address file" in detail, detail
    assert "Stop the job" not in detail, "no controller was seen, so there is no job to stop"


def test_a_broken_enumeration_is_doubt_and_not_an_empty_list(monkeypatch):
    """Fail closed on a discovery that moved: the private helpers this reads are Ray's, so
    a refactor there must surface as doubt (refused, with the reason) rather than as
    nothing to probe."""
    from ray._private import services

    def moved():
        raise RuntimeError("find_gcs_addresses moved")

    monkeypatch.setattr(services, "find_gcs_addresses", moved)
    try:
        restore_tool_mod._discoverable_cluster_addresses()
    except restore_tool_mod._ClusterDiscoveryFailed as exc:
        assert "find_gcs_addresses moved" in str(exc), exc
    else:
        raise AssertionError("a broken enumeration must not report an empty one")


def test_each_instance_is_detached_before_the_next_is_joined(monkeypatch):
    """The loop's one hard requirement: ``ray.init`` raises if Ray is already initialized,
    so a probe that forgot to detach would decide the first candidate and report the rest
    as unjoinable. Pinned as an ordering, with no real cluster involved."""
    import ray

    calls: list[str] = []
    addresses = ("10.255.255.1:6379", "10.255.255.2:6379")

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(restore_tool_mod, "_discoverable_cluster_addresses", lambda: addresses)
    monkeypatch.setattr(restore_tool_mod, "_address_answers", lambda _address: True)
    monkeypatch.setattr(ray, "init", lambda address, **kwargs: calls.append(f"init {address}"))
    monkeypatch.setattr(ray, "shutdown", lambda: calls.append("shutdown"))
    monkeypatch.setattr(restore_tool_mod, "_probe_controller", lambda: calls.append("probe"))

    record: dict = {}
    assert restore_tool_mod._live_controller(probe_record=record) is None
    assert calls == [
        f"init {addresses[0]}",
        "probe",
        "shutdown",
        f"init {addresses[1]}",
        "probe",
        "shutdown",
    ], calls
    assert record["outcome"] == PROBE_IN_DISCOVERED_CLUSTER, record
    assert record["discovered"] == list(addresses), record
    assert record["answered"] == dict.fromkeys(addresses, True), record


def test_a_sighting_anywhere_stops_the_search_and_names_its_cluster(monkeypatch):
    """One instance holding the controller is one live run, so the first sighting is the
    answer and the remaining candidates are not joined. The refusal has to name the cluster
    it saw, because that is the one the operator has to go and stop."""
    import ray

    holder, never = "10.255.255.1:6379", "10.255.255.2:6379"
    joined: list[str] = []
    seen = {"actor": "TransferQueueController", "namespace": "transfer_queue", "reachable": True}

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(restore_tool_mod, "_discoverable_cluster_addresses", lambda: (holder, never))
    monkeypatch.setattr(restore_tool_mod, "_address_answers", lambda _address: True)
    monkeypatch.setattr(ray, "init", lambda address, **kwargs: joined.append(address))
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(restore_tool_mod, "_probe_controller", lambda: dict(seen))

    record: dict = {}
    found = restore_tool_mod._live_controller(probe_record=record)
    assert found is not None and found["reachable"] is True, found
    assert found["cluster"] == holder, found
    assert joined == [holder], f"the search continued past a sighting: {joined}"
    assert record["outcome"] == PROBE_IN_DISCOVERED_CLUSTER, record


def test_doubt_about_one_instance_outranks_the_ones_that_came_back_clean(monkeypatch):
    """An instance that answers a TCP connection and then cannot be joined could be holding
    the controller, and no other instance's clean answer disproves it. So the aggregate
    verdict is doubt, refused, with every unjoined address named: the remedy is per-pointer
    and an operator cannot act on a count."""
    import ray

    dead, first, second = DEAD_POINTER, "10.255.255.1:6379", "10.255.255.2:6379"

    def unjoinable(address, **kwargs):
        raise ConnectionError(f"Failed to connect to Ray cluster at {address}")

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(restore_tool_mod, "_discoverable_cluster_addresses", lambda: (dead, first, second))
    monkeypatch.setattr(restore_tool_mod, "_address_answers", lambda address: address != dead)
    monkeypatch.setattr(ray, "init", unjoinable)

    record: dict = {}
    found = restore_tool_mod._live_controller(probe_record=record)
    assert found is not None, "an unjoined cluster is doubt, and doubt counts as presence"
    assert record["outcome"] == PROBE_CLUSTER_UNJOINABLE, record
    assert record["answered"] == {dead: False, first: True, second: True}, record
    assert first in found["reachable"] and second in found["reachable"], found
    assert "clusters" in found["reachable"], "two of them, so the plural"
    detail = restore_tool_mod._live_controller_detail(found, record)
    assert dead in detail and first in detail and second in detail, detail


def test_all_dead_pointers_still_read_as_no_cluster(monkeypatch):
    """The case the TCP pre-check exists for, now with more than one pointer. Every
    discovered address refusing a connection is the post-crash state, and it must stay a
    proceed rather than becoming a refusal: the operator is running this tool BECAUSE the
    cluster is gone."""
    import ray

    addresses = (DEAD_POINTER, "127.0.0.1:2")

    def forbidden(*args, **kwargs):  # pragma: no cover - reached only on a regression
        raise AssertionError("a dead pointer must never be joined")

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(restore_tool_mod, "_discoverable_cluster_addresses", lambda: addresses)
    monkeypatch.setattr(ray, "init", forbidden)

    record: dict = {}
    assert restore_tool_mod._live_controller(probe_record=record) is None
    assert record["outcome"] == PROBE_ADDRESS_IS_DEAD, record
    assert record["answered"] == dict.fromkeys(addresses, False), record


_FOREIGN_POINTER_PROBE = """
import json
import os
from pathlib import Path

import ray
from verl.experimental.durable_buffer import restore_tool as rt

out = {"initialized_before": ray.is_initialized()}
result = rt.restore(
    store=Path(os.environ["PROBE_STORE"]),
    journal_root=None,
    ckpt_dir=Path(os.environ["PROBE_CKPT"]),
    partition=os.environ["PROBE_PARTITION"],
    dry_run=True,
)
out["status"] = result["status"]
out["plan_available"] = result["plan_available"]
out["controller"] = result.get("controller")
out["controller_probe"] = result["controller_probe"]
out["detail"] = result["detail"]
out["initialized_after"] = ray.is_initialized()
print("RESULT " + json.dumps(out, default=str))
"""


def test_a_stale_pointer_does_not_hide_a_local_controller(tq_session, store_root, partition_id, tmp_path):
    """The defect end to end, in the process shape an operator runs and on the arrangement
    that produced it: a pointer to a cluster that is gone, and the run's cluster up on the
    same host.

    ``RAY_ADDRESS`` is the first thing Ray's resolution reads, so the old probe canonicalized
    it, found nothing listening, and reported no live cluster while the session's
    TransferQueue controller was running one local ``gcs_server`` away. A reconcile then
    deleted objects under it. The enumeration reaches both, and the dead one no longer
    decides the answer.

    The cluster the sighting is reported in is asserted by MEMBERSHIP of what discovery
    named, not by equality with the session's cluster. This test's premise is a foreign
    ``RAY_ADDRESS``, so the leading candidate cannot be pinned to the session's cluster the
    way the round-11 refusal test pins it, and the order among local instances is then the
    digits of a randomly assigned GCS port. A second concurrent run of this suite holds a
    controller under the same name in its own GCS and would be sighted first about half the
    time. What carries the regression is host-independent: the enumeration reached past the
    dead pointer, the session's cluster is among the addresses it named, the sighting is in
    a cluster that answered, and the pointer itself answered nothing.
    """
    import ray

    finish_group(tq_session, partition_id, "stale13", n=2, value=4)
    ckpt_dir, step_dir = _make_ckpt(tq_session, tmp_path, step=5)
    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)
    ambient = ray.get_runtime_context().gcs_address

    env = dict(os.environ)
    env.update(
        RAY_ADDRESS=DEAD_POINTER,
        PROBE_STORE=str(store_root),
        PROBE_CKPT=str(ckpt_dir),
        PROBE_PARTITION=partition_id,
    )
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", _FOREIGN_POINTER_PROBE], capture_output=True, text=True, env=env, timeout=600
    )
    assert done.returncode == 0, f"probe failed:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    line = next((ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, f"probe printed no RESULT:\n{done.stdout[-3000:]}"
    got = json.loads(line[len("RESULT ") :])

    assert got["initialized_before"] is False, "the point of the subprocess is to have no Ray"
    assert got["status"] == "live_controller_present", got
    assert got["plan_available"] is False, "the refusal precedes the preflight"
    assert got["controller"]["reachable"] is True, got["controller"]
    probe = got["controller_probe"]
    assert probe["outcome"] == PROBE_IN_DISCOVERED_CLUSTER, probe
    assert ambient in probe["discovered"], f"the live cluster was never enumerated: {probe}"
    assert any(address.endswith(":1") for address in probe["discovered"]), probe
    reported = got["controller"]["cluster"]
    assert reported in probe["discovered"], (
        f"the probe reported the controller in {reported!r}, which discovery never named: {probe}"
    )
    assert probe["answered"][reported] is True, f"the cluster the sighting is reported in answered: {probe}"
    # `is not False` rather than `is True`: the loop returns on the FIRST sighting, so a
    # second concurrent suite whose controller is probed earlier leaves this key absent.
    # Nothing can make it False, because the session's cluster is up and answering.
    assert probe["answered"].get(ambient) is not False, f"the live cluster read as a dead pointer: {probe}"
    assert any(answered is False for answered in probe["answered"].values()), (
        f"the stale pointer must be recorded as answering nothing: {probe}"
    )
    assert "Stop the job" in got["detail"], got["detail"]
    assert got["initialized_after"] is False, "the probe must detach the cluster it joined"
    assert _tree_bytes(store_root) == before_store, "a refused restore touched the store"
    assert _tree_bytes(ckpt_dir) == before_ckpt, "a refused restore touched the checkpoint"
    assert not (step_dir / "reconcile.lock").exists(), "the refusal must precede even the lock"
    assert time.monotonic() - started < 300, "the dead pointer must not have been joined"


def test_the_docs_name_the_residual_the_enumeration_leaves():
    """The boundary is disclosed in three places that have to agree: the module docstring,
    the CLI help for the override, and the README. The enumeration is local, so what it
    cannot see is a cluster this host cannot discover or reach, and none of the three may
    claim otherwise.

    Whitespace is collapsed before matching. Both documents are wrapped for reading, so
    where a sentence happens to break is not a property worth asserting on.
    """

    def flat(text: str) -> str:
        return " ".join(text.split())

    package = Path(restore_tool_mod.__file__).parent
    source = flat(Path(restore_tool_mod.__file__).read_text(encoding="utf-8"))
    readme = flat((package / "README.md").read_text(encoding="utf-8"))

    assert "EVERY Ray instance this host can discover" in source, "the CLI help must state the enumeration"
    assert "every Ray instance this host can discover" in readme, "the README must state the enumeration"
    for text, where in ((source, "the module docstring"), (readme, "the README")):
        assert "cannot discover" in text, f"{where} must name the undiscoverable-cluster residual"
        assert "no store-level writer lease" in text, f"{where} must keep the store-exclusivity limit"
