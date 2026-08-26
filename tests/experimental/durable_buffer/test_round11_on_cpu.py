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
"""Regressions for the round-11 code review of the round-10 hardening.

Three defects, all of them cases where a mechanism was present and did less than the
text around it claimed:

1.  **The live-controller refusal could not fire on the CLI path.** The probe ran
    ``_ensure_isolated_ray()`` first, and actor names are scoped to a GCS, so
    ``ray.get_actor`` ran inside a private instance that holds nothing by construction:
    ``live_controller_present`` was unreachable for every CLI invocation and
    ``--allow-live-controller`` had nothing to override. The probe now runs before any
    isolation — in the caller's Ray when it has one, else in a temporary attachment to
    the cluster Ray's own discovery finds — and the isolation happens afterwards, for
    the reconcile. The store half of exclusivity is still unchecked, which the README
    now says.
2.  **The ring lease raised out of ``JournalWriter.__init__`` at both wiring sites.** A
    driver hard-killed inside ``max(30s, interval_s * 5)`` still holds a lease whose
    heartbeat looks fresh, so the crash restart this package exists to serve died at
    trainer init. Both wirings now degrade to journaling-off with one warning and a
    ``durable_buffer/journaling_enabled`` signal; ``JournalWriter`` itself still raises.
3.  **The schema fingerprint could not move when TransferQueue moved.** It was
    ``",".join(sorted(SNAPSHOT_ATTRIBUTES))`` over a tuple hardcoded in verl, identical
    either side of any TransferQueue upgrade, so a rename took the warn-and-proceed
    branch. The recorded value is now read back from the snapshot the tick actually
    wrote, and ``load_journal`` reads ``custom_meta`` by plain attribute access like the
    other four instead of absorbing its absence with a ``getattr`` default.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import finish_group
from omegaconf import OmegaConf
from test_round6_on_cpu import _make_ckpt, _tree_bytes

from verl.experimental.durable_buffer import restore_tool as restore_tool_mod
from verl.experimental.durable_buffer.buffer import DurableReplayBuffer
from verl.experimental.durable_buffer.fsck import (
    SNAPSHOT_ATTRIBUTES,
    UnrecoverableJournalError,
    check_journal_recoverable,
    check_snapshot_schema,
    load_journal,
    observed_snapshot_schema_fingerprint,
    snapshot_schema_fingerprint,
    snapshot_schema_fingerprint_of,
)
from verl.experimental.durable_buffer.journal import (
    STAMP_FILE,
    WRITER_LEASE_FILE,
    JournalWriter,
    JournalWriterLeaseError,
)
from verl.experimental.durable_buffer.restore_tool import TOOL_RAY_NAMESPACE, restore

# --------------------------------------------------------------------------------------
# 1. the exclusivity probe runs BEFORE the isolation
# --------------------------------------------------------------------------------------


def test_rays_own_discovery_finds_the_cluster_the_probe_has_to_reach(tq_session):
    """The CLI path depends entirely on this lookup, so pin it against a real cluster.

    The three sources are Ray's own (``RAY_ADDRESS``, ``/tmp/ray/ray_current_cluster``,
    every local ``gcs_server`` process). The last of those is what matters here: the
    session's Ray was brought up by a driver rather than by ``ray start``, so nothing
    wrote an address file, and a lookup that only read one would report no cluster while a
    controller was plainly running.

    Membership, not equality. The enumeration is host-global: another Ray instance on the
    same box (a concurrent suite, a leaked ``gcs_server``) adds an address and takes none
    away, so the property that has to hold is that the session's cluster is among them.
    Asserting it was the ONLY one is what made this test fail on a shared box while the
    code under it was working.
    """
    import ray

    assert ray.is_initialized(), "the session TQ must have brought Ray up"
    mine = ray.get_runtime_context().gcs_address
    found = restore_tool_mod._discoverable_cluster_addresses()
    assert isinstance(found, tuple), f"the probe iterates this, so it must be a sequence: {found!r}"
    assert len(found) == len(set(found)), f"the same cluster must not be probed twice: {found}"
    assert mine in found, f"discovery reported {found} while the running cluster is at {mine!r}"


def test_the_probe_does_not_isolate_before_it_looks(tq_session, monkeypatch):
    """The defect in one assertion: the probe used to open with ``_ensure_isolated_ray``,
    which on the CLI path creates the very Ray it then searches. Isolation must not be
    reachable from the probe at all."""

    def forbidden():  # pragma: no cover - reached only if the ordering regresses
        raise AssertionError("_live_controller must not isolate Ray before probing it")

    monkeypatch.setattr(restore_tool_mod, "_ensure_isolated_ray", forbidden)
    found = restore_tool_mod._live_controller()
    assert found is not None and found["reachable"] is True, found


def test_no_discoverable_cluster_means_no_controller_and_no_ray(monkeypatch):
    """With nothing to probe, the tool joins nothing: a Ray of its own comes later, in
    the reconcile, and only after the refusal has had its chance."""
    import ray

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(restore_tool_mod, "_discoverable_cluster_addresses", lambda: ())

    def forbidden(*args, **kwargs):  # pragma: no cover - reached only on a regression
        raise AssertionError("the probe must not initialize Ray when no cluster is discoverable")

    monkeypatch.setattr(ray, "init", forbidden)
    assert restore_tool_mod._live_controller() is None


def test_discovery_that_cannot_decide_counts_as_presence(tq_session, store_root, partition_id, tmp_path, monkeypatch):
    """FAIL CLOSED on doubt, applied to the new signal: if Ray's discovery itself breaks
    (a future refactor of the private helper), presence cannot be disproved, so it is
    assumed and reported with the reason. ``--allow-live-controller`` is the escape."""
    import ray

    finish_group(tq_session, partition_id, "doubt11", n=2, value=3)
    ckpt_dir, step_dir = _make_ckpt(tq_session, tmp_path, step=5)
    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)

    def broken():
        raise restore_tool_mod._ClusterDiscoveryFailed("RuntimeError: canonicalize_bootstrap_address moved")

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(restore_tool_mod, "_discoverable_cluster_addresses", broken)

    out = restore(store=store_root, journal_root=None, ckpt_dir=ckpt_dir, partition=partition_id, dry_run=True)
    assert out["status"] == "live_controller_present", out
    assert "discovery failed" in out["controller"]["reachable"]
    assert out["plan_available"] is False
    assert _tree_bytes(store_root) == before_store
    assert _tree_bytes(ckpt_dir) == before_ckpt
    assert not (step_dir / "reconcile.lock").exists()


def _subprocess_probe(script: str, extra_env: dict[str, str] | None = None, timeout: int = 900) -> dict:
    """Run ``script`` in a FRESH interpreter with no Ray and return its ``RESULT`` payload.

    A subprocess is the only honest way to exercise the uninitialized-Ray path: every
    other test in this package runs with the session's Ray up (``tq_session`` brings it
    up), which is exactly the branch the CLI never takes.
    """
    env = dict(os.environ)
    env.update(extra_env or {})
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=timeout)
    assert done.returncode == 0, f"probe failed:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    line = next((ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, f"probe printed no RESULT:\n{done.stdout[-3000:]}"
    return json.loads(line[len("RESULT ") :])


_ISOLATION_PROBE = """
import json
import ray
from verl.experimental.durable_buffer import restore_tool as rt

out = {"initialized_before": ray.is_initialized()}
out["started"] = rt._ensure_isolated_ray()
out["namespace"] = ray.get_runtime_context().namespace
out["gcs_address"] = ray.get_runtime_context().gcs_address
out["discoverable"] = rt._discoverable_cluster_addresses()
# The reason the probe cannot live here: this instance holds no actors, whatever the
# cluster is doing.
out["probe_inside_private"] = rt._probe_controller()
out["second_call"] = rt._ensure_isolated_ray()
ray.shutdown()
print("RESULT " + json.dumps(out, default=str))
"""


def test_the_cli_path_starts_a_private_ray_even_with_a_cluster_in_reach(tq_session):
    """The line that carries the whole CLI-path protection, exercised for real.

    ``ray.init(address="local", namespace=..., include_dashboard=False,
    log_to_driver=False)`` had no test at all: the only one asserted the no-op branch,
    and a wrong kwarg or a change to what ``address="local"`` means would have surfaced
    first in an outage. Run with a cluster discoverable AND ``RAY_ADDRESS`` set, because
    the claim is that neither of them can capture the tool: ``address="local"`` starts a
    new instance regardless (``RAY_ADDRESS`` overrides only ``None`` and ``"auto"``).
    """
    import ray

    ambient = ray.get_runtime_context().gcs_address
    got = _subprocess_probe(_ISOLATION_PROBE, {"RAY_ADDRESS": "127.0.0.1:1"})

    assert got["initialized_before"] is False, "the point of the subprocess is to have no Ray"
    assert got["started"] is True, "a private instance must be started when the tool has none"
    assert got["namespace"] == TOOL_RAY_NAMESPACE
    assert got["gcs_address"] and got["gcs_address"] != ambient, (
        f"the tool joined the ambient cluster {ambient} instead of starting its own ({got['gcs_address']})"
    )
    # Discovery reports RAY_ADDRESS among its candidates (canonicalized, so 127.0.0.1
    # comes back as the node's own IP); the port is what identifies it. The private
    # instance is on another one, and other local instances may be listed beside it.
    assert any(address.endswith(":1") for address in got["discoverable"]), got["discoverable"]
    assert got["probe_inside_private"] is None, (
        "a controller was visible inside the private instance, so this test no longer "
        "demonstrates why the probe must run before the isolation"
    )
    assert got["second_call"] is False, "an initialized Ray is left alone"


_REFUSAL_PROBE = """
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
out["exit_ok"] = rt._exit_ok(result)
out["initialized_after"] = ray.is_initialized()
print("RESULT " + json.dumps(out, default=str))
"""


def test_the_cli_path_refuses_a_controller_it_could_not_see_before(tq_session, store_root, partition_id, tmp_path):
    """The defect end to end, from the process shape an operator actually runs.

    A fresh interpreter with no Ray is the documented CLI path, and it is the path on
    which the refusal used to be unreachable: the probe created a private Ray and asked
    it about a controller living in another GCS. The session's TransferQueue controller
    is running here, so this must now refuse — and the refused report must name the
    cluster it found the controller in, which is the proof that the probe ran somewhere
    it could see one. Ray is left uninitialized in the child, because the refusal
    precedes the reconcile and the temporary attachment is detached again.

    ``RAY_ADDRESS`` is pinned to this session's cluster so the address asserted below is
    the one probed first. The child inherits pytest's environment, where nothing sets that
    variable, and the enumeration then orders local instances by ``sorted(ip:port)``, which
    is the digits of a randomly assigned GCS port. A second concurrent run of this suite
    holds a controller of its own, under the same name in another GCS, and the probe reports
    the FIRST sighting: without the pin, half of those runs would report the other suite's
    address and this equality would be a property of the host rather than of the tool.
    """
    import ray

    finish_group(tq_session, partition_id, "cli11", n=2, value=6)
    ckpt_dir, step_dir = _make_ckpt(tq_session, tmp_path, step=5)
    before_store, before_ckpt = _tree_bytes(store_root), _tree_bytes(ckpt_dir)
    ambient = ray.get_runtime_context().gcs_address

    got = _subprocess_probe(
        _REFUSAL_PROBE,
        {
            "PROBE_STORE": str(store_root),
            "PROBE_CKPT": str(ckpt_dir),
            "PROBE_PARTITION": partition_id,
            "RAY_ADDRESS": ambient,
        },
    )

    assert got["initialized_before"] is False
    assert got["status"] == "live_controller_present", got
    assert got["plan_available"] is False, "the refusal precedes the preflight"
    assert got["controller"]["reachable"] is True, got["controller"]
    assert got["controller"]["cluster"] == ambient, (
        f"the probe reported the controller in {got['controller'].get('cluster')!r}, not in the "
        f"running cluster {ambient!r}"
    )
    assert got["exit_ok"] is False
    assert got["initialized_after"] is False, "the probe must detach the cluster it joined"
    assert _tree_bytes(store_root) == before_store, "a refused restore touched the store"
    assert _tree_bytes(ckpt_dir) == before_ckpt, "a refused restore touched the checkpoint"
    assert not (step_dir / "reconcile.lock").exists(), "the refusal must precede even the lock"


# --------------------------------------------------------------------------------------
# 2. a refused ring lease turns journaling off, not the run
# --------------------------------------------------------------------------------------


def test_the_writer_itself_still_raises_on_a_held_lease(tmp_path):
    """The guard belongs at the wiring sites, not in the class: a direct constructor call
    is a caller asking for a journal writer on this root, and there is not one to give."""
    root = tmp_path / "ring"
    holder = JournalWriter(root, interval_s=2.0, ring=2)
    try:
        with pytest.raises(JournalWriterLeaseError, match="another journal writer holds"):
            JournalWriter(root, interval_s=2.0, ring=2)
    finally:
        holder.stop()


def test_start_journal_degrades_to_journaling_off_when_the_lease_is_held(tmp_path, caplog):
    """``register.start_journal`` constructed the writer inline, so the lease error
    propagated through trainer init and killed the resumed run — inside a window of up to
    ``max(30s, interval_s * 5)`` after a hard kill, which is the restart this package
    exists to serve. Journaling is what degrades."""
    from verl.experimental.durable_buffer import register

    root = tmp_path / "ring"
    holder = JournalWriter(root, interval_s=2.0, ring=2)
    try:
        with caplog.at_level(logging.WARNING):
            writer = register.start_journal(str(root), interval_s=2.0)
        assert writer is None, "a held lease must not raise, and must not hand back a writer"
        assert register.journal() is None
        assert "JOURNALING IS OFF" in caplog.text
        assert caplog.text.count("JOURNALING IS OFF") == 1, "one clear warning, not a stack trace per tick"
        assert holder._holds_lease, "the real writer keeps its lease"
    finally:
        register.stop_journal()
        holder.stop()


def test_start_journal_still_works_once_the_lease_is_free(tmp_path):
    """The degrade must not be sticky: the next start after the holder releases journals
    normally (a restart outside the stale window, or a second attempt after a takeover)."""
    from verl.experimental.durable_buffer import register

    root = tmp_path / "ring"
    holder = JournalWriter(root, interval_s=2.0, ring=2)
    holder.stop()
    try:
        writer = register.start_journal(str(root), interval_s=3600.0)
        assert writer is not None and writer.summary()["journaling_enabled"] is True
    finally:
        register.stop_journal()


def _buffer_kwargs(**durable):
    return dict(
        trainer_mode="separate_async",
        trainer_config=OmegaConf.create({}),
        max_off_policy_threshold=10**6,
        max_off_policy_strategy="drop",
        sampler_kwargs=OmegaConf.create({"durable_buffer": durable}),
        poll_interval=0.05,
    )


class _FakeBatch:
    keys = ["k0"]


def _stub_sample_internals(rb, monkeypatch):
    """Stub what ``ReplayBufferAsync.sample``'s body calls, on the instance, so the real
    override, the real peel and the real metrics merge stay in the call path."""
    monkeypatch.setattr(rb, "wait_for_sampleable", lambda *a, **k: (["k0"], {}))
    monkeypatch.setattr(rb, "_select_prompt_uids", lambda *a, **k: (["u0"], {}, {}))
    monkeypatch.setattr(rb, "_materialize_batch", lambda *a, **k: _FakeBatch())


def test_the_buffer_degrades_to_journaling_off_when_the_lease_is_held(tmp_path, caplog, monkeypatch):
    """The other documented wiring, with the same defect: ``DurableReplayBuffer.__init__``
    constructed the writer inline, so a fast restart raised out of the sampler's
    constructor. The buffer is now built and usable, and it says journaling is off in the
    metrics rather than only in a log line nobody reads until the postmortem."""
    from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

    monkeypatch.setattr(ReplayBufferAsync, "_sync_metadata_from_transfer_queue", lambda self: None)
    root = tmp_path / "ring"
    holder = JournalWriter(root, interval_s=2.0, ring=2)
    try:
        with caplog.at_level(logging.WARNING):
            rb = DurableReplayBuffer(refill_fn=None, **_buffer_kwargs(journal_root=str(root), background=False))
        assert rb._journal is None, "a held lease must leave the buffer running without a journal"
        assert "JOURNALING IS OFF" in caplog.text
        rb._sync_metadata_from_transfer_queue()  # must not raise with no journal
        _stub_sample_internals(rb, monkeypatch)
        _batch, metrics = rb.sample(3, "train", 1)
        assert metrics["durable_buffer/journaling_enabled"] is False
    finally:
        holder.stop()


def test_a_journaling_buffer_reports_the_signal_as_on(tmp_path, monkeypatch):
    """The negative control for the metric: with a journal running it reads True, so a 0
    on a dashboard means something."""
    from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

    monkeypatch.setattr(ReplayBufferAsync, "_sync_metadata_from_transfer_queue", lambda self: None)
    rb = DurableReplayBuffer(refill_fn=None, **_buffer_kwargs(journal_root=str(tmp_path / "ring"), background=False))
    try:
        assert rb._journal is not None
        assert rb._journal.summary()["journaling_enabled"] is True
        _stub_sample_internals(rb, monkeypatch)
        _batch, metrics = rb.sample(3, "train", 1)
        assert metrics["durable_buffer/journaling_enabled"] is True
    finally:
        rb._journal.stop()


# --------------------------------------------------------------------------------------
# 3. the schema fingerprint is read off the snapshot the tick wrote
# --------------------------------------------------------------------------------------


def _one_slot(tmp_path: Path, tag: str) -> tuple[Path, JournalWriter]:
    writer = JournalWriter(tmp_path / f"ring-{tag}", interval_s=0.0, ring=2)
    slot = Path(writer.tick()["dir"])
    writer.stop()
    return slot, writer


def _rewrite_partition(slot: Path, partition: str, mutate) -> None:
    """Apply ``mutate`` to a slot's partition snapshot object and re-stamp the slot the
    way a tick would, so the fingerprint recorded is the one this snapshot deserves."""
    state = pickle.loads((slot / "controller_state.pkl").read_bytes())
    mutate(state["partitions"][partition])
    (slot / "controller_state.pkl").write_bytes(pickle.dumps(state))
    stamp = json.loads((slot / STAMP_FILE).read_text())
    stamp["snapshot_schema"] = snapshot_schema_fingerprint_of(slot)
    (slot / STAMP_FILE).write_text(json.dumps(stamp))


def test_the_recorded_fingerprint_is_read_off_the_snapshot_not_off_a_tuple(tq_session, tmp_path, partition_id):
    """The mechanism the claim rests on. A fingerprint derived from
    ``SNAPSHOT_ATTRIBUTES`` alone has the same value either side of any TransferQueue
    upgrade, so it could only ever detect a change to verl's own reader contract. The
    recorded value now comes from the controller pickle the tick wrote."""
    finish_group(tq_session, partition_id, "fp11", n=2, value=2)
    slot, _writer = _one_slot(tmp_path, "fp")

    stamp = json.loads((slot / STAMP_FILE).read_text())
    assert stamp["snapshot_schema"] == snapshot_schema_fingerprint_of(slot), (
        "the stamp does not describe the snapshot in its own slot"
    )
    # With an intact TransferQueue the observed set IS the reader contract, which is why
    # every other journal in the suite still validates.
    assert stamp["snapshot_schema"] == snapshot_schema_fingerprint()
    state = pickle.loads((slot / "controller_state.pkl").read_bytes())
    assert observed_snapshot_schema_fingerprint(state["partitions"].values()) == stamp["snapshot_schema"]

    # And what the writer would record for a snapshot that carries LESS, which is the
    # assertion a revert to the hardcoded tuple fails: an intact TransferQueue makes the
    # two agree, so only a drifted snapshot tells them apart.
    del state["partitions"][partition_id].custom_meta
    (slot / "controller_state.pkl").write_bytes(pickle.dumps(state))
    reader = JournalWriter(tmp_path / "ring-fp-reader", interval_s=0.0, ring=2)
    try:
        assert reader._snapshot_schema(slot) == ",".join(sorted(set(SNAPSHOT_ATTRIBUTES) - {"custom_meta"}))
    finally:
        reader.stop()


def test_a_transfer_queue_side_rename_moves_the_fingerprint_and_is_refused(tq_session, tmp_path, partition_id):
    """The case the pin exists for and could not catch: TransferQueue renames an attribute
    reconciliation reads, so a snapshot taken afterwards no longer carries it. The
    attribute drops out of the tick-time fingerprint, the mismatch is refused fail-closed
    at restore, and the message names the attribute that moved."""
    finish_group(tq_session, partition_id, "rename11", n=2, value=4)
    slot, _writer = _one_slot(tmp_path, "rename")

    def rename_keys_mapping(snapshot):
        snapshot.key_map = snapshot.keys_mapping  # the shape an upstream rename leaves
        del snapshot.keys_mapping

    _rewrite_partition(slot, partition_id, rename_keys_mapping)

    recorded = json.loads((slot / STAMP_FILE).read_text())["snapshot_schema"]
    assert "keys_mapping" not in recorded.split(","), recorded
    assert recorded != snapshot_schema_fingerprint(), "a renamed attribute must move the fingerprint"

    with pytest.raises(UnrecoverableJournalError, match="different TransferQueue controller schema") as exc:
        check_snapshot_schema(slot)
    assert "keys_mapping" in str(exc.value), "the message must name the field sets it compared"
    # And the gate every consumer inherits refuses the slot, not just the schema check.
    with pytest.raises(UnrecoverableJournalError):
        check_journal_recoverable(slot)


def test_a_rename_that_reached_the_parse_is_unrecoverable_not_a_traceback(tq_session, tmp_path, partition_id):
    """The second half of the same drift, on the reader side: with the stamp gone (a
    pre-pin journal) the parse is what meets the renamed attribute, and it must classify
    fail-closed like every other untrusted artifact."""
    finish_group(tq_session, partition_id, "parse11", n=2, value=5)
    slot, _writer = _one_slot(tmp_path, "parse")

    def rename_keys_mapping(snapshot):
        snapshot.key_map = snapshot.keys_mapping
        del snapshot.keys_mapping

    _rewrite_partition(slot, partition_id, rename_keys_mapping)
    stamp = json.loads((slot / STAMP_FILE).read_text())
    stamp.pop("snapshot_schema")
    (slot / STAMP_FILE).write_text(json.dumps(stamp))

    with pytest.raises(UnrecoverableJournalError, match="cannot be parsed into the reconciliation view"):
        load_journal(slot, partition_id)


def test_a_snapshot_missing_custom_meta_is_refused_like_the_other_four(tq_session, tmp_path, partition_id):
    """``custom_meta`` was read as ``getattr(mine, "custom_meta", {})`` while being listed
    in ``SNAPSHOT_ATTRIBUTES`` as an attribute reconciliation reads, so a rename of that
    one field was absorbed silently — the "readable with drifted semantics" case the
    finding was about. It is now read like the rest."""
    finish_group(tq_session, partition_id, "meta11", n=2, value=6)
    slot, _writer = _one_slot(tmp_path, "meta")

    def drop_custom_meta(snapshot):
        del snapshot.custom_meta

    _rewrite_partition(slot, partition_id, drop_custom_meta)

    recorded = json.loads((slot / STAMP_FILE).read_text())["snapshot_schema"]
    assert "custom_meta" not in recorded.split(","), recorded
    with pytest.raises(UnrecoverableJournalError, match="cannot be parsed into the reconciliation view"):
        load_journal(slot, partition_id)


def test_the_observed_fingerprint_needs_every_partition_to_carry_a_name(tq_session, tmp_path, partition_id):
    """``load_journal`` walks every partition, so a name counts as present only when they
    all have it, and nothing to observe records the reader contract unchanged (such a
    snapshot has no partitions and is refused on those terms instead)."""

    class Full:
        def __init__(self):
            for name in SNAPSHOT_ATTRIBUTES:
                setattr(self, name, {})

    class Partial:
        def __init__(self):
            for name in SNAPSHOT_ATTRIBUTES:
                if name != "custom_meta":
                    setattr(self, name, {})

    assert observed_snapshot_schema_fingerprint([Full()]) == snapshot_schema_fingerprint()
    assert observed_snapshot_schema_fingerprint([Full(), Partial()]) == ",".join(
        sorted(set(SNAPSHOT_ATTRIBUTES) - {"custom_meta"})
    )
    assert observed_snapshot_schema_fingerprint([]) == snapshot_schema_fingerprint()


def test_the_fingerprint_is_observed_once_per_writer(tq_session, tmp_path, partition_id, monkeypatch):
    """The read-back unpickles a snapshot that is O(live metadata), and the shape it
    reports comes from the TransferQueue classes loaded in this process, which cannot
    change while it runs. So it is paid once, and every slot the writer publishes carries
    it."""
    # The package exports a function named ``fsck``, which shadows the submodule
    # attribute, so reach the module itself rather than through the package.
    fsck_mod = importlib.import_module("verl.experimental.durable_buffer.fsck")
    finish_group(tq_session, partition_id, "cache11", n=2, value=7)
    reads: list[str] = []
    real = fsck_mod.snapshot_schema_fingerprint_of

    def counting(journal_dir):
        reads.append(str(journal_dir))
        return real(journal_dir)

    monkeypatch.setattr(fsck_mod, "snapshot_schema_fingerprint_of", counting)
    writer = JournalWriter(tmp_path / "ring-cache", interval_s=0.0, ring=2)
    try:
        slots = [Path(writer.tick()["dir"]) for _ in range(3)]
    finally:
        writer.stop()

    assert len(reads) == 1, f"the snapshot schema was read back {len(reads)} times"
    recorded = {json.loads((slot / STAMP_FILE).read_text())["snapshot_schema"] for slot in slots}
    assert recorded == {snapshot_schema_fingerprint()}, recorded


def test_a_writer_whose_snapshot_cannot_be_read_back_publishes_no_slot(tq_session, tmp_path, partition_id):
    """Fail closed, consistent with the offsets pre-capture: a controller pickle the
    WRITING process cannot parse is a slot that would poison recovery, so the tick fails
    (no stamp, ``latest_journal`` skips it) instead of publishing an unpinned one."""
    finish_group(tq_session, partition_id, "unread11", n=2, value=8)
    writer = JournalWriter(tmp_path / "ring-unread", interval_s=0.0, ring=2)
    writer._snapshot_schema_fp = None
    original = writer._snapshot_schema

    def broken(target):
        (target / "controller_state.pkl").write_bytes(b"not a pickle")
        return original(target)

    writer._snapshot_schema = broken
    try:
        assert writer.tick() is None, "a slot whose schema cannot be established must not be published"
        assert writer.stats["failures"] == 1
        assert not (Path(writer.root) / "journal-000" / STAMP_FILE).exists()
    finally:
        writer.stop()
    assert not (tmp_path / "ring-unread" / WRITER_LEASE_FILE).exists()
