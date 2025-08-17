import asyncio
import os
import time
import pytest

from typing import Any, Dict, List, Optional

from infrastructure.distributed_coordinator import DistributedCoordinator


class FakeBus:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.registered_workers: Dict[str, Dict[str, Any]] = {}
        self.partitions: Dict[str, List[str]] = {}
        self.published_events: List[Dict[str, Any]] = []
        self.failures: List[str] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def register_worker(self, worker_id: str, capabilities: Dict[str, Any]) -> None:
        self.registered_workers[worker_id] = capabilities

    async def create_partition(self, partition_id: str, agents: List[str]) -> None:
        self.partitions[partition_id] = agents

    async def publish_event(self, event_type: str, event_data: Dict[str, Any], target_partition: Optional[str]) -> None:  # noqa: D401
        # Record all published events for assertions
        self.published_events.append(
            {"type": event_type, "data": event_data, "target_partition": target_partition}
        )

    async def handle_worker_failure(self, worker_id: str) -> None:
        self.failures.append(worker_id)


class SimConfig:
    def __init__(self, tick_interval_seconds: float, coordinator_ack_timeout_seconds: float) -> None:
        self.tick_interval_seconds = tick_interval_seconds
        self.coordinator_ack_timeout_seconds = coordinator_ack_timeout_seconds


@pytest.mark.asyncio
async def test_tick_ack_timeout_removes_missing_worker_and_advances_tick():
    """
    Verify that when a worker fails to acknowledge a tick within the configured timeout,
    the coordinator:
      - marks the worker as failed (calls bus.handle_worker_failure)
      - removes the worker from the active registry
      - advances the global tick once remaining acks suffice
      - publishes GlobalTickAdvanceEvent
    """
    bus = FakeBus()
    # Very small intervals to keep test fast and deterministic
    cfg = SimConfig(tick_interval_seconds=0.02, coordinator_ack_timeout_seconds=0.2)

    coord = DistributedCoordinator(bus, cfg)
    await coord.start()

    # Spawn two workers across two partitions
    w1 = await coord.spawn_worker({"p1": ["a1"]})
    w2 = await coord.spawn_worker({"p2": ["a2"]})

    assert coord.current_global_tick == 0
    assert w1 in coord._workers and w2 in coord._workers

    # Only worker 1 acknowledges tick 0
    await coord._handle_tick_acknowledgement({"worker_id": w1, "tick_number": 0})

    # Wait longer than the ack timeout (+ small margin) to trigger removal of w2
    await asyncio.sleep(cfg.coordinator_ack_timeout_seconds + 0.3)

    # The coordinator should have removed the missing worker and advanced to tick 1
    assert w2 in bus.failures, "Missing worker should be reported to the bus"
    assert w2 not in coord._workers, "Missing worker should be removed from registry"
    assert coord.current_global_tick == 1, "Tick should advance after pruning missing worker"

    # Validate a GlobalTickAdvanceEvent was published with new_tick_number == 1
    advance_events = [
        e for e in bus.published_events
        if e["type"] == "GlobalTickAdvanceEvent" and e["data"].get("new_tick_number") == 1
    ]
    assert len(advance_events) >= 1, "GlobalTickAdvanceEvent should be published for tick 1"

    await coord.stop()


@pytest.mark.asyncio
async def test_tick_ack_timeout_uses_env_override(monkeypatch):
    """
    Verify the environment variable FBA_COORDINATOR_ACK_TIMEOUT_SECONDS overrides timeout
    when config does not specify coordinator_ack_timeout_seconds.
    """
    bus = FakeBus()

    # Config without coordinator_ack_timeout_seconds
    cfg = SimConfig(tick_interval_seconds=0.02, coordinator_ack_timeout_seconds=0.0)
    # Remove attribute so the code path uses env (simulate absence)
    delattr(cfg, "coordinator_ack_timeout_seconds")

    # Use env override to a very small timeout for the test
    monkeypatch.setenv("FBA_COORDINATOR_ACK_TIMEOUT_SECONDS", "0.15")

    coord = DistributedCoordinator(bus, cfg)
    await coord.start()

    w1 = await coord.spawn_worker({"p1": ["a1"]})
    w2 = await coord.spawn_worker({"p2": ["a2"]})
    assert coord.current_global_tick == 0

    # Only ack from w1
    await coord._handle_tick_acknowledgement({"worker_id": w1, "tick_number": 0})

    # Wait for > env timeout + small margin
    await asyncio.sleep(0.25)

    # Tick should have advanced after removing w2
    assert w2 in bus.failures
    assert coord.current_global_tick == 1

    await coord.stop()