import asyncio
from pathlib import Path
import pytest

from infrastructure.deployment import DeploymentManager


@pytest.mark.asyncio
async def test_deployment_lifecycle(tmp_path):
    dm = DeploymentManager()  # base_dir defaults to repo root (pytest cwd)

    config = {
        "name": "unit-test-benchmark",
        "type": "benchmark",
        "resources": {"cpu": 1, "memory": "256Mi", "storage": "1Gi"},
        "scaling": {"enabled": True, "min_instances": 1, "max_instances": 1},
    }

    deployment_id = await dm.deploy(config)
    assert isinstance(deployment_id, str) and deployment_id

    status = await dm.get_status(deployment_id)
    assert status["status"] == "running"
    assert status["name"] == "unit-test-benchmark"
    assert status["resources"]["cpu"] == 1

    ok = await dm.scale(deployment_id, 3)
    assert ok is True
    status = await dm.get_status(deployment_id)
    assert status["scaling"]["enabled"] is True
    assert status["scaling"]["max_instances"] == 3
    assert status["status"] == "running"

    # list deployments should include ours
    deployments = await dm.list_deployments()
    ids = [d["deployment_id"] for d in deployments]
    assert deployment_id in ids

    ok = await dm.teardown(deployment_id)
    assert ok is True
    status = await dm.get_status(deployment_id)
    assert status["status"] == "terminated"


@pytest.mark.asyncio
async def test_deploy_multiple_and_query_status():
    dm = DeploymentManager()

    ids = []
    for i in range(2):
        ids.append(await dm.deploy({"name": f"unit-multi-{i}", "type": "benchmark"}))

    # All should be running
    for did in ids:
        st = await dm.get_status(did)
        assert st["status"] in ("running", "provisioning")  # allow transient

    # Teardown all
    for did in ids:
        await dm.teardown(did)
        st = await dm.get_status(did)
        assert st["status"] == "terminated"