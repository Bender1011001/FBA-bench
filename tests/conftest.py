"""Test configuration and fixtures for FBA-bench testing framework."""
import pytest
from fba_bench.simulation import Simulation
from fba_bench.adversarial_events import AdversarialEventCatalog


@pytest.fixture
def sim_factory():
    """Factory for creating simulation instances with consistent configuration."""
    def _create_sim(seed=42, days=30, config=None, **kwargs):
        sim = Simulation(seed=seed, **kwargs)
        # Add a default product for testing
        sim.launch_product("B000TEST", "DEFAULT", cost=5.0, price=19.99, qty=100)
        return sim
    return _create_sim


@pytest.fixture
def clean_simulation():
    """Create a clean simulation instance for testing."""
    return Simulation(seed=42)


@pytest.fixture
def adversarial_catalog():
    """Create an adversarial event catalog for testing."""
    return AdversarialEventCatalog()