import asyncio
from decimal import Decimal
from typing import Dict, Any

import pytest
from pydantic import ValidationError

# Import scenario modules to ensure registration happens on import
from benchmarking.scenarios import registry as sc_reg  # noqa: F401
from benchmarking.scenarios.complex_marketplace import (
    ComplexMarketplaceConfig,
    generate_input as cm_generate_input,
    postprocess as cm_postprocess,
)
from benchmarking.scenarios.research_summarization import (
    ResearchSummarizationConfig,
    generate_input as rs_generate_input,
)
from benchmarking.scenarios.multiturn_tool_use import (
    MultiTurnToolUseConfig,
    generate_input as mt_generate_input,
)


def test_complex_marketplace_generate_input_determinism():
    params = {"num_products": 10, "num_orders": 15, "max_quantity": 3, "price_variance": 0.12, "allow_backorder": False}
    seed = 42
    p1 = cm_generate_input(seed=seed, params=params)
    p2 = cm_generate_input(seed=seed, params=params)
    assert p1 == p2, "ComplexMarketplace generate_input should be deterministic for same seed/params"

    # Different seed should differ
    p3 = cm_generate_input(seed=seed + 1, params=params)
    assert p1 != p3, "ComplexMarketplace generate_input should vary with different seed"


def test_research_summarization_generate_input_determinism():
    params = {"num_docs": 5, "max_tokens": 120, "focus_keywords": ["Q3", "revenue"], "noise_probability": 0.1}
    seed = 123
    p1 = rs_generate_input(seed=seed, params=params)
    p2 = rs_generate_input(seed=seed, params=params)
    assert p1 == p2
    p3 = rs_generate_input(seed=seed + 2, params=params)
    assert p1 != p3


def test_multiturn_tool_use_generate_input_determinism():
    params = {"steps": 3, "include_math": True, "include_extraction": True, "include_transform": True}
    seed = 7
    p1 = mt_generate_input(seed=seed, params=params)
    p2 = mt_generate_input(seed=seed, params=params)
    assert p1 == p2
    p3 = mt_generate_input(seed=seed + 8, params=params)
    assert p1 != p3


def test_complex_marketplace_config_validation_errors():
    # Negative num_products should raise
    with pytest.raises(ValidationError):
        ComplexMarketplaceConfig(num_products=-1)

    # price_variance out of bounds
    with pytest.raises(ValidationError):
        ComplexMarketplaceConfig(price_variance=1.5)


def test_research_summarization_config_validation_errors():
    with pytest.raises(ValidationError):
        ResearchSummarizationConfig(num_docs=0)

    with pytest.raises(ValidationError):
        ResearchSummarizationConfig(noise_probability=0.75)


def test_multiturn_tool_use_config_validation_errors():
    with pytest.raises(ValidationError):
        MultiTurnToolUseConfig(steps=0)


def test_complex_marketplace_postprocess_rounding_normalization():
    raw = {"revenue": 123.456789, "fulfilled_rate": 0.987654321}
    out = cm_postprocess(raw)
    # Revenue rounded to 2 decimals, fulfilled_rate to 4 decimals as per implementation
    assert out["revenue"] == 123.46
    assert abs(out["fulfilled_rate"] - 0.9877) < 1e-9


@pytest.mark.parametrize(
    "params",
    [
        {"num_products": 5, "num_orders": 10, "max_quantity": 2, "price_variance": 0.05, "allow_backorder": False},
        {"num_products": 8, "num_orders": 12, "max_quantity": 4, "price_variance": 0.12, "allow_backorder": True},
    ],
)
def test_complex_marketplace_generate_input_schema(params: Dict[str, Any]):
    seed = 99
    payload = cm_generate_input(seed=seed, params=params)
    assert "catalog" in payload and isinstance(payload["catalog"], list)
    assert "orders" in payload and isinstance(payload["orders"], list)
    assert "policies" in payload and isinstance(payload["policies"], dict)
    assert payload["config"]["num_products"] == params["num_products"]
    assert payload["config"]["allow_backorder"] == params["allow_backorder"]