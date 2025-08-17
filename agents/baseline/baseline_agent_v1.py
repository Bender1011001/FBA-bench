from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from money import Money
from services.toolbox_api_service import ToolboxAPIService
from services.toolbox_schemas import (
    ObserveRequest,
    SetPriceRequest,
    SetPriceResponse,
)


@dataclass(frozen=True)
class BaselineConfig:
    min_margin_pct: float = 10.0  # reserved for future extension
    max_price_change_pct: float = 20.0  # hard cap on a single decision


class BaselineAgentV1:
    """
    Minimal deterministic pricing agent.

    Policy:
    - If conversion_rate < 0.05: decrease price by up to 5% (capped by max_price_change_pct)
    - If conversion_rate > 0.2: increase price by up to 5% (capped by max_price_change_pct)
    - Else: no action

    Implementation details:
    - All price math is performed with Decimal and Money to avoid float contamination
    - Rounds to integer cents deterministically via Money's rounding
    """

    def __init__(
        self,
        agent_id: str,
        toolbox: ToolboxAPIService,
        min_margin_pct: float = 10.0,
        max_price_change_pct: float = 20.0,
    ):
        self.agent_id = agent_id
        self.toolbox = toolbox
        self.config = BaselineConfig(
            min_margin_pct=min_margin_pct, max_price_change_pct=max_price_change_pct
        )

    def decide(self, asin: str) -> Optional[SetPriceResponse]:
        obs = self.toolbox.observe(ObserveRequest(asin=asin))
        # Require an existing observation with price for this trivial policy
        if not obs.found or obs.price is None:
            return None

        current_price: Money = obs.price
        cr = obs.conversion_rate

        # Decide change direction and magnitude (desired 5%)
        if cr is not None and cr < 0.05:
            desired_change = Decimal("-0.05")
        elif cr is not None and cr > 0.2:
            desired_change = Decimal("0.05")
        else:
            return None  # no change

        # Cap change by max bound
        max_bound = Decimal(str(self.config.max_price_change_pct)) / Decimal("100")
        magnitude = min(abs(desired_change), max_bound)
        signed_change = magnitude.copy_negate() if desired_change < 0 else magnitude

        # Compute new price with deterministic rounding to cents
        factor = Decimal("1") + signed_change
        new_price = current_price * factor  # Money * Decimal is supported with ROUND_HALF_UP

        # Safety: ensure at least 1 cent
        if new_price.cents <= 0:
            new_price = Money(1, current_price.currency)

        # Avoid publishing no-op if rounding yielded same price
        if new_price == current_price:
            return None

        # Publish command via toolbox
        rsp = self.toolbox.set_price(
            SetPriceRequest(
                agent_id=self.agent_id,
                asin=asin,
                new_price=new_price,
                reason=f"baseline_v1 {'decrease' if signed_change < 0 else 'increase'} {abs(magnitude) * 100}%",
            )
        )
        return rsp