from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import os
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator, root_validator
from money import Money


# Defaults and common validators
_DEFAULT_ASIN_REGEX = os.getenv("ASIN_REGEX", r"^[A-Z0-9]{8,12}$")
_ASIN_PATTERN = re.compile(_DEFAULT_ASIN_REGEX)


class _CommonModel(BaseModel):
    class Config:
        # Allow arbitrary types like Money while still validating via validators
        arbitrary_types_allowed = True
        validate_assignment = True
        anystr_strip_whitespace = True


def _validate_asin(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ASIN must be a non-empty string")
    if not _ASIN_PATTERN.match(value):
        # Do not overconstrain: permit if it looks like a plausible ID even if it doesn't match default regex
        # Strategy: if default regex fails, still allow alnum length 5-20.
        loose = re.fullmatch(r"[A-Za-z0-9\-_.]{5,24}", value.strip())
        if not loose:
            raise ValueError(f"ASIN '{value}' failed validation")
    return value.strip()


def _validate_money_instance(val: Money) -> Money:
    if isinstance(val, float):
        raise TypeError("Money must not be a float - use Money.from_dollars or Money(int_cents)")
    if not isinstance(val, Money):
        raise TypeError(f"Value must be Money, got {type(val)}")
    return val


def _require_positive_int(name: str, val: Optional[int], allow_zero: bool = False) -> Optional[int]:
    if val is None:
        return val
    if not isinstance(val, int):
        raise TypeError(f"{name} must be an integer")
    if allow_zero:
        if val < 0:
            raise ValueError(f"{name} must be >= 0")
    else:
        if val <= 0:
            raise ValueError(f"{name} must be > 0")
    return val


# Observe

class ObserveRequest(_CommonModel):
    asin: str = Field(...)
    fields: Optional[List[str]] = Field(default=None)

    _asin_validator = validator("asin", allow_reuse=True)(_validate_asin)


class ObserveResponse(_CommonModel):
    asin: str
    found: bool
    price: Optional[Money] = None
    inventory: Optional[int] = None
    bsr: Optional[int] = None
    conversion_rate: Optional[float] = None
    timestamp: datetime

    _asin_validator = validator("asin", allow_reuse=True)(_validate_asin)

    @validator("price")
    def _validate_money(cls, v):
        if v is None:
            return v
        return _validate_money_instance(v)

    @validator("inventory")
    def _validate_inventory(cls, v):
        return _require_positive_int("inventory", v, allow_zero=True)

    @validator("bsr")
    def _validate_bsr(cls, v):
        if v is None:
            return v
        if not isinstance(v, int) or v < 1:
            raise ValueError("bsr must be an integer >= 1")
        return v

    @validator("conversion_rate")
    def _validate_cr(cls, v):
        if v is None:
            return v
        if not isinstance(v, (int, float)):
            raise TypeError("conversion_rate must be numeric")
        if v < 0.0:
            raise ValueError("conversion_rate must be >= 0.0")
        return float(v)


# Set Price

class SetPriceRequest(_CommonModel):
    agent_id: str
    asin: str
    new_price: Money
    reason: Optional[str] = None

    _asin_validator = validator("asin", allow_reuse=True)(_validate_asin)

    @validator("agent_id")
    def _validate_agent_id(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("agent_id must be a non-empty string")
        return v.strip()

    @validator("new_price")
    def _validate_new_price(cls, v):
        v = _validate_money_instance(v)
        if v.cents <= 0:
            raise ValueError("new_price must be positive (at least 1 cent)")
        return v


class SetPriceResponse(_CommonModel):
    accepted: bool
    command_id: str
    asin: str
    new_price: Money
    details: Optional[str] = None

    _asin_validator = validator("asin", allow_reuse=True)(_validate_asin)
    _new_price_validator = validator("new_price", allow_reuse=True)(_validate_money_instance)


# Launch Product

class LaunchProductRequest(_CommonModel):
    asin: str
    initial_price: Money
    initial_inventory: int = Field(..., ge=0)
    category: Optional[str] = None
    dimensions_inches: Optional[List[Decimal]] = None  # [L, W, H]
    weight_oz: Optional[Decimal] = None  # >= 0

    _asin_validator = validator("asin", allow_reuse=True)(_validate_asin)
    _initial_price_validator = validator("initial_price", allow_reuse=True)(_validate_money_instance)

    @validator("initial_price")
    def _validate_initial_price_positive(cls, v):
        if v.cents <= 0:
            raise ValueError("initial_price must be positive")
        return v

    @validator("initial_inventory")
    def _validate_initial_inventory(cls, v):
        return _require_positive_int("initial_inventory", v, allow_zero=True)

    @validator("weight_oz")
    def _validate_weight(cls, v):
        if v is None:
            return v
        if not isinstance(v, Decimal):
            v = Decimal(v)
        if v < 0:
            raise ValueError("weight_oz must be >= 0")
        return v

    @root_validator(skip_on_failure=True)
    def _validate_dimensions(cls, values: Dict):
        dims = values.get("dimensions_inches")
        if dims is None:
            return values
        if not isinstance(dims, list) or len(dims) != 3:
            raise ValueError("dimensions_inches must be a list of length 3 [L, W, H]")
        converted: List[Decimal] = []
        for d in dims:
            dd = d if isinstance(d, Decimal) else Decimal(str(d))
            if dd <= 0:
                raise ValueError("Each dimension must be > 0")
            converted.append(dd)
        values["dimensions_inches"] = converted
        return values


class LaunchProductResponse(_CommonModel):
    accepted: bool
    asin: str
    message: str

    _asin_validator = validator("asin", allow_reuse=True)(_validate_asin)