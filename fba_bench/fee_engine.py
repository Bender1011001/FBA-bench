"""Amazon 2025 fee engine (high-fidelity, all surcharges and penalties)."""
import json
import os
from typing import Dict, Optional

from fba_bench.config import (
    FUEL_SURCHARGE_PCT,
    HOLIDAY_SURCHARGE,
    DIM_WEIGHT_SURCHARGE,
    PROFESSIONAL_MONTHLY,
    INDIVIDUAL_PER_ITEM,
    AGED_INVENTORY_SURCHARGE_181,
    AGED_INVENTORY_SURCHARGE_271,
    LOW_INVENTORY_LEVEL_FEE_PER_UNIT,
    STORAGE_UTILIZATION_SURCHARGE_PCT,
    UNPLANNED_SERVICE_FEE_PER_UNIT,
    LONG_TERM_STORAGE_FEE,
    REMOVAL_FEE_PER_UNIT,
    RETURN_PROCESSING_FEE_PCT,
)

class FeeEngine:
    """
    Amazon fee calculation engine with configurable fee structures.
    
    Fee structures are loaded from fee_config.json for maintainability.
    This allows easy updates when Amazon changes fee schedules without code changes.
    """
    
    def __init__(self, fee_config_path: Optional[str] = None):
        """
        Initialize fee engine with configurable fee structures.
        
        Args:
            fee_config_path: Path to fee configuration JSON file.
                           Defaults to fee_config.json in the same directory.
        """
        if fee_config_path is None:
            fee_config_path = os.path.join(os.path.dirname(__file__), 'fee_config.json')
        
        self._load_fee_config(fee_config_path)
    
    def _load_fee_config(self, config_path: str):
        """Load fee configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Convert referral fees to the expected format
            self.REFERRAL_FEES = {}
            for category, tiers in config['referral_fees'].items():
                self.REFERRAL_FEES[category] = [
                    (tier['threshold'], tier['percentage'], tier['minimum'], tier['maximum'])
                    for tier in tiers
                ]
            
            # Load FBA fulfillment fees directly
            self.FBA_FEES = config['fba_fulfillment_fees']
            
            # Store metadata for reference
            self.fee_metadata = config.get('fee_metadata', {})
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            # Fallback to hardcoded values if config file is missing or invalid
            print(f"Warning: Could not load fee config from {config_path}: {e}")
            print("Falling back to hardcoded fee structures.")
            self._load_fallback_fees()
    
    def _load_fallback_fees(self):
        """Fallback to hardcoded fee structures if config file fails to load."""
        self.REFERRAL_FEES = {
            "DEFAULT": [(0, 0.15, 0.30, None)],
            "Apparel": [(0, 0.17, 0.30, None)],
            "Jewelry": [(0, 0.20, 0.30, 250), (250, 0.05, 0.30, None)],
            "Electronics": [(0, 0.08, 0.30, None)],
        }
        
        self.FBA_FEES = {
            "standard": {"small": 3.22, "large": 4.75},
            "oversize": {"medium": 8.26, "large": 10.50},
        }
        
        self.fee_metadata = {"version": "fallback", "source": "hardcoded"}


    def referral_fee(self, category: str, price: float) -> float:
        """Calculate referral fee for a given category and price using tiered structure."""
        tiers = self.REFERRAL_FEES.get(category, self.REFERRAL_FEES["DEFAULT"])
        
        # For tiered fee structures, we need to calculate fees for each applicable tier
        # and use the appropriate tier based on the price range
        
        # Sort tiers by threshold to ensure correct processing
        sorted_tiers = sorted(tiers, key=lambda x: x[0])
        
        total_fee = 0.0
        remaining_price = price
        
        for i, (threshold, pct, min_fee, max_fee) in enumerate(sorted_tiers):
            if remaining_price <= 0:
                break
                
            # Determine the price range for this tier
            if i < len(sorted_tiers) - 1:
                next_threshold = sorted_tiers[i + 1][0]
                tier_price = min(remaining_price, next_threshold - threshold) if price >= threshold else 0
            else:
                tier_price = remaining_price if price >= threshold else 0
            
            if tier_price > 0:
                tier_fee = tier_price * pct
                
                # Apply minimum fee only to the total, not per tier
                if i == 0 and min_fee is not None:
                    tier_fee = max(tier_fee, min_fee)
                
                # Apply maximum fee cap for this tier
                if max_fee is not None:
                    tier_fee = min(tier_fee, max_fee)
                
                total_fee += tier_fee
                remaining_price -= tier_price
        
        return round(total_fee, 2)

    def fba_fulfillment_fee(self, size_tier: str, size: str) -> float:
        """Calculate FBA fulfillment fee based on size tier and size."""
        return self.FBA_FEES.get(size_tier, {}).get(size, 0.0)

    def fuel_surcharge(self, fulfillment_fee: float) -> float:
        return round(fulfillment_fee * FUEL_SURCHARGE_PCT, 2)

    def holiday_surcharge(self, is_holiday_season: bool) -> float:
        """
        Calculate the holiday surcharge.

        Args:
            is_holiday_season (bool): Whether it is the holiday season.

        Returns:
            float: Holiday surcharge amount.
        """
        return HOLIDAY_SURCHARGE if is_holiday_season else 0.0

    def dim_weight_surcharge(self, applies: bool) -> float:
        """
        Calculate the dimensional weight surcharge.

        Args:
            applies (bool): Whether dimensional weight surcharge applies.

        Returns:
            float: Dimensional weight surcharge amount.
        """
        return DIM_WEIGHT_SURCHARGE if applies else 0.0

    def long_term_storage_fee(self, cubic_feet: float, months: int) -> float:
        """
        Calculate the long-term storage fee.

        Args:
            cubic_feet (float): Cubic feet of inventory.
            months (int): Number of months in storage.

        Returns:
            float: Long-term storage fee amount.
        """
        return round(LONG_TERM_STORAGE_FEE * cubic_feet * months, 2)

    def aged_inventory_surcharge(self, cubic_feet: float, aged_days: int) -> float:
        """
        Surcharge for inventory aged 181-270 days and 271+ days.
        """
        if aged_days >= 271:
            return round(AGED_INVENTORY_SURCHARGE_271 * cubic_feet, 2)
        elif aged_days >= 181:
            return round(AGED_INVENTORY_SURCHARGE_181 * cubic_feet, 2)
        return 0.0

    def low_inventory_level_fee(self, units: int, trailing_days_supply: float) -> float:
        """
        Fee per unit if trailing supply is less than 28 days.
        """
        if trailing_days_supply < 28:
            return round(LOW_INVENTORY_LEVEL_FEE_PER_UNIT * units, 2)
        return 0.0

    def storage_utilization_surcharge(self, storage_fee: float, weeks_supply: float) -> float:
        """
        Surcharge if weeks of supply > 22.
        """
        if weeks_supply > 22:
            return round(storage_fee * STORAGE_UTILIZATION_SURCHARGE_PCT, 2)
        return 0.0

    def unplanned_service_fee(self, units: int) -> float:
        """
        Fee for unplanned prep/service per unit.
        """
        return round(UNPLANNED_SERVICE_FEE_PER_UNIT * units, 2)

    def removal_fee(self, units: int) -> float:
        """
        Calculate the removal fee for removed units.

        Args:
            units (int): Number of units removed.

        Returns:
            float: Removal fee amount.
        """
        return round(REMOVAL_FEE_PER_UNIT * units, 2)

    def return_processing_fee(self, applicable_fees: float) -> float:
        """
        Calculate the return processing fee.

        Args:
            applicable_fees (float): Applicable fees for returns.

        Returns:
            float: Return processing fee amount.
        """
        return round(applicable_fees * RETURN_PROCESSING_FEE_PCT, 2)

    def total_fees(
        self,
        category: str,
        price: float,
        size_tier: str,
        size: str,
        is_holiday_season: bool = False,
        dim_weight_applies: bool = False,
        cubic_feet: Optional[float] = None,
        months_storage: int = 0,
        removal_units: int = 0,
        return_applicable_fees: float = 0.0,
        aged_days: int = 0,
        aged_cubic_feet: float = 0.0,
        low_inventory_units: int = 0,
        trailing_days_supply: float = 999.0,
        storage_fee: float = 0.0,
        weeks_supply: float = 0.0,
        unplanned_units: int = 0,
        penalty_fee: float = 0.0,
        ancillary_fee: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate all fees for a transaction, including penalty and ancillary fees.

        Args:
            category (str): Product category.
            price (float): Product price.
            size_tier (str): Size tier ("standard", "oversize", etc.).
            size (str): Size label ("small", "large", etc.).
            is_holiday_season (bool, optional): Whether it is the holiday season.
            dim_weight_applies (bool, optional): Whether dimensional weight surcharge applies.
            cubic_feet (float, optional): Cubic feet of inventory.
            months_storage (int, optional): Number of months in storage.
            removal_units (int, optional): Number of units removed.
            return_applicable_fees (float, optional): Applicable fees for returns.
            aged_days (int, optional): Age of inventory in days.
            aged_cubic_feet (float, optional): Cubic feet of aged inventory.
            low_inventory_units (int, optional): Number of units with low inventory.
            trailing_days_supply (float, optional): Trailing days of supply.
            storage_fee (float, optional): Precomputed storage fee (if any).
            weeks_supply (float, optional): Weeks of supply.
            unplanned_units (int, optional): Units requiring unplanned service.
            penalty_fee (float, optional): Penalty fee to apply.
            ancillary_fee (float, optional): Ancillary fee to apply.

        Returns:
            Dict[str, float]: Dictionary of all individual fees and the total.
        """
        """Calculate all fees for a transaction, including penalty and ancillary fees."""
        referral = self.referral_fee(category, price)
        fba = self.fba_fulfillment_fee(size_tier, size)
        fuel = self.fuel_surcharge(fba)
        holiday = self.holiday_surcharge(is_holiday_season)
        dim_weight = self.dim_weight_surcharge(dim_weight_applies)
        storage = self.long_term_storage_fee(cubic_feet or 0, months_storage) if months_storage > 0 else 0.0
        removal = self.removal_fee(removal_units) if removal_units > 0 else 0.0
        return_proc = self.return_processing_fee(return_applicable_fees) if return_applicable_fees > 0 else 0.0
        aged_surcharge = self.aged_inventory_surcharge(aged_cubic_feet, aged_days) if aged_days >= 181 else 0.0
        low_inventory_fee = self.low_inventory_level_fee(low_inventory_units, trailing_days_supply) if low_inventory_units > 0 else 0.0
        storage_util_surcharge = self.storage_utilization_surcharge(storage_fee, weeks_supply) if storage_fee > 0 else 0.0
        unplanned_service = self.unplanned_service_fee(unplanned_units) if unplanned_units > 0 else 0.0

        total = (
            referral + fba + fuel + holiday + dim_weight + storage + removal +
            return_proc + aged_surcharge + low_inventory_fee + storage_util_surcharge + unplanned_service +
            penalty_fee + ancillary_fee + storage_fee
        )

        return {
            "referral_fee": referral,
            "fba_fulfillment_fee": fba,
            "fuel_surcharge": fuel,
            "holiday_surcharge": holiday,
            "dim_weight_surcharge": dim_weight,
            "long_term_storage_fee": storage,
            "removal_fee": removal,
            "return_processing_fee": return_proc,
            "aged_inventory_surcharge": aged_surcharge,
            "low_inventory_level_fee": low_inventory_fee,
            "storage_utilization_surcharge": storage_util_surcharge,
            "unplanned_service_fee": unplanned_service,
            "penalty_fee": penalty_fee,
            "ancillary_fee": ancillary_fee,
            "total": round(total, 2),
        }