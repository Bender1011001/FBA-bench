"""Amazon 2025 fee engine (high-fidelity, all surcharges and penalties)."""
from typing import Dict, Optional, Union

from fba_bench.money import Money
from fba_bench.config_loader import load_config
from fba_bench.config_models import FBABenchConfig

class FeeEngine:
    """
    Amazon fee calculation engine with unified configuration system.
    
    Fee structures are loaded from the unified YAML configuration for maintainability.
    This allows easy updates when Amazon changes fee schedules without code changes.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize fee engine with unified configuration system.
        
        Args:
            config_path: Path to unified YAML configuration file.
                        If None, uses the default configuration location.
        """
        self.config = load_config(config_path)
        self._load_fee_structures()
    
    def _load_fee_structures(self):
        """Load fee structures from unified configuration."""
        # Convert referral fees to the expected format for backward compatibility
        self.REFERRAL_FEES = {}
        for category, tiers in self.config.referral_fees.items():
            self.REFERRAL_FEES[category] = [
                (tier.threshold, tier.percentage, tier.minimum, tier.maximum)
                for tier in tiers
            ]
        
        # Load FBA fulfillment fees directly
        self.FBA_FEES = {
            "standard": self.config.fba_fulfillment_fees.standard,
            "oversize": self.config.fba_fulfillment_fees.oversize,
        }
        
        # Store metadata for reference
        self.fee_metadata = {
            "version": self.config.fee_metadata.version,
            "effective_date": self.config.fee_metadata.effective_date,
            "source": self.config.fee_metadata.source,
            "last_updated": self.config.fee_metadata.last_updated,
            "notes": self.config.fee_metadata.notes,
        }


    def referral_fee(self, category: str, price: Union[float, Money]) -> Money:
        """Calculate referral fee for a given category and price using tiered structure."""
        # Convert input to Money if needed
        if isinstance(price, float):
            if self.config.simulation.money_strict:
                raise TypeError("Float prices not allowed when MONEY_STRICT=True. Use Money type.")
            price_money = Money.from_dollars(price)
        else:
            price_money = price
        
        tiers = self.REFERRAL_FEES.get(category, self.REFERRAL_FEES["DEFAULT"])
        
        # Sort tiers by threshold to ensure correct processing
        sorted_tiers = sorted(tiers, key=lambda x: x[0])
        
        total_fee = Money.zero()
        remaining_price = price_money
        
        for i, (threshold, pct, min_fee, max_fee) in enumerate(sorted_tiers):
            if remaining_price <= Money.zero():
                break
                
            threshold_money = Money.from_dollars(threshold)
            
            # Determine the price range for this tier
            if i < len(sorted_tiers) - 1:
                next_threshold = Money.from_dollars(sorted_tiers[i + 1][0])
                tier_price = min(remaining_price, next_threshold - threshold_money) if price_money >= threshold_money else Money.zero()
            else:
                tier_price = remaining_price if price_money >= threshold_money else Money.zero()
            
            if tier_price > Money.zero():
                # Calculate tier fee using Decimal for precision, then convert to Money
                from decimal import Decimal
                tier_fee_decimal = tier_price.to_decimal() * Decimal(str(pct))
                tier_fee = Money.from_dollars(tier_fee_decimal)
                
                # Apply minimum fee only to the total, not per tier
                if i == 0 and min_fee is not None:
                    min_fee_money = Money.from_dollars(min_fee)
                    tier_fee = max(tier_fee, min_fee_money)
                
                # Apply maximum fee cap for this tier
                if max_fee is not None:
                    max_fee_money = Money.from_dollars(max_fee)
                    tier_fee = min(tier_fee, max_fee_money)
                
                total_fee += tier_fee
                remaining_price -= tier_price
        
        return total_fee

    def fba_fulfillment_fee(self, size_tier: str, size: str) -> Money:
        """Calculate FBA fulfillment fee based on size tier and size."""
        fee_amount = self.FBA_FEES.get(size_tier, {}).get(size, 0.0)
        return Money.from_dollars(fee_amount)

    def fuel_surcharge(self, fulfillment_fee: Union[float, Money]) -> Money:
        """Calculate fuel surcharge based on fulfillment fee."""
        if isinstance(fulfillment_fee, float):
            if self.config.simulation.money_strict:
                raise TypeError("Float fulfillment_fee not allowed when MONEY_STRICT=True. Use Money type.")
            fulfillment_money = Money.from_dollars(fulfillment_fee)
        else:
            fulfillment_money = fulfillment_fee
        
        from decimal import Decimal
        surcharge_decimal = fulfillment_money.to_decimal() * Decimal(str(self.config.fees.fuel_surcharge_pct))
        return Money.from_dollars(surcharge_decimal)

    def holiday_surcharge(self, is_holiday_season: bool) -> Money:
        """
        Calculate the holiday surcharge.

        Args:
            is_holiday_season (bool): Whether it is the holiday season.

        Returns:
            Money: Holiday surcharge amount.
        """
        return Money.from_dollars(self.config.fees.holiday_surcharge) if is_holiday_season else Money.zero()

    def dim_weight_surcharge(self, applies: bool, weight: float = 1.0) -> Money:
        """
        Calculate the dimensional weight surcharge using tiered schedule.

        Args:
            applies (bool): Whether dimensional weight surcharge applies.
            weight (float): Package weight in pounds for tier calculation.

        Returns:
            Money: Dimensional weight surcharge amount based on weight tier.
        """
        if not applies:
            return Money.zero()
        
        # BUGFIX: Use tiered dimensional weight schedule instead of flat rate
        for tier in self.config.fees.dim_weight_tiers:
            if tier.weight_min <= weight < tier.weight_max:
                return Money.from_dollars(tier.surcharge)
        
        # Fallback to highest tier if weight exceeds all ranges
        if self.config.fees.dim_weight_tiers:
            return Money.from_dollars(self.config.fees.dim_weight_tiers[-1].surcharge)
        
        # Fallback to $1.25 if no tiers configured (backward compatibility)
        return Money.from_dollars(1.25)

    def long_term_storage_fee(self, cubic_feet: float, months: int) -> Money:
        """
        Calculate the long-term storage fee.

        Args:
            cubic_feet (float): Cubic feet of inventory.
            months (int): Number of months in storage.

        Returns:
            Money: Long-term storage fee amount.
        """
        from decimal import Decimal
        fee_decimal = Decimal(str(self.config.fees.long_term_storage_fee)) * Decimal(str(cubic_feet)) * Decimal(str(months))
        return Money.from_dollars(fee_decimal)

    def aged_inventory_surcharge(self, cubic_feet: float, aged_days: int) -> Money:
        """
        Surcharge for inventory aged 181-270 days and 271+ days.
        """
        from decimal import Decimal
        if aged_days >= 271:
            fee_decimal = Decimal(str(self.config.fees.aged_inventory_surcharge_271)) * Decimal(str(cubic_feet))
            return Money.from_dollars(fee_decimal)
        elif aged_days >= 181:
            fee_decimal = Decimal(str(self.config.fees.aged_inventory_surcharge_181)) * Decimal(str(cubic_feet))
            return Money.from_dollars(fee_decimal)
        return Money.zero()

    def low_inventory_level_fee(self, units: int, trailing_days_supply: float) -> Money:
        """
        Fee per unit if trailing supply is less than 28 days.
        """
        if trailing_days_supply < 28:
            from decimal import Decimal
            fee_decimal = Decimal(str(self.config.fees.low_inventory_level_fee_per_unit)) * Decimal(str(units))
            return Money.from_dollars(fee_decimal)
        return Money.zero()

    def storage_utilization_surcharge(self, storage_fee: Union[float, Money], weeks_supply: float) -> Money:
        """
        Surcharge if weeks of supply > 22.
        """
        if weeks_supply > 22:
            if isinstance(storage_fee, float):
                if self.config.simulation.money_strict:
                    raise TypeError("Float storage_fee not allowed when MONEY_STRICT=True. Use Money type.")
                storage_money = Money.from_dollars(storage_fee)
            else:
                storage_money = storage_fee
            
            from decimal import Decimal
            surcharge_decimal = storage_money.to_decimal() * Decimal(str(self.config.fees.storage_utilization_surcharge_pct))
            return Money.from_dollars(surcharge_decimal)
        return Money.zero()

    def unplanned_service_fee(self, units: int) -> Money:
        """
        Fee for unplanned prep/service per unit.
        """
        from decimal import Decimal
        fee_decimal = Decimal(str(self.config.fees.unplanned_service_fee_per_unit)) * Decimal(str(units))
        return Money.from_dollars(fee_decimal)

    def removal_fee(self, units: int) -> Money:
        """
        Calculate the removal fee for removed units.

        Args:
            units (int): Number of units removed.

        Returns:
            Money: Removal fee amount.
        """
        from decimal import Decimal
        fee_decimal = Decimal(str(self.config.fees.removal_fee_per_unit)) * Decimal(str(units))
        return Money.from_dollars(fee_decimal)

    def return_processing_fee(self, applicable_fees: Union[float, Money]) -> Money:
        """
        Calculate the return processing fee.

        Args:
            applicable_fees (Union[float, Money]): Applicable fees for returns.

        Returns:
            Money: Return processing fee amount.
        """
        if isinstance(applicable_fees, float):
            if self.config.simulation.money_strict:
                raise TypeError("Float applicable_fees not allowed when MONEY_STRICT=True. Use Money type.")
            fees_money = Money.from_dollars(applicable_fees)
        else:
            fees_money = applicable_fees
        
        from decimal import Decimal
        fee_decimal = fees_money.to_decimal() * Decimal(str(self.config.fees.return_processing_fee_pct))
        return Money.from_dollars(fee_decimal)

    def total_fees(
        self,
        category: str,
        price: Union[float, Money],
        size_tier: str,
        size: str,
        is_holiday_season: bool = False,
        dim_weight_applies: bool = False,
        weight: float = 1.0,  # BUGFIX: Add weight parameter for tiered dim weight calculation
        cubic_feet: Optional[float] = None,
        months_storage: int = 0,
        removal_units: int = 0,
        return_applicable_fees: Union[float, Money] = 0.0,
        aged_days: int = 0,
        aged_cubic_feet: float = 0.0,
        low_inventory_units: int = 0,
        trailing_days_supply: float = 999.0,
        storage_fee: Union[float, Money] = 0.0,
        weeks_supply: float = 0.0,
        unplanned_units: int = 0,
        penalty_fee: Union[float, Money] = 0.0,
        ancillary_fee: Union[float, Money] = 0.0,
    ) -> Dict[str, Union[float, Money]]:
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
        # Convert input parameters to Money if needed
        def _to_money(value):
            if isinstance(value, Money):
                return value
            elif isinstance(value, (int, float)):
                if self.config.simulation.money_strict and isinstance(value, float):
                    raise TypeError("Float values not allowed when MONEY_STRICT=True. Use Money type.")
                return Money.from_dollars(value)
            else:
                return Money.zero()
        
        # Calculate all fee components using Money arithmetic
        referral = self.referral_fee(category, price)
        fba = self.fba_fulfillment_fee(size_tier, size)
        fuel = self.fuel_surcharge(fba)
        holiday = self.holiday_surcharge(is_holiday_season)
        # BUGFIX: Pass weight parameter for tiered dimensional weight calculation
        dim_weight = self.dim_weight_surcharge(dim_weight_applies, weight)
        
        storage = self.long_term_storage_fee(cubic_feet or 0, months_storage) if months_storage > 0 else Money.zero()
        removal = self.removal_fee(removal_units) if removal_units > 0 else Money.zero()
        
        return_proc = Money.zero()
        if return_applicable_fees and (isinstance(return_applicable_fees, Money) and return_applicable_fees > Money.zero() or
                                      isinstance(return_applicable_fees, (int, float)) and return_applicable_fees > 0):
            return_proc = self.return_processing_fee(return_applicable_fees)
        
        aged_surcharge = self.aged_inventory_surcharge(aged_cubic_feet, aged_days) if aged_days >= 181 else Money.zero()
        low_inventory_fee = self.low_inventory_level_fee(low_inventory_units, trailing_days_supply) if low_inventory_units > 0 else Money.zero()
        
        storage_util_surcharge = Money.zero()
        if storage_fee and (isinstance(storage_fee, Money) and storage_fee > Money.zero() or
                           isinstance(storage_fee, (int, float)) and storage_fee > 0):
            storage_util_surcharge = self.storage_utilization_surcharge(storage_fee, weeks_supply)
        
        unplanned_service = self.unplanned_service_fee(unplanned_units) if unplanned_units > 0 else Money.zero()
        
        # Convert penalty and ancillary fees to Money
        penalty_money = _to_money(penalty_fee)
        ancillary_money = _to_money(ancillary_fee)
        storage_money = _to_money(storage_fee)

        # Calculate total using Money arithmetic
        # BUGFIX: Don't double-count storage fees - storage_fee parameter is for pre-computed storage,
        # not to be added on top of calculated long_term_storage_fee
        total = (
            referral + fba + fuel + holiday + dim_weight + storage + removal +
            return_proc + aged_surcharge + low_inventory_fee + storage_util_surcharge +
            unplanned_service + penalty_money + ancillary_money
        )
        
        # Only add storage_money if no months_storage was specified (pre-computed storage fee)
        if months_storage == 0 and storage_fee and (
            isinstance(storage_fee, Money) and storage_fee > Money.zero() or
            isinstance(storage_fee, (int, float)) and storage_fee > 0
        ):
            total += storage_money

        # Return Money types if MONEY_STRICT, otherwise convert to float for backward compatibility
        if self.config.simulation.money_strict:
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
                "penalty_fee": penalty_money,
                "ancillary_fee": ancillary_money,
                "total": total,
            }
        else:
            # Backward compatibility: return floats
            return {
                "referral_fee": referral.to_float(),
                "fba_fulfillment_fee": fba.to_float(),
                "fuel_surcharge": fuel.to_float(),
                "holiday_surcharge": holiday.to_float(),
                "dim_weight_surcharge": dim_weight.to_float(),
                "long_term_storage_fee": storage.to_float(),
                "removal_fee": removal.to_float(),
                "return_processing_fee": return_proc.to_float(),
                "aged_inventory_surcharge": aged_surcharge.to_float(),
                "low_inventory_level_fee": low_inventory_fee.to_float(),
                "storage_utilization_surcharge": storage_util_surcharge.to_float(),
                "unplanned_service_fee": unplanned_service.to_float(),
                "penalty_fee": penalty_money.to_float(),
                "ancillary_fee": ancillary_money.to_float(),
                "total": round(total.to_float(), 2),
            }