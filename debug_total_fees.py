from fba_bench.fee_engine import FeeEngine

fee_engine = FeeEngine()

total_fees = fee_engine.total_fees(
    category="DEFAULT",
    price=20.0,
    size_tier="standard",
    size="small",
    is_holiday_season=False,
    dim_weight_applies=False,
    cubic_feet=1.0,
    months_storage=1,
    removal_units=0,
    return_applicable_fees=0,
    aged_days=100,  # No aged inventory surcharge
    aged_cubic_feet=0,
    low_inventory_units=0,
    trailing_days_supply=30,
    storage_fee=0.78,
    weeks_supply=10,
    unplanned_units=0
)

print("Total fees breakdown:")
for key, value in total_fees.items():
    print(f"  {key}: ${value:.2f}")

print(f"\nExpected components:")
print(f"  Referral fee: ${fee_engine.referral_fee('DEFAULT', 20.0):.2f}")
print(f"  FBA fulfillment: ${fee_engine.fba_fulfillment_fee('standard', 'small'):.2f}")
print(f"  Fuel surcharge: ${fee_engine.fuel_surcharge(3.22):.2f}")
print(f"  Storage fee: $0.78")
print(f"  Expected total: ${3.00 + 3.22 + 0.06 + 0.78:.2f}")