# Golden Run v1 (Deterministic Finance Scenario)

This example executes a fully in-process, deterministic end-to-end flow that exercises the EventBus, sales events, fee calculations, BSR engine (EMA-based), double-entry ledger, financial audit, and fee aggregation. It produces a golden KPI snapshot artifact you can diff over time to ensure stability.

Key components used (in-process, no web server):
- EventBus: Async in-memory pub/sub
- DoubleEntryLedgerService: Balanced debits/credits with statements
- FinancialAuditService: Deterministic audit with Money-safe arithmetic
- BsrEngineV3Service: EMA-based relative indices vs. market
- FeeCalculationService: Comprehensive breakdown with Money and Decimal
- FeeMetricsAggregatorService: Aggregates fees by type from SaleOccurred

Artifacts:
- scenario_results/golden_run_snapshot.json

Requirements:
- Python 3.10+
- Repository dependencies installed (e.g., `pip install -r requirements.txt` if provided by your environment)


## How to Run

1) Execute the example runner

```bash
python docs/examples/golden-run-example/run_golden_run_example.py
```

2) Output
- A KPI snapshot is written to:
  - scenario_results/golden_run_snapshot.json
- The script prints a concise summary with:
  - processed_transactions (from audit)
  - accounting identity validity (from ledger)
  - top fee types with totals/counts
  - BSR composite index for the golden ASIN

3) Determinism
- Money arithmetic uses integer cents; no Money*float or Money/float is performed.
- FeeCalculationService is passed `peak_season=False` and `advertising_spend=Money.zero()`.
- The competitor seeding and sales sequence are fixed.
- BSR indices use Decimal EMAs with quantization for stable results.
- Event processing is in-process using AsyncioQueueBackend, and we await a small flush delay after publishing events.


## Scenario Details

ASIN:
- "B00GOLDEN"

Market seed (one CompetitorPricesUpdated):
- COMP1: price=$17.99, bsr=1200, velocity=2.0
- COMP2: price=$21.49, bsr=800, velocity=3.0

Sales sequence (three SaleOccurred events):
- sale_price=$19.99
- cost_per_unit=$6.00
- Events:
  1) units_sold=2, units_demanded=5
  2) units_sold=3, units_demanded=6
  3) units_sold=1, units_demanded=3

Fee calculation inputs:
- Product stub: category="default", weight_oz=16, dimensions_inches=[10, 6, 3]
- additional_context:
  - peak_season=False
  - requires_prep=False
  - requires_labeling=False
  - advertising_spend=$0.00
  - months_in_storage=0


## Expected Snapshot Shape

File: scenario_results/golden_run_snapshot.json

Top-level:
```json
{
  "timestamp": "ISO-8601",
  "asin": "B00GOLDEN",
  "audit": { ... },
  "ledger": { ... },
  "bsr": { ... },
  "fees_by_type": { ... }
}
```

audit (from FinancialAuditService.get_audit_statistics()):
- processed_transactions: 3
- total_violations: number
- critical_violations: number
- total_revenue_audited: "$x.xx"
- total_fees_audited: "$y.yy"
- total_profit_audited: "$z.zz"
- current_position:
  - total_assets: "$..."
  - total_liabilities: "$..."
  - total_equity: "$..."
  - accounting_identity_valid: true/false
  - identity_difference: "$..."

ledger (from DoubleEntryLedgerService.get_financial_position()):
- cash: "$..."
- inventory_value: "$..."
- accounts_receivable: "$..."
- accounts_payable: "$..."
- accrued_liabilities: "$..."
- total_assets: "$..."
- total_liabilities: "$..."
- total_equity: "$..."
- accounting_identity_valid: true/false
- identity_difference: "$..."
- timestamp: "ISO-8601"

bsr:
- products: [
  {
    "asin": "B00GOLDEN",
    "velocity_index": number,
    "conversion_index": number,
    "composite_index": number
  }
]
- market_ema_velocity: string (Decimal serialized, e.g., "2.3")
- market_ema_conversion: string (Decimal serialized)
- competitor_count: 2

fees_by_type (from FeeMetricsAggregatorService):
```json
{
  "referral": {
    "total_amount": "$x.xx",
    "count": 3,
    "average_amount": "$y.yy"
  },
  "fba": {
    "total_amount": "$x.xx",
    "count": 3,
    "average_amount": "$y.yy"
  }
}
```
- These values aggregate fee_breakdown across the 3 sale events.
- Averages are integer-cents division: Money(total.cents // count).


## Interpreting Results

- processed_transactions: Should be exactly 3 (one per sale event).
- accounting_identity_valid: Preferably true. If false, identity_difference shows the exact Money imbalance (should be "$0.00" in balanced conditions).
- BSR indices: Non-null numeric values for the golden ASIN are expected after 3 sales and seeded competitor EMAs.
- Fees: Non-empty dictionary. Each fee type shows Money-formatted totals and averages, and count >= 1.


## Integration Test

A matching integration test exists at:
- tests/integration/test_golden_run.py

Run:
```bash
pytest -q tests/integration/test_golden_run.py
```

The test:
- Wires services in-process with EventBus.
- Publishes the competitor snapshot and three sales.
- Awaits flush.
- Builds KPI snapshot in tmp_path/golden_run_snapshot.json.
- Asserts:
  - audit.processed_transactions == 3
  - accounting identity shape/boolean type
  - BSR indices are present (not null) for ASIN
  - fees_by_type is non-empty with Money strings and count >= 1
  - ledger totals present; identity_difference is Money string


## Optional Configuration

A template is provided in:
- docs/examples/golden-run-example/golden_run_config.yaml

This is informative (the example runner is self-contained). You can adapt it for future parameterization if desired.