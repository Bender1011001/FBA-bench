# Metrics API Endpoints

Base path: `/api/metrics`

Notes:
- All Money values are serialized as strings via Money.__str__ (e.g., "$12.34").
- BSR indices are numbers or null.
- Timestamps are ISO-8601 strings.

## GET /api/metrics/audit

Returns real-time audit counters and current financial position from the FinancialAuditService. If the audit service is not wired, fields will return defaults with `audit_enabled: false`.

Response shape:
```json
{
  "processed_transactions": 0,
  "total_violations": 0,
  "critical_violations": 0,
  "total_revenue_audited": "$0.00",
  "total_fees_audited": "$0.00",
  "total_profit_audited": "$0.00",
  "current_position": {
    "total_assets": "$0.00",
    "total_liabilities": "$0.00",
    "total_equity": "$0.00",
    "accounting_identity_valid": true,
    "identity_difference": "$0.00"
  },
  "audit_enabled": true,
  "halt_on_violation": false,
  "tolerance_cents": 0
}
```

## GET /api/metrics/ledger

Returns ledger balances and accounting identity from DoubleEntryLedgerService.generate_balance_sheet()/get_financial_position(). All Money values are strings; timestamp is ISO.

Response shape:
```json
{
  "cash": "$0.00",
  "inventory_value": "$0.00",
  "accounts_receivable": "$0.00",
  "accounts_payable": "$0.00",
  "accrued_liabilities": "$0.00",
  "total_assets": "$0.00",
  "total_liabilities": "$0.00",
  "total_equity": "$0.00",
  "accounting_identity_valid": true,
  "identity_difference": "$0.00",
  "timestamp": "2025-08-09T08:59:12.345678+00:00"
}
```

## GET /api/metrics/bsr

Returns BSR engine snapshot with per-product indices and market EMA metrics. Indices are numbers or null. Market EMA values are serialized as strings.

Response shape:
```json
{
  "products": [
    {
      "asin": "B00EXAMPLE",
      "velocity_index": 0.874321,
      "conversion_index": 1.132211,
      "composite_index": 0.994201
    },
    {
      "asin": "B00NEWITEM",
      "velocity_index": null,
      "conversion_index": null,
      "composite_index": null
    }
  ],
  "market_ema_velocity": "2.753219",
  "market_ema_conversion": "0.635112",
  "competitor_count": 2,
  "timestamp": "2025-08-09T08:59:12.345678+00:00"
}
```

## GET /api/metrics/fees

Returns fee totals, counts, and integer-cents averages by fee type, aggregated from SaleOccurred.fee_breakdown across events.

Response shape:
```json
{
  "referral": {
    "total_amount": "$3.00",
    "count": 1,
    "average_amount": "$3.00"
  },
  "fba": {
    "total_amount": "$4.00",
    "count": 1,
    "average_amount": "$4.00"
  }
}
```

## Integration and Usage

- These endpoints are served by the Metrics builder on the backend service layer and can be exercised with in-process ASGI testing.
- Example (httpx AsyncClient):
```python
from httpx import AsyncClient
# app constructed via the service's build_app(); run without network server
async with AsyncClient(app=app, base_url="http://test") as client:
    r = await client.get("/api/metrics/audit")
    assert r.status_code == 200
```

## Source and Wiring

- Audit metrics are retrieved from FinancialAuditService live counters; position comes from DoubleEntryLedgerService when wired.
- Ledger metrics are derived from DoubleEntryLedgerService.get_financial_position().
- BSR metrics come from BsrEngineV3Service.get_snapshot()/get_product_indices().
- Fees are aggregated by an internal FeeMetricsAggregatorService subscribing to SaleOccurred on EventBus, keeping running totals and averages by fee type.

Conventions
- Money: Always strings (e.g., "$10.00").
- Indices: Numbers or null.
- Timestamps: ISO strings.