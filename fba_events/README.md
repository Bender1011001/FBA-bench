# fba_events

Split from the monolithic event schema file into a maintainable package.

## Modules

- `base.py`: BaseEvent
- `time_events.py`: TickEvent
- `competitor.py`: CompetitorState, CompetitorPricesUpdated
- `sales.py`: SaleOccurred
- `pricing.py`: SetPriceCommand, ProductPriceUpdated
- `inventory.py`: InventoryUpdate, LowInventoryEvent, WorldStateSnapshotEvent
- `budget.py`: BudgetWarning, BudgetExceeded, ConstraintViolation
- `adversarial.py`: AdversarialEvent, PhishingEvent, MarketManipulationEvent, ComplianceTrapEvent, AdversarialResponse
- `skills.py`: SkillActivated, SkillActionGenerated, SkillConflictDetected, MultiDomainDecisionMade
- `agent.py`: AgentDecisionEvent
- `customer.py`: CustomerMessageReceived, NegativeReviewEvent, ComplaintEvent, RespondToCustomerMessageCommand
- `supplier.py`: SupplierResponseEvent, PlaceOrderCommand
- `marketing.py`: MarketTrendEvent, RunMarketingCampaignCommand
- `reporting.py`: ProfitReport, LossEvent

## Usage

```python
from fba_events import EVENT_TYPES, get_event_type, SaleOccurred

evt_cls = get_event_type("SaleOccurred")
```

This preserves the original registry semantics while enabling focused imports per domain.
