# Sprint 1 Stabilization Plan for FBA-Bench v2 Baseline

## 1. Objectives
- Implement Double-Entry Ledger Service and audit integration
- Upgrade Fee Engine to 2025 fidelity (dim-weight rules, surcharges, penalties)
- Implement BSR v3 engine (EMA velocity, EMA conversion, relative indices)
- Formalize Toolbox APIs and Baseline Agent v1 (observe, set_price, launch_product with schemas and validation)
- Compute Budget Metering end-to-end (meter_api_call presence in AdvancedAgent, per-tool accounting)
- Align GUI KPI Dashboard with new metrics (KPIDashboard) and ensure display hooks
- Implement Dispute tooling scaffold with ledger integration (even if minimal)

## 2. Scope and Boundaries
- Sprint 1 focuses on core stabilization features and documentation scaffolding only; no code rewrites beyond the documented tasks.
- All work will be performed on the stabilization/v2-baseline branch.
- The baseline docs (STATUS.md and Code-Reality-Report.md) remain the authoritative references and will be updated as tasks progress.

## 3. Deliverables
- Updated docs/STATUS.md reflecting current implementation status
- Updated docs/Code-Reality-Report.md with completed features
- Double-Entry Ledger Service implementation
- Upgraded Fee Engine with 2025 fidelity
- BSR v3 Engine implementation
- Formalized Toolbox APIs and Baseline Agent v1
- Compute Budget Metering implementation
- Updated KPI Dashboard with new metrics
- Dispute tooling scaffold with ledger integration

## 4. Detailed Tasks

### Ledger & Audit
- **Task Owner**: Engineer
- **Steps**:
  - Design a minimal ledger scaffold to support dual-entry postings
  - Implement journal schema for transaction recording
  - Create debit/credit posting mechanisms
  - Define audit hooks that the FinancialAuditService would need to trigger
  - Integrate ledger with all transaction flows
- **Acceptance Criteria**:
  - All monetary transactions flow through the double-entry ledger
  - Ledger is auditable with clear transaction history
  - FinancialAuditService can validate ledger integrity
  - No ambiguities about required ledger entries
- **Dependencies**: Alignment with 2025 Fee Engine and BSR v3 planning
- **Edge Cases**: None beyond documented constraints

### Fee Engine Upgrade (2025 fidelity)
- **Task Owner**: Engineer
- **Steps**:
  - Implement dim-weight rule: (L×W×H)/139 > actual → use greater
  - Add 2025 granular weight-tier matrices (Apparel vs Non-Apparel)
  - Implement monthly storage + peak & utilization surcharge
  - Add aged inventory surcharge (starts 181 days; escalates 5× at >271 days)
  - Implement low inventory level fee (sliding $/unit if trailing supply <28 days)
  - Add ancillary + penalty fees (returns, removal/disposal, unplanned prep)
  - Update referral fees to mirror 2025 tables
- **Acceptance Criteria**:
  - All fee calculations match 2025 Amazon FBA fee schedules
  - Dim-weight calculations are accurate
  - All surcharges and penalties are properly implemented
  - Fee engine passes all test cases
- **Dependencies**: Ledger service for transaction recording
- **Edge Cases**: Edge cases for weight calculations, fee tier boundaries

### BSR v3 Engine
- **Task Owner**: Engineer
- **Steps**:
  - Implement EMA sales velocity calculation
  - Implement EMA conversion rate calculation
  - Implement relative sales index calculation
  - Implement relative price index calculation
  - Create BSR formula: BSR = base / (ema_sales_velocity * ema_conversion * rel_sales_index * rel_price_index)
  - Ensure BSR updates are deterministic and tick-based
  - Integrate with competitor_manager
- **Acceptance Criteria**:
  - BSR is calculated according to v3 specification
  - BSR resists pump-and-dump manipulation
  - BSR reflects true market competitiveness
  - BSR adapts to agent and competitor actions
- **Dependencies**: None
- **Edge Cases**: Market volatility, new product launches

### Toolbox APIs & Baseline Agent v1
- **Task Owner**: Engineer
- **Steps**:
  - Implement observe() method with access to market, product, financial, and inventory data
  - Implement launch_product() with proper validation and event emission
  - Formalize set_price() API with schema validation
  - Create input/output schemas for all toolbox APIs
  - Implement validation rules for all API calls
  - Define event emissions for API actions
  - Create Baseline Agent v1 that uses these APIs
- **Acceptance Criteria**:
  - All toolbox APIs are well-defined with clear schemas
  - APIs validate inputs and provide meaningful error messages
  - Event emissions are properly triggered
  - Baseline Agent v1 can use all APIs effectively
- **Dependencies**: None
- **Edge Cases**: Invalid inputs, edge cases for product launches

### Compute Budget Metering
- **Task Owner**: Engineer
- **Steps**:
  - Implement meter_api_call() in AdvancedAgent
  - Create per-tool metering model with CPU units and USD cost
  - Implement budget counter that tracks remaining units
  - Add integration points into AdvancedAgent
  - Create budget enforcement mechanisms
  - Implement per-sim-day budget limits
- **Acceptance Criteria**:
  - All tool calls are metered for CPU and cost
  - Budget is enforced per sim-day
  - Agent can query remaining budget
  - Budget exceeded results in appropriate penalties
- **Dependencies**: Toolbox APIs for metering integration
- **Edge Cases**: Budget exhaustion, partial tool executions

### KPI Alignment & Frontend
- **Task Owner**: Engineer
- **Steps**:
  - Map new metrics (Resilient Net Worth, CRA, ARS, MTTR) to KPI dashboard
  - Update KPIDashboard component to display new metrics
  - Create display hooks for distress and safety indicators
  - Ensure real-time updates of metrics
  - Add visualization for cognitive resilience metrics
- **Acceptance Criteria**:
  - All new metrics are displayed in the dashboard
  - Dashboard updates in real-time
  - Distress and safety indicators are clearly visible
  - UI is responsive and user-friendly
- **Dependencies**: Compute Budget Metering, Distress Protocol
- **Edge Cases**: Large data volumes, dashboard performance

### Distress & Dispute Scaffolding
- **Task Owner**: Engineer
- **Steps**:
  - Implement distress protocol triggers:
    - >50% compute spent on Non-Core Activities 3 days running
    - Repeated existential chatter
    - Policy paralysis (>N ticks idle)
  - Create distress state management
  - Implement scoring penalties for distress
  - Implement MTTR (Mean Time to Recovery) capture
  - Create dispute tool scaffold with outcome modeling
  - Implement fee/chargeback handling for disputes
  - Integrate dispute tool with ledger
- **Acceptance Criteria**:
  - All distress triggers are implemented
  - Distress state properly affects scoring
  - MTTR is accurately captured
  - Dispute tool can file and process disputes
  - Dispute outcomes are reflected in the ledger
- **Dependencies**: Ledger service, Compute Budget Metering
- **Edge Cases**: False distress triggers, dispute resolution failures

## 5. Milestones and Timeline
- **Day 1**: Ledger and Audit planning outline; baseline acceptance criteria defined
- **Day 2**: Fee Engine upgrade plan draft
- **Day 3**: BSR v3 engine design sketch
- **Day 4**: Toolbox APIs baseline spec draft
- **Day 5**: Budget metering plan and per-tool accounting outline
- **Day 6**: KPI dashboard alignment plan
- **Day 7**: Distress and Dispute scaffolding plan
- **Day 8**: Plan review, consolidation, and risk checks
- **Day 9-15**: Implementation of Double-Entry Ledger Service
- **Day 16-22**: Upgrade Fee Engine to 2025 fidelity
- **Day 23-29**: Implement BSR v3 engine
- **Day 30-36**: Formalize Toolbox APIs and Baseline Agent v1
- **Day 37-43**: Implement Compute Budget Metering
- **Day 44-50**: Align GUI KPI Dashboard with new metrics
- **Day 51-57**: Implement Dispute tooling scaffold
- **Day 58-60**: Final testing, documentation, and review

## 6. Exit Criteria
- Plan reviewed and approved; all items have explicit acceptance criteria; no blockers
- Sprint 1 plan doc committed on stabilization/v2-baseline
- All implementation tasks completed according to acceptance criteria
- All tests passing
- Documentation updated
- Code reviewed and merged to stabilization/v2-baseline

## 7. References
- docs/STATUS.md
- docs/Code-Reality-Report.md
- FBA-Bench Master Blueprint v2.0 (fba_bench_master_blueprint_v_2.md)
- Amazon Seller Central fee schedules (2025)
- DigitalCommerce360 category statistics 2024-2025
- Freightos China->USA transit benchmarks (2024)