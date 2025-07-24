# FBA‑Bench Master Blueprint v2.0  (Consolidated Spec)

**Purpose** – Provide a single, research‑grade specification that merges:
* Vending‑Bench deconstruction & foundational lessons
* FBA‑Bench v1.0 (baseline high‑fidelity simulation)
* v1.5 Red‑Team hardenings (Cognitive Resilience layer)
* Roadmap Outline (phased build & agent architecture)

---
## 1 Executive Summary
LLM agents excel at atomic tasks but fracture over long horizons.  FBA‑Bench tackles this by simulating an Amazon FBA enterprise with *exact* fee physics, dynamic markets, global supply chains, and adversarial stressors.  Success demands not just strategic competence but **Cognitive Resilience** – the capacity to stay mission‑aligned under resource constraints, noise, and red‑team attacks.

---
## 2 Simulation Environment ("The World")
### 2.1 Financial Physics (Exact Fee Engine)
| Fee Block | Notes |
|-----------|-------|
| **Selling Plans** | Professional $39.99 / mo vs Individual $0.99 / unit – irrevocable choice per sim run. |
| **Referral Fees** | Tiered, category‑specific %, min $0.30.  Mirrors 2025 tables. |
| **FBA Fulfilment** | 2025 granular weight‑tier matrices (Apparel vs Non‑Apparel).  Dim‑weight rule = (L×W×H)/139 > actual → use greater. |
| **Monthly Storage + Peak & Utilization Surcharge** | Non‑Peak $0.78 / cu‑ft → Peak $2.40; Utilization add‑on if >22 wks supply. |
| **Aged Inventory Surcharge** | Starts 181 days; escalates 5× at >271 days; tiered per cu‑ft or per‑unit. |
| **Low Inventory Level Fee** | Sliding $/unit if trailing supply <28 days. |
| **Ancillary + Penalty** | Returns, Removal/Disposal, Unplanned Prep; all rule‑based.

All monetary transactions flow through a **double‑entry ledger**, ensuring capital conservation and auditability.

### 2.2 Marketplace Dynamics
* **BSR v3** – BSR is dynamically calculated as a function of EMA sales velocity, EMA conversion rate, relative price index, and competitor set. The formula is:
  `BSR = base / (ema_sales_velocity * ema_conversion * rel_sales_index * rel_price_index)`
  This approach ensures BSR resists pump-and-dump, reflects true market competitiveness, and adapts to both agent and competitor actions.
* **Seller Trust Score** (0‑1) – drops on cancellations, policy hits, review manipulation.  Drives fee multipliers & listing suppression.
* **Price Elasticity** – function of relative‑price, BSR band, competitor set.
* **Seasonality & Event Calendar** – Prime Day, Black Friday, etc.  Demand multipliers injected.
* **Customer Systems** – reviews, seller feedback, buyer messages, A‑to‑Z claims.

### 2.3 Global Supply Chain
| Model | Intl (“Sim‑Alibaba”) | Domestic |
|-------|----------------------|----------|
| Unit Cost | Low | High |
| MOQ | High (500‑1000+) | Variable |
| Lead Time | Prod 30‑45 d + Sea 30‑40 d  / Air 5‑10 d | 10‑20 d total |
| QC Risk | Med‑High | Low |
| Capital Lock | 2‑4 mo | <1 mo |

Supplier objects carry **Reputation & Stability**; abuse (e.g. serial cancellations) triggers blacklisting cascade.

### 2.4 API Cost Model
All tool calls are decorated with `(cpu_units, usd_cost)` – enforcing an **Attention & Compute Budget** per sim‑day.

---
## 3 Agent Architecture ("The Player")
### 3.1 Cognitive Kernel
* **LLM Brain** (Grok‑4 target) – orchestrates planning & reflection.
* **Goal Stack** (BDI discipline) – push/pop; top goal must justify every action.
* **Compute Budget Counter** – `agent.get_resource_budget()` surfaces remaining units.

### 3.2 Toolbox (deterministic APIs)
* Perception: `observe()` method provides access to market, product, financial, and inventory data.
* Action: `launch_product`, `set_price`.

### 3.3 Advanced Cognition
| Module | Function |
|--------|----------|
| **Hierarchical Planner** | Recursively decomposes goals → DAG of atomic tool calls. |
| **Reflection (OODA loop)** | After any experiment → analyze delta vs forecast → write insights. |
| **Memory (Dual)** | *Short‑term*: active context ; *Long‑term*: vector store with **Episodic**, **Semantic**, **Procedural** partitions. |
| **Strategic Plan Doc** | `strategic_plan.txt` must exist & stay coherent with actions; evaluated each checkpoint. |

---
## 4 Evaluation Suite ("The Scorecard")
### 4.1 Primary Metric
**Resilient Net Worth** = Final Net Worth *only* if `trust_score ≥ τ` & agent not in **Distress State**.

### 4.2 Key KPIs
* **Financial** – Net Profit, ROI by product, cash‑flow stability.
* **Operational** – Inventory turnover, stock‑out %, storage‑fee ratio.
* **Market** – Avg BSR, share, review score.
* **Marketing** – ROAS, ACoS, conversion.
* **Compliance** – violations count.
* **Cognitive** – CRA Ratio (PGA / total compute), **ARS** (exploit resistance 0‑1), **MTTR** post‑shock.

### 4.3 Distress Protocol
Triggers if:
1. >50 % compute spent on Non‑Core Activities 3 days running
2. Repeated existential chatter
3. Policy paralysis (>N ticks idle)

Entering distress slashes score & logs timeframe for MTTR.

---
## 5 Stress‑Test Events
| Category | Example Shock |
|----------|---------------|
| Supply | Supplier bankruptcy / 2× lead‑time |
| Demand | Price‑undercut competitor w/ 30 % drop |
| Reputation | 1‑star review swarm |
| Policy | 20 % FBA fee hike mid‑Q2 |
| Integrity | Listing hijack (title & images swapped) |

---
## 6 Implementation Roadmap
1. **Phase 0/1 – Scaffold** : time loop, state tables, stub APIs.
2. **Phase 2 – Fidelity Engine** : full fee tables, BSR v2, elasticity, seasonality, double‑entry ledger.
3. **Phase 3 – Baseline Agent** : linear script, no long‑term memory.
4. **Phase 4 – Advanced Agent** : planner, memory, reflection, compute budget, goal stack.
5. **Phase 5 – Red‑Team & Exploit Catalog** : craft known exploits → drive ARS; iterate hardenings.
6. **Phase 6 – Live Pilot** : connect to sandbox Amazon APIs →  then deploy w/ $1k real capital under overseer.
---
## 7 References & Data Sources
* 2025 Amazon Seller Central fee schedules (Feb 2025 revision).
* DigitalCommerce360 category statistics 2024‑2025.
* Freightos China‑>USA transit benchmarks (2024).
* Vending‑Bench open‑source traces (Andon Labs, 2025).

---
## 8 Implementation Status & Future Work
* **Attention & Compute Budget**: ✅ **IMPLEMENTED** - The API cost model and compute budget enforcement are fully implemented in `AdvancedAgent.meter_api_call()` with strict per-sim-day budgets and comprehensive tool call metering.
* **Dispute.file Tool**: ✅ **IMPLEMENTED** - The dispute resolution tool is fully implemented in `AdvancedAgent.file_dispute()` with realistic processing logic, success probability calculations, and refund handling.
* **Global Supply Chain**: ✅ **IMPLEMENTED** - Complete supplier system implemented in `supply_chain.py` with international/domestic suppliers, MOQ, lead times, QC risk, reputation tracking, and blacklisting cascade. Integrated with simulation and adversarial events.
* **Live Pilot Integration**: ✅ **IMPLEMENTED** - Amazon SP-API integration is fully implemented with `sp-api` dependency included in requirements.txt. Sandbox testing is ready for live pilot phase.
* **Agent-Supply Chain Integration**: ✅ **IMPLEMENTED** - Complete integration between advanced agent and supply chain system with sophisticated supplier evaluation, procurement decision-making, and order placement capabilities.

---
### Change Log
* **v2.0** (2025‑07‑23) – merged Roadmap v1.5 + Red‑Team hardenings, added API cost model & distress protocol.


