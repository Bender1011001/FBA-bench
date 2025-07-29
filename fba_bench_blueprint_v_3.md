# FBA‑Bench v3 Master Blueprint  
_Revision date: 2025‑07‑28_

---

## 0. Executive Summary  
Version 3 fuses the shipping **v2.0 Master Blueprint** with the architectural and research‑tooling demands of the "Definitive" spec.  The result is a **fully event‑driven, multi‑agent, audit‑tight simulation kernel** wrapped with best‑in‑class experimentation UX.

---

## 1. Guiding Principles
1. **Architectural Purity** – single‑responsibility services, clean API contracts, zero hidden state.  
2. **Uncompromising Financial Integrity** – `Money` strict mode on, ledger invariants checked **per transaction**.  
3. **High‑Fidelity Chaos** – stochastic demand, irrational competitors, external shocks, fog‑of‑war.  
4. **Multi‑Agent First** – sandboxed agents share one event bus; no privileged insight.  
5. **Integrated Research Toolkit** – real‑time dashboard, Jupyter observer, param‑sweep CLI.  
6. **Cognitive‑Resilience Metrics** – CRA, ARS & distress signals built‑in.  
7. **Extensibility & Reproducibility** – deterministic seeds, plug‑in strategy registry.

---

## 2. Core Architecture – Event‑Driven / Service‑Oriented Hybrid
| Layer | Component | Responsibility |
|-------|-----------|----------------|
| **Communication** | **Central Event Bus** (`asyncio.Queue` or RabbitMQ) | Publish/subscribe transport for all messages. |
| **State** | **World Store** (Redis JSONB) | Canonical market state, versioned snapshots. |
| **Services** | Stateless micro‑services (`SalesSvc`, `FeeSvc`, `TrustScoreSvc`, etc.) | Consume events → run logic → emit events.  |
| **Bridges** | **Legacy Phase‑Loop Adapter** | Lets existing v2 time‑loop tests run on the bus. |
| **Agents** | Sandboxed processes/containers | Subscribe to perception topics, publish command events. |
| **Observability** | Structured logging, OpenTelemetry traces | Event lineage & debugging. |

### 2.1 Event Taxonomy (minimum set)
* `TickEvent` – heartbeat / time advance.
* `SaleOccurred` – order completed.
* `SetPriceCommand` – agent intent.
* `ExternalShockEvent` – macro shock injected.
* `LedgerTxnPosted` – successful double‑entry post.

---

## 3. Financial Core – Purist Money Type & Ledger Audit
* **Factory‑Only Instantiation** – `Money.from_dollars`, `Money.from_cents`.  `_skip_guard` removed.
* **`MONEY_STRICT = True` global** – no floats/ints in financial domain objects.
* **Per‑Txn Audit Hook** – verify `Assets = Liabilities + Equity` after every `LedgerTxnPosted`; abort on violation.

---

## 4. Simulation Realism & Chaos Modules
1. **Competitor Persona Library**  
   * `IrrationalSlasher` – random floor cuts.  
   * `SlowFollower` – lagged reactions.  
   * `BrandPremium` – rigid high price.  
2. **Fog‑of‑War**  
   * KPI latency (6‑12 h) & Gaussian noise.  
   * Scraping API cost + stale probability.
3. **External Shock Injector**  
   * Deterministic replays or live NewsAPI feed (with seed).

---

## 5. Multi‑Agent Framework
* **Sandboxing** – Docker compose or Python `multiprocessing`; each agent gets read‑only env vars and event channels.
* **World Arbitration** – `WorldStore` applies commands, resolves conflicts, emits new state diff.
* **Reputation & History** – `TrustScoreSvc` tracks agent SLA violations, ODR, review manipulation.

---

## 6. Integrated Research Toolkit
1. **Real‑Time Dashboard**  
   * Consumes event bus; Grafana/React front‑end.  
2. **Jupyter Kernel (Observer Mode)**  
   * Read‑only DB creds; allows Pandas/Matplotlib exploration.  
3. **Experiment CLI**  
   * `bench run sweep.yaml` ⇒ parameter grid, auto log aggregation.  
4. **Trace & Replay**  
   * Persist raw event stream for deterministic rerun.

---

## 7. Cognitive‑Resilience Metrics (CRA | ARS | Distress)
* Events tagged with `attention_cost` and `stress_level` outputs per agent step.
* `DistressDetectorSvc` publishes `DistressEvent` on threshold breach.

---

## 8. Testing & Validation
* **Unit + Property Tests** – numeric invariants, Money arithmetic.  
* **Golden Master Tests** – snapshot event streams for version drift detection.  
* **Chaos Tests** – injected shock events verify graceful degradation.  
* **Multi‑Agent Regression** – price war, stockout race, exploit detection.

---

## 9. Implementation Roadmap (90‑day)
| Phase | Weeks | Deliverables |
|-------|-------|--------------|
| **P1 – Bus Core** | 0‑3 | Event schema, in‑mem bus, migrate LoggerSvc. |
| **P2 – Money Hardening** | 3‑5 | Strict Money, remove `_skip_guard`, green tests. |
| **P3 – Service Migration** | 5‑8 | All v2 services on bus; legacy adapter ready. |
| **P4 – Personas & Fog** | 8‑11 | Persona lib, KPI delay/noise, scrape cost. |
| **P5 – Multi‑Agent & WorldStore** | 11‑14 | Sandbox container harness, TrustScore integration. |
| **P6 – Toolkit Alpha** | 14‑18 | Dashboard v1, Jupyter observer, CLI sweeps. |

---

## 10. Deployment & Ops
* **Docker Compose Reference Stack** – bus, Redis, services, 2 demo agents, dashboard.  
* **K8s Helm Charts** – optional production scaling.  
* **CI/CD** – GitHub Actions, codecov, Dependabot.

---

## 11. Contributor Guide (excerpt)
* Feature PRs **must** include: event schema doc, unit tests, Golden snapshot update, and changelog entry.  
* Use `pre‑commit` hooks for formatting (`black`, `ruff`).  
* Design discussions → GitHub Discussions with ADR template.

---

## 12. License & Governance
* MIT license; amber‑light clause for simulation output used in commercial ML training.  
* Core maintainers: @lead‑architect, @finance‑chair, @research‑ux.  
* Quarterly feature freeze before version tags.

---

### End of Blueprint  
“Build agents tough enough for the bazaar, then unleash the bazaar on them.”

