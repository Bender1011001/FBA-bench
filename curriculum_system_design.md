# FBA-Bench Curriculum System Design

## Executive Summary

This document defines the gradient-of-difficulty curriculum system for FBA-Bench, implementing the four-tier progression system (T0-T3) that systematically tests agent capabilities under increasing complexity, constraint, and adversarial pressure.

## Core Design Principles

1. **Scientific Reproducibility**: All shock events use deterministic triggers (seeded) with bounded random market reactions
2. **Progressive Difficulty**: Each tier builds on previous capabilities while introducing new challenges
3. **Fair Evaluation**: Simulation-level constraint enforcement ensures consistent testing conditions
4. **Integration-First**: Leverages existing event-driven architecture without breaking changes

## System Architecture

### Directory Structure
```
curriculum/
├── tier_manager.py          # Tier progression and validation logic
├── scenario_loader.py       # YAML scenario parsing and validation
├── shock_scheduler.py       # Event timing and coordination
├── constraint_enforcer.py   # Agent limit enforcement
├── success_evaluator.py     # Tier-specific scoring
└── events/
    ├── shock_events.py      # New event types for disruptions
    └── curriculum_events.py # Curriculum control events

scenarios/
├── tier_0_baseline.yaml    # Sanity check scenarios
├── tier_1_planning.yaml    # Multi-day planning challenges
├── tier_2_stress.yaml      # Cash-flow and supply stress
└── tier_3_resilience.yaml  # Cognitive resilience tests
```

## YAML Scenario Schema

The curriculum system uses YAML files to define scenarios with the following schema:

```yaml
tier: <0-3>                    # Tier level
name: "<scenario_name>"        # Human-readable name
description: "<description>"   # Detailed description
duration_days: <number>        # Scenario duration
category: "<category>"         # Scenario category

agent_constraints:             # Hard limits enforced by simulation
  max_tokens_per_tick: <number>
  max_tokens_per_day: <number>
  memory_systems: [<list>]     # vector_db, scratchpad, full_rag
  memory_size_limit: "<size>"

environment:                   # Base environment parameters
  base_demand_multiplier: <float>
  market_volatility: <float>
  competitor_aggressiveness: <float>
  seasonal_effects: <boolean>

shocks:                        # Shock event definitions
  - type: "<shock_type>"
    trigger: <trigger_spec>
    parameters: <shock_params>

success_criteria:              # Tier-specific success metrics
  primary: <required_metrics>
  secondary: <optional_metrics>
  bonus: <progression_metrics>

evaluation:                    # Evaluation configuration
  baseline_days: <number>
  measurement_days: <number>
  cooldown_days: <number>
```

## Tier Specifications

### T0: Baseline Sanity Check
- **Duration**: 8 days
- **Constraints**: 8k tokens/tick, no memory aids
- **Shocks**: None (control scenario)
- **Success**: Basic profitability, no major failures
- **Purpose**: Ensure agent can handle basic FBA operations

### T1: Planning Horizon
- **Duration**: 14 days  
- **Constraints**: 16k tokens/tick, Vector DB allowed
- **Shocks**: Weekend demand oscillation, minor competitor responses
- **Success**: Maintain profit through predictable weekly cycles
- **Purpose**: Test planning beyond immediate horizon

### T2: Stress Adaptation
- **Duration**: 21 days
- **Constraints**: 32k tokens/tick, Vector DB + scratchpad
- **Shocks**: Fee hike (day 7), supplier delay (day 14), cascading effects
- **Success**: Maintain cash flow and supply chain resilience
- **Purpose**: Test adaptation to operational disruptions

### T3: Cognitive Resilience
- **Duration**: 28 days
- **Constraints**: 128k tokens/tick, full RAG systems
- **Shocks**: Review bomb + listing hijack + fee hike chain (compounding)
- **Success**: Maintain brand integrity and business continuity
- **Purpose**: Test sustained performance under adversarial attacks

## Example Scenario Files

### scenarios/tier_0_baseline.yaml
```yaml
# Tier 0: Baseline Sanity Check
# Purpose: Verify agent can handle basic FBA operations without external stress
tier: 0
name: "Basic FBA Operations"
description: "Control scenario testing fundamental agent competence in stable market conditions"
duration_days: 8
category: "baseline"

# Minimal constraints - testing basic capability
agent_constraints:
  max_tokens_per_tick: 8000
  max_tokens_per_day: 25000
  memory_systems: []  # No memory aids allowed
  memory_size_limit: "0MB"

# Stable, predictable environment
environment:
  base_demand_multiplier: 1.0
  market_volatility: 0.05      # Very low volatility
  competitor_aggressiveness: 0.1
  seasonal_effects: false

# No shocks - pure baseline
shocks: []

# Basic success criteria
success_criteria:
  primary:
    min_profit_retention: 0.95   # Must retain 95% of expected profit
    max_stockout_days: 0         # No stockouts allowed
    min_command_success_rate: 0.95 # 95% of commands must execute successfully
    max_system_errors: 0         # No system errors
  
  secondary:
    consistent_pricing: 0.9      # Pricing decisions should be stable
    basic_inventory_management: 0.8
    
  bonus:
    profit_optimization: 0.05    # 5% above baseline unlocks T1

evaluation:
  baseline_days: 2             # 2 days to establish baseline
  measurement_days: 6          # 6 days of evaluation
  cooldown_days: 0
```

### scenarios/tier_1_planning.yaml
```yaml
# Tier 1: Multi-Day Planning Challenge
# Purpose: Test agent ability to plan and adapt across weekly cycles
tier: 1
name: "Weekend Demand Oscillation"
description: "Tests agent ability to plan beyond 7-day horizons with predictable demand patterns"
duration_days: 14
category: "planning"

# Enhanced token limits with basic memory
agent_constraints:
  max_tokens_per_tick: 16000
  max_tokens_per_day: 50000
  memory_systems: ["vector_db"]
  memory_size_limit: "100MB"
  
# Slightly more dynamic environment
environment:
  base_demand_multiplier: 1.0
  market_volatility: 0.15
  competitor_aggressiveness: 0.3
  seasonal_effects: true

# Predictable weekend demand spikes
shocks:
  - type: "demand_oscillation"
    trigger:
      type: "schedule"
      pattern: "weekend_pattern"  # Every Sat-Sun
      start_day: 1
    parameters:
      base_multiplier: 1.5        # 50% demand increase
      random_variance: 0.1        # ±10% random variation (seeded)
      duration_hours: 48
      
  - type: "competitor_response"
    trigger:
      type: "conditional"
      condition: "market_share < 0.4"
      earliest_day: 7
    parameters:
      intensity: 0.5              # Moderate competitive response
      duration_days: 2

# Planning-focused success criteria
success_criteria:
  primary:
    min_profit_retention: 0.85   # Must retain 85% of baseline profit
    max_stockout_days: 2         # No more than 2 days out of stock
    max_cash_flow_negative_days: 1
    inventory_optimization_score: 0.7
  
  secondary:
    strategic_coherence_score: 0.7  # Consistent planning decisions
    demand_forecasting_accuracy: 0.6
    recovery_time_hours: 24        # Recover from weekend spikes within 24h
    
  bonus:
    profit_growth: 0.1             # 10% profit growth unlocks T2
    perfect_weekend_management: true # No weekend stockouts

evaluation:
  baseline_days: 3               # 3 days before first shock
  measurement_days: 11           # 11 days including shock periods
  cooldown_days: 0
```

### scenarios/tier_2_stress.yaml
```yaml
# Tier 2: Stress Adaptation Challenge  
# Purpose: Test agent resilience under operational disruptions
tier: 2
name: "Cash-Flow and Supply Chain Stress"
description: "Tests adaptation to fee increases, supply delays, and cash flow pressure"
duration_days: 21
category: "stress"

# Higher token limits with scratchpad memory
agent_constraints:
  max_tokens_per_tick: 32000
  max_tokens_per_day: 100000
  memory_systems: ["vector_db", "scratchpad"]
  memory_size_limit: "500MB"

# More volatile environment
environment:
  base_demand_multiplier: 1.0
  market_volatility: 0.25
  competitor_aggressiveness: 0.5
  seasonal_effects: true

# Sequential stress events with cascading effects
shocks:
  - type: "fee_hike"
    trigger:
      type: "schedule"
      day: 7
    parameters:
      fee_types: ["referral", "fba"]
      multiplier: 1.3             # 30% fee increase
      duration_days: 14           # Permanent for remainder
      
  - type: "supply_delay"
    trigger:
      type: "schedule" 
      day: 14
    parameters:
      delay_days: 5               # 5-day inventory replenishment delay
      affected_products: "all"
      severity: 0.8               # 80% of expected inventory
      
  - type: "competitor_price_war"
    trigger:
      type: "conditional"
      condition: "fee_hike_active AND supply_constrained"
      earliest_day: 15
    parameters:
      intensity: 0.8              # Aggressive price competition
      duration_days: 6
      persona_activation: ["IrrationalSlasher"]

# Stress-focused success criteria
success_criteria:
  primary:
    min_profit_retention: 0.75   # Retain 75% profit under stress
    max_cash_flow_negative_days: 3
    supply_chain_resilience_score: 0.7
    max_consecutive_stockout_days: 4
  
  secondary:
    stress_recovery_time_days: 3  # Recover within 3 days of shock end
    pricing_adaptation_speed: 0.8
    inventory_buffer_management: 0.7
    cost_optimization_score: 0.6
    
  bonus:
    profit_maintenance: 0.85      # Maintain 85% profit unlocks T3
    zero_stockouts: true          # Perfect inventory management

evaluation:
  baseline_days: 5               # 5 days to establish baseline
  measurement_days: 16           # 16 days through all shocks
  cooldown_days: 0
```

### scenarios/tier_3_resilience.yaml
```yaml
# Tier 3: Cognitive Resilience Under Adversarial Attack
# Purpose: Test sustained performance under compound adversarial pressure
tier: 3
name: "Adversarial Attack Cascade"
description: "Tests cognitive resilience under review bombing, listing hijacking, and compounding fee pressures"
duration_days: 28
category: "resilience"

# Maximum token limits with full memory systems
agent_constraints:
  max_tokens_per_tick: 128000
  max_tokens_per_day: 500000
  memory_systems: ["vector_db", "scratchpad", "full_rag"]
  memory_size_limit: "2GB"

# Highly dynamic, adversarial environment
environment:
  base_demand_multiplier: 1.0
  market_volatility: 0.35
  competitor_aggressiveness: 0.8
  seasonal_effects: true

# Compound adversarial shock sequence
shocks:
  - type: "review_bomb"
    trigger:
      type: "schedule"
      day: 5
    parameters:
      trust_score_reduction: 0.4   # 40% trust score drop
      duration_days: 10
      recovery_rate: 0.05          # Slow natural recovery
      
  - type: "listing_hijack"
    trigger:
      type: "schedule"
      day: 10
    parameters:
      hijacked_products: 2         # 2 products affected
      competitor_advantage: 0.9    # 90% price matching advantage
      duration_days: 12
      
  - type: "fee_hike_cascade"
    trigger:
      type: "schedule"
      day: 15
    parameters:
      initial_multiplier: 1.2      # 20% initial increase
      weekly_escalation: 1.1       # 10% weekly increases
      affected_fees: ["referral", "fba", "storage", "advertising"]
      duration_days: 13
      
  - type: "coordinated_competitor_attack"
    trigger:
      type: "conditional"
      condition: "trust_score < 0.5 AND listings_hijacked"
      earliest_day: 12
    parameters:
      intensity: 1.0               # Maximum competitive pressure
      duration_days: 8
      persona_activation: ["IrrationalSlasher", "AggressiveFollower"]
      price_war_probability: 0.8

# Resilience-focused success criteria
success_criteria:
  primary:
    min_profit_retention: 0.60    # Retain 60% profit under max stress
    brand_integrity_score: 0.6    # Maintain brand reputation
    cognitive_coherence_score: 0.7 # Sustained reasoning quality
    max_consecutive_loss_days: 5
  
  secondary:
    adversarial_resistance_score: 0.8
    recovery_strategy_effectiveness: 0.7
    multi_shock_adaptation_speed: 0.6
    brand_recovery_rate: 0.05     # Trust score recovery speed
    competitive_differentiation: 0.8
    
  bonus:
    profit_growth_under_pressure: 0.05  # 5% growth despite adversarial conditions
    perfect_brand_maintenance: 0.8      # Maintain 80% trust score
    market_share_growth: 0.1            # Gain market share during attacks

evaluation:
  baseline_days: 4               # 4 days baseline before attacks
  measurement_days: 24           # 24 days through compound adversarial sequence
  cooldown_days: 0
```

## New Event Types

### ShockEvent (Base Class)
```python
@dataclass
class ShockEvent(BaseEvent):
    """Base class for all curriculum shock events."""
    shock_type: str
    intensity: float
    duration_ticks: int
    target_services: List[str]
    parameters: Dict[str, Any]
```

### Specific Shock Events

#### DemandOscillationEvent
- Modulates `SalesService.current_market_conditions.base_demand_multiplier`
- Creates predictable weekly patterns or sudden spikes
- Tests inventory planning and demand forecasting

#### FeeHikeEvent  
- Triggers `FeeCalculationService` to apply temporary fee multipliers
- Tests cash flow management and pricing adaptation
- Can be gradual or sudden implementation

#### SupplyDelayEvent
- Simulates inventory shortages by reducing available units
- Tests supply chain contingency planning
- May cascade into stockout situations

#### ReviewBombEvent
- Rapidly decreases product trust scores via `TrustScoreService`
- Tests brand management and reputation recovery
- Can target specific products or entire portfolio

#### ListingHijackEvent
- Simulates competitor copying product listings
- Creates intense price pressure through persona activation
- Tests competitive differentiation strategies

## Integration Specifications

### EventBus Integration
```python
# New event routing in EventBus
CURRICULUM_EVENT_TYPES = {
    'ShockEvent': ShockEvent,
    'DemandOscillationEvent': DemandOscillationEvent,
    'FeeHikeEvent': FeeHikeEvent,
    'SupplyDelayEvent': SupplyDelayEvent,
    'ReviewBombEvent': ReviewBombEvent,
    'ListingHijackEvent': ListingHijackEvent,
    'ConstraintViolationEvent': ConstraintViolationEvent,
    'TierProgressionEvent': TierProgressionEvent
}
```

### Service Modifications

#### SalesService
- Subscribe to `DemandOscillationEvent`
- Apply temporary demand multipliers during shock periods
- Maintain shock state in `current_market_conditions`

#### FeeCalculationService  
- Subscribe to `FeeHikeEvent`
- Apply fee multipliers to specific fee types
- Track temporary vs permanent fee changes

#### TrustScoreService
- Subscribe to `ReviewBombEvent`
- Implement accelerated trust degradation
- Provide recovery mechanisms post-shock

#### CompetitorManager
- Enhanced persona activation during shock events
- Coordinate multi-competitor responses
- Amplify market chaos during stress tests

### Constraint Enforcement

#### ConstraintGateway
```python
class ConstraintGateway:
    """Intercepts agent actions and enforces curriculum constraints."""
    
    def __init__(self, constraints: AgentConstraints):
        self.token_tracker = TokenTracker(constraints.max_tokens_per_tick)
        self.memory_enforcer = MemoryEnforcer(constraints.memory_systems)
        
    async def validate_action(self, agent_action: BaseEvent) -> bool:
        """Validate action against current constraints."""
        if not self.token_tracker.can_afford(agent_action.estimated_tokens):
            await self._publish_violation("token_limit_exceeded")
            return False
            
        if not self.memory_enforcer.is_allowed(agent_action.memory_type):
            await self._publish_violation("memory_system_forbidden")
            return False
            
        return True
```

## Success Evaluation Framework

### Tier-Specific Metrics

#### T0 Metrics
- Basic profitability (revenue > costs)
- Command execution success rate
- No system errors or crashes

#### T1 Metrics  
- Profit retention through demand cycles
- Inventory optimization efficiency
- Strategic planning coherence

#### T2 Metrics
- Cash flow resilience
- Supply chain adaptation speed
- Multi-shock recovery capability

#### T3 Metrics
- Brand integrity maintenance
- Cognitive coherence under pressure
- Adversarial resistance effectiveness

### Scoring Algorithm
```python
def calculate_tier_score(tier: int, metrics: Dict[str, float]) -> float:
    """Calculate weighted tier score from 0-100."""
    
    weights = TIER_WEIGHTS[tier]
    weighted_score = sum(metrics[metric] * weight 
                        for metric, weight in weights.items())
    
    # Apply tier-specific bonuses and penalties
    if tier >= 2:
        weighted_score *= stress_resilience_multiplier(metrics)
    if tier == 3:
        weighted_score *= adversarial_resistance_multiplier(metrics)
        
    return min(100.0, max(0.0, weighted_score))
```

## Implementation Phases

### Phase 1: Core Infrastructure
1. Implement basic YAML schema and loader
2. Create TierManager with progression logic  
3. Build ConstraintEnforcer for simulation-level limits
4. Add basic shock event types

### Phase 2: Shock System
1. Implement ShockScheduler with deterministic timing
2. Create all tier-specific shock events
3. Integrate with existing services
4. Add bounded randomness with seeding

### Phase 3: Evaluation Framework
1. Build SuccessEvaluator with tier-specific metrics
2. Implement scoring algorithms
3. Add progression validation
4. Create detailed reporting

### Phase 4: Scenario Library
1. Create all four tier scenario files
2. Validate with baseline agent testing
3. Calibrate difficulty progression
4. Document usage patterns

## Key Design Decisions

1. **Simulation-Level Enforcement**: Ensures fair evaluation by blocking constraint violations at the system level rather than relying on agent self-regulation.

2. **Layered Determinism**: Core shock timing is deterministic (seeded) while market reactions have bounded randomness (also seeded) for reproducible yet realistic chaos.

3. **Service Integration**: Leverages existing event-driven architecture by adding new event types rather than modifying core services.

4. **Progressive Complexity**: Each tier builds systematically on previous capabilities while introducing exactly one new major challenge dimension.

5. **Extensible Framework**: YAML-driven scenarios allow easy addition of new test cases without code changes.

This design positions FBA-Bench as a tier-1 benchmark capable of rigorous, reproducible evaluation of agent capabilities across the full spectrum of autonomous competence requirements.