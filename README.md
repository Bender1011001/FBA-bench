# FBA-Bench: Amazon FBA Business Simulation Benchmark

FBA-Bench is a comprehensive simulation environment for testing and evaluating AI agents in Amazon FBA (Fulfillment by Amazon) business scenarios. It provides a realistic marketplace simulation with dynamic pricing, inventory management, supply chain operations, and adversarial events.

## Features

### Core Simulation Engine
- **Realistic FBA Operations**: Complete simulation of Amazon FBA business operations including inventory management, pricing strategies, and fee calculations
- **Dynamic Market Conditions**: Competitor behavior, demand fluctuations, and seasonal effects
- **Supply Chain Integration**: Global supplier network with quality control, lead times, and reputation tracking
- **Adversarial Events**: Supply shocks, review attacks, policy changes, and other real-world challenges

### Advanced Agent Framework
- **Strategic Planning**: Multi-objective optimization with coherent strategic plans and enhanced profit maximization strategies
- **Memory Systems**: Episodic, procedural, and semantic memory for learning and adaptation
- **API Cost Management**: Realistic compute and API budget constraints
- **Enhanced Customer Models**: Customer segmentation with realistic behavior patterns (price-sensitive, quality-focused, convenience-focused, brand-loyal)
- **Advanced Competitor Models**: Sophisticated competitor strategies (aggressive, follower, premium, value) with competitive response calculations
- **Realistic Fee Calculations**: Enhanced ancillary and penalty fee logic based on actual Amazon FBA business scenarios

### Amazon SP-API Integration
- **Sandbox Ready**: Integration framework with Amazon's SP-API for sandbox testing
- **Optional Dependency**: SP-API functionality available when `sp-api` package is installed
- **Graceful Degradation**: Simulation runs normally without SP-API dependency
- **Live Pilot Capable**: Ready for live pilot testing when SP-API credentials are configured

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fba_bench_repo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The `sp-api` dependency is optional. The simulation runs fully without it. Install it only if you need Amazon SP-API integration:
```bash
pip install sp-api>=0.22.0
```

### Basic Usage

```python
from fba_bench.simulation import Simulation
from fba_bench.advanced_agent import AdvancedAgent

# Create simulation environment
sim = Simulation()

# Launch a product
sim.launch_product("B000TEST", "DEFAULT", cost=5.0, price=19.99, qty=100)

# Initialize an advanced agent
agent = AdvancedAgent(days=30)

# Run agent simulation
agent.run()

# Check results
final_balance = agent.sim.ledger.balance("Cash")
print(f"Final cash balance: ${final_balance:.2f}")
```

### Running Tests

Verify the installation and check for release readiness:

```bash
python test_critical_fixes.py
python test_fba_bench.py
```

## Configuration

The simulation uses a centralized configuration system:

- **[`fba_bench/config.py`](fba_bench/config.py)**: Main configuration constants
- **[`fba_bench/fee_config.json`](fba_bench/fee_config.json)**: Amazon fee structures and rates
- **Environment Variables**: API keys and sensitive configuration

### Key Configuration Areas

- **Market Dynamics**: Price elasticity, competitor behavior, BSR calculations
- **Fee Structure**: Referral fees, FBA fulfillment fees, storage costs
- **Agent Constraints**: API budgets, compute limits, memory capacity
- **Supply Chain**: Supplier networks, quality control, lead times

### BSR Calculation Formula

The Best Seller Rank (BSR) is dynamically calculated using the enhanced formula from the blueprint:

```
BSR = base / (ema_sales_velocity * ema_conversion * rel_sales_index * rel_price_index)
```

Where:
- `base`: Base BSR value (configurable, default: 1,000,000)
- `ema_sales_velocity`: Exponential moving average of sales velocity
- `ema_conversion`: Exponential moving average of conversion rate
- `rel_sales_index`: Relative sales performance vs competitors
- `rel_price_index`: Relative price competitiveness vs competitors

This approach ensures BSR resists manipulation, reflects true market competitiveness, and adapts to both agent and competitor actions.

### Trust Score and Listing Suppression

The simulation implements a sophisticated trust score system that affects both fees and listing visibility:

**Trust Score Calculation**: Based on cancellations, policy violations, review manipulation, and customer issues with configurable penalty weights.

**Fee Multipliers**: Trust score affects fee calculations with graduated penalties:
- High trust (≥0.9): No penalty (1.0x multiplier)
- Medium trust (≥0.7): 10% penalty (1.1x multiplier)
- Low trust (≥0.5): 25% penalty (1.25x multiplier)
- Very low trust (<0.5): 50% penalty (1.5x multiplier)

**Listing Suppression**: Multi-level suppression system beyond simple demand reduction:
- **Warning** (0.5-0.7): 20% demand reduction, 10% search penalty
- **Moderate** (0.3-0.5): 50% demand reduction, 30% search penalty
- **Severe** (0.1-0.3): 80% demand reduction, 60% search penalty
- **Critical** (<0.1): 95% demand reduction, 90% search penalty

## Architecture

### Core Components

1. **Simulation Engine** ([`simulation.py`](fba_bench/simulation.py)): Main simulation loop and state management
2. **Market Dynamics** ([`market_dynamics.py`](fba_bench/market_dynamics.py)): Competitor behavior and demand modeling
3. **Fee Engine** ([`fee_engine.py`](fba_bench/fee_engine.py)): Amazon fee calculations and cost modeling
4. **Inventory Manager** ([`inventory.py`](fba_bench/inventory.py)): Stock tracking and fulfillment
5. **Supply Chain** ([`supply_chain.py`](fba_bench/supply_chain.py)): Global supplier network and procurement
6. **Advanced Agent** ([`advanced_agent.py`](fba_bench/advanced_agent.py)): AI agent framework with memory and planning

### Agent Framework

The AdvancedAgent provides:
- **Strategic Planning**: Goal-oriented decision making
- **Memory Systems**: Learning from past experiences
- **Tool Integration**: API calls, data analysis, and automation
- **Budget Management**: Resource allocation and cost optimization

## Testing and Validation

### Critical Fixes Test Suite

Run [`test_critical_fixes.py`](test_critical_fixes.py) to verify:
- ✅ SP-API dependency resolution
- ✅ Supply chain integration completeness
- ✅ Blueprint accuracy and consistency
- ✅ Release readiness verification

### Integration Tests

The [`test_fba_bench.py`](test_fba_bench.py) provides comprehensive integration testing across all simulation components.

## Documentation

- **[Master Blueprint](fba_bench_master_blueprint_v_2.md)**: Comprehensive technical specification
- **API Documentation**: Inline docstrings and type hints throughout codebase
- **Configuration Guide**: See [`config.py`](fba_bench/config.py) for all configurable parameters

## Release Status

**✅ READY FOR RELEASE**

All critical issues have been resolved:
- SP-API integration completed
- Supply chain integration verified
- Configuration management centralized
- Documentation updated
- Tests passing

## Contributing

1. Ensure all tests pass: `python test_critical_fixes.py`
2. Follow the existing code style and documentation patterns
3. Update configuration in [`config.py`](fba_bench/config.py) rather than hardcoding values
4. Add appropriate tests for new features

## License

[Add appropriate license information]

## Support

For technical issues or questions about the simulation framework, please refer to the [Master Blueprint](fba_bench_master_blueprint_v_2.md) for detailed technical specifications.