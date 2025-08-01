# Tutorial 3: Running Large-Scale Distributed Simulations

This tutorial covers setting up and running FBA-Bench simulations in a distributed and scalable manner, utilizing features like LLM batching and fast-forward simulation.

## Distributed Simulation Setup

FBA-Bench leverages a distributed architecture to run multiple agents concurrently across different processes or machines.

### Using Docker Compose for Local Distribution

For a local distributed setup, you can use the provided Docker Compose file:

```bash
cd infrastructure/deployment
docker-compose up --scale agent_runner=5 # Runs 5 agent runner instances
```

This command will start the core FBA-Bench services and 5 agent runner instances, allowing for parallel simulation execution.

## LLM Request Batching

To optimize cost and performance, FBA-Bench can batch multiple LLM requests into a single API call when supported by the LLM provider.

```python
# tutorial_llm_batching.py
from fba_bench.infrastructure.llm_batcher import LLMBatcher
from fba_bench.llm_interface.openrouter_client import OpenRouterClient # Example client

# Initialize LLM client with batching enabled
# This assumes your LLM client supports a 'batch_size' parameter or similar
llm_client = OpenRouterClient(api_key="YOUR_API_KEY", batch_size=10)

# The LLMBatcher will intercept and queue requests, then send them in batches.
# This is usually integrated automatically when distributed simulation is active.
# For manual testing:
# batcher = LLMBatcher(llm_client, batch_interval_seconds=0.1)
# batcher.add_request({"prompt": "Hello"}, callback=lambda r: print(r))
# batcher.add_request({"prompt": "World"}, callback=lambda r: print(r))
# batcher.process_batch()
```

## Fast-Forward Simulation

For long-duration scenarios (e.g., year-long simulations), FBA-Bench provides a fast-forward engine to skip less critical simulation steps or accelerate phases.

```python
# tutorial_fast_forward.py
from fba_bench.infrastructure.fast_forward_engine import FastForwardEngine
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.scenarios.tier_1_moderate import tier_1_scenario
from fba_bench.events import SimulationEvent

class AcceleratedScenarioEngine(ScenarioEngine):
    def __init__(self, scenario_config):
        super().__init__(scenario_config)
        self.fast_forward_engine = FastForwardEngine()

    def run_simulation(self, agent):
        # Example of applying a fast-forward rule
        # Skip initial setup phase and jump to a critical period
        self.fast_forward_engine.add_rule(
            event_type=SimulationEvent.DAY_START,
            condition=lambda day: day <= 30, # Skip first 30 days
            action=lambda: self.advance_simulation_time(30) # Advance by 30 days silently
        )
        return super().run_simulation(agent)

# Usage
# accelerated_engine = AcceleratedScenarioEngine(tier_1_scenario)
# agent = YourAgent(...)
# results = accelerated_engine.run_simulation(agent)

print("Fast-forward simulation is typically configured via scenario or infrastructure settings.")
print("Refer to docs/infrastructure/distributed-simulation.md for full details.")
```

For comprehensive details on distributed simulation setup, deployment options (Docker, Kubernetes), and performance optimization, consult the [`docs/infrastructure/`](docs/infrastructure/) documentation.