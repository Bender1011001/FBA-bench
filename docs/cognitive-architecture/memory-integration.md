
## Core Components

-   **Dual Memory Manager**: Manages both short-term (episodic) and long-term (semantic) memories, allowing for flexible knowledge retention and retrieval.
-   **Memory Validator**: Actively checks the consistency and validity of information stored in the agent's memory, ensuring data integrity.
-   **Memory Enforcer**: Applies rules and constraints to memory updates to prevent contradictions or irrational knowledge.

## How it Works

The memory system components (found primarily in `memory_experiments/` such as [`dual_memory_manager.py`](memory_experiments/dual_memory_manager.py), [`memory_validator.py`](memory_experiments/memory_validator.py), and [`memory_enforcer.py`](memory_experiments/memory_enforcer.py)) work together to:

1.  **Ingest Information**: Agent observations, actions, simulation events, and reflection insights are systematically added to the memory.
2.  **Validation**: The `MemoryValidator` continuously (or at configurable intervals) assesses the coherence and correctness of memory entries, flagging or resolving inconsistencies. This can involve cross-referencing information, checking numerical sanity, or verifying logical relationships.
3.  **Consistency Checking**: When new information is added, it's checked against existing memory to avoid contradictions, ensuring the agent's internal model remains consistent.
4.  **Retrieval**: Agents can query their memory for relevant information to inform decision-making or planning. The system optimizes retrieval for both speed and relevance.
5.  **Forgetting/Pruning (Optional)**: Mechanisms can be configured to selectively prune less relevant or outdated information to maintain memory efficiency and prevent overload.

## Memory Validation and Consistency Checking

The [`memory_validator.py`](memory_experiments/memory_validator.py) plays a key role in ensuring robust memory. It performs checks such as:
- **Numerical Sanity Checks**: Are financial figures within reasonable bounds?
- **Temporal Consistency**: Do event timestamps make sense chronologically?
- **Fact-Checking**: Does new information contradict established facts or previously verified data?
- **Relational Integrity**: Are relationships between concepts (e.g., product, price, demand) consistent?

When inconsistencies are detected, the `MemoryValidator` can be configured to:
-   Log a warning.
-   Attempt to resolve the inconsistency using predefined rules or by consulting an LLM.
-   Flag the inconsistent data for manual review.

## Configuration

Memory integration is configured within the agent's cognitive settings, typically via `cognitive_config.yaml`.

```yaml
# Example cognitive_config.yaml snippet for memory
memory:
  enabled: true
  validation: true # Enable active memory validation
  consistency_checks: true # Enable consistency checks on memory updates
  forgetting_strategy: "least_recent_access" # Example: "least_recent_access", "fixed_size_window"
  max_memory_size: 1000 # Example: Max entries in short-term memory
  long_term_storage_backend: "json_file" # Example: "json_file", "database"
```

## Example Usage

While the memory system operates largely in the background as part of the agent's core processes, here's how you might interact with a simplified memory manager or see validation in action.

```python
from fba_bench.memory_experiments.dual_memory_manager import DualMemoryManager
from fba_bench.memory_experiments.memory_validator import MemoryValidator
from fba_bench.agents.cognitive_config import CognitiveConfig

memory_config = CognitiveConfig(memory={"enabled": True, "validation": True, "consistency_checks": True}).memory_config

memory_manager = DualMemoryManager(agent_name="MyAgent", config=memory_config)
memory_validator = MemoryValidator(memory_manager) # Validator needs access to the manager's data

# Add some initial, consistent data
memory_manager.add_to_short_term("ProductA current price: $10.00")
memory_manager.add_to_short_term("Sales for ProductA last week: 50 units")
print("Initial memory state is valid:", memory_validator.validate_all_memory())

# Add inconsistent data
memory_manager.add_to_short_term("ProductA current price: $-5.00") # Invalid price
print("Memory state after adding inconsistent data is valid:", memory_validator.validate_all_memory())

# Adding new information that might conflict with existing (if conflict resolution logic implemented)
# memory_manager.add_to_short_term("ProductA current price: $12.00 after discount")
# The validation process would then detect the conflict or integrate responsibly.

print("Current Short-Term Memory:", memory_manager.get_short_term_memory())
```

For more on agent learning and how memory supports it, see [`Episodic Learning`](../learning-system/episodic-learning.md).