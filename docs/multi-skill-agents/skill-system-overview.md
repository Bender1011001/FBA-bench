# Multi-Skill Agent System Overview and Coordination

FBA-Bench's Multi-Skill Agent system introduces the capability for agents to possess and coordinate multiple specialized "skills" or domains of expertise. Instead of a single monolithic decision-making process, agents can delegate tasks to individual skill modules (e.g., marketing, finance, supply chain) and then synthesize their outputs.

## Architecture

The multi-skill agent architecture is designed around the following core components:

-   **Multi-Domain Controller (`agents/multi_domain_controller.py`)**: The central orchestrator for a multi-skill agent. It receives the overall simulation state, determines which skills are relevant, invokes them, and coordinates their outputs into a coherent set of actions.
-   **Skill Modules (`agents/skill_modules/`)**: Individual, specialized units of expertise. Each skill module (e.g., [`supply_manager.py`](agents/skill_modules/supply_manager.py), [`marketing_manager.py`](agents/skill_modules/marketing_manager.py)) is designed to handle a specific area of business operations. They encapsulate their own logic, tools, and potentially even their own sub-agents. All skill modules inherit from [`base_skill.py`](agents/skill_modules/base_skill.py).
-   **Skill Coordinator (`agents/skill_coordinator.py`)**: A component responsible for resolving conflicts or integrating overlapping recommendations from different skill modules. This ensures that the agent's final actions are consistent and optimal across domains.
-   **Event-Driven Triggers**: Skills can be activated or informed by specific events occurring in the simulation (e.g., a "demand spike" event might trigger the Marketing Manager skill).

## Design Principles

### Specialization
Each skill module is focused on a narrow but deep area of expertise, allowing for highly optimized and context-specific decision-making within that domain.

### Modularity and Extensibility
New skill modules can be easily added or existing ones modified without impacting the core multi-domain controller. This allows for rapid experimentation and community contributions.

### Hierarchical Decision-Making
The Multi-Domain Controller acts as a higher-level decision-maker, managing the flow of control and information between specialized skills, akin to a CEO coordinating department heads.

### Conflict Resolution
Mechanisms are in place to identify and resolve potential conflicts or redundancies between the actions proposed by different skills, ensuring cohesive agent behavior.

### Observability
Detailed logs and traces are captured for each skill invocation, its internal reasoning, and its proposed actions, providing deep insights into the multi-skill decision process.

## How Multi-Skill Agents Operate

1.  **Observe**: The Multi-Domain Controller receives the current global simulation state and relevant events.
2.  **Route**: Based on the state and configured skill routing rules, it identifies which enabled skill modules are most relevant to the current situation.
3.  **Execute Skills**: Relevant skill modules are invoked with their specific context. Each skill module executes its logic, potentially calling LLMs, using internal tools, or performing calculations.
4.  **Propose Actions**: Each invoked skill module proposes a set of actions relevant to its domain (e.g., "change product price", "order 500 units of inventory").
5.  **Coordinate and Resolve**: The `SkillCoordinator` collects all proposed actions. If there are conflicting or overlapping actions (e.g., Marketing wants to lower price to gain market share, Finance wants to raise price to increase profit margin), the coordinator applies a defined strategy (e.g., "prioritize financial impact") to produce a single, coherent set of final actions.
6.  **Act**: The Multi-Domain Controller executes the coordinated actions in the simulation.

For detailed documentation on individual skill modules, custom skill creation, and coordination mechanisms, refer to:
- [`Skill Modules`](skill-modules.md)
- [`Skill Coordination`](skill-coordination.md)
- [`Custom Skills`](custom-skills.md)
- [`Performance Optimization`](performance-optimization.md)
- [`Multi-Skill Agent Configuration Guide`](../configuration/skill-config.md)