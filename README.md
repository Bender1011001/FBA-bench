# FBA-Bench v3 Research Toolkit

FBA-Bench v3 is a powerful, deterministic, and scalable research toolkit designed for simulating complex financial and business scenarios, particularly in e-commerce environments (e.g., Fulfillment by Amazon - FBA). It provides a robust framework for developing, benchmarking, and analyzing the performance of AI agents and business strategies under various market conditions and adversarial events.

This toolkit emphasizes reproducibility, financial integrity, and a modular, event-driven architecture to support advanced research and rigorous analysis.

## 🚀 Features and Capabilities

*   **Deterministic Simulation Engine**: Ensures reproducible results across multiple runs and environments, critical for scientific benchmarking.
*   **Event-Driven Architecture**: Utilizes a flexible `EventBus` (`event_bus.py`) for decoupled communication, supporting both in-memory and distributed (Redis-backed) event processing.
*   **Modular Agent System**:
    *   **Skills Framework**: Agents are composed of specialized `BaseSkill` modules (`agents/skill_modules/base_skill.py`) that handle domain-specific responsibilities (e.g., `FinancialAnalystSkill`, `SupplyManagerSketch`).
    *   **Skill Coordination**: `agents/skill_coordinator.py` manages event dispatch to skills, prioritizes actions, resolves conflicts, and tracks skill performance.
    *   **Strategic Control**: `agents/multi_domain_controller.py` provides a "CEO-level" arbitration layer, aligning agent actions with high-level business objectives and managing resource allocation.
*   **Plug-and-Play LLM Integration**: Seamlessly integrate with various Large Language Models (LLMs) (e.g., OpenAI, OpenRouter) via a standardized `llm_interface/` for advanced reasoning and decision-making within agents.
*   **Comprehensive Scenario Management**:
    *   Define complex business scenarios and market dynamics using flexible YAML configurations (`scenarios/scenario_framework.py`).
    *   Inject external events (e.g., supply disruptions, market shifts, adversarial attacks) to test agent resilience.
*   **Rigorous Financial Auditing**: `financial_audit.py` strictly validates all financial transactions against accounting principles, ensuring data integrity and halting simulations on critical violations.
*   **Benchmarking & Evaluation**: Robust framework ([`benchmarking/`](benchmarking/)) for defining evaluation metrics, running experiments, and analyzing agent performance.
*   **Scalable Architecture**: Supports distributed simulation execution leveraging a Redis-backed `DistributedEventBus` (`infrastructure/distributed_event_bus.py`) for multi-process/multi-node deployments.
*   **RESTful API & Real-time Frontend**:
    *   `fba_bench_api/` offers a FastAPI backend for simulation control, configuration management, and real-time data streaming (via WebSockets).
    *   `frontend/` provides a React/TypeScript-based user interface for intuitive interaction, visualization, and monitoring.
*   **Toolbox API Abstraction**: `services/toolbox_api_service.py` provides a simplified interface for agents to interact with the simulation environment, decoupling agent logic from low-level event mechanisms.

## 🏛️ Architecture Overview

FBA-Bench v3 employs a microservices-inspired, event-driven architecture designed for modularity, scalability, and maintainability.

```mermaid
graph TD
    subgraph Frontend [User Interface (React/TypeScript)]
        UI[Web Application] -- Real-time Data / Controls --> FastAPI_API
    end

    subgraph Backend [FBA-Bench API (FastAPI)]
        FastAPI_API[RESTful API & WebSockets] -- Config/Control --> SimulationManager[Simulation Manager]
        FastAPI_API -- Data Query --> Persistence[Database / Persistence Layer]
        SimulationManager -- Orchestration --> EventBus
    end

    subgraph Simulation Core [Python Services & Agents]
        Orchestrator[SimulationOrchestrator] -- Tick Events --> EventBus
        EventBus --> FinancialAuditService[Financial Audit Service]
        EventBus --> AgentSystem[Agent System]
        EventBus --> ToolboxAPIService[Toolbox API Service]
        EventBus --> OtherServices[Other Simulation Services]
        
        subgraph Agent System
            Skill1[Skill Module 1 (e.g., Financial Analyst)]
            Skill2[Skill Module 2 (e.g., Marketing Manager)]
            SkillN[...]
            SkillCoordinator[Skill Coordinator] -- Arbitrates & Prioritizes --> MultiDomainController[Multi-Domain Controller]
            MultiDomainController -- Approved Actions --> ToolboxAPIService
            SkillCoordinator --> Skill1 & Skill2 & SkillN
        end

        ToolboxAPIService -- Publish Commands --> EventBus
        ToolboxAPIService -- Observe Data --> WorldState[World State (via snapshot/updates)]
        WorldState -- Updates --> EventBus
    end

    subgraph Experiment Execution [CLI]
        ExperimentCLI[experiment_cli.py] -- Run Scenarios --> ScenarioEngine[Scenario Engine]
        ScenarioEngine -- Initializes --> Orchestrator & AgentSystem & OtherServices
        ScenarioEngine -- Results --> ExperimentCLI
    end

    subgraph Distributed Infrastructure
        EventBus <--- DistributedEventBus[Distributed Event Bus (Redis, Kafka)]
        DistributedEventBus -- Worker Reg. & Load Bal. --> Workers[Multiple Simulation Workers]
    end

    subgraph LLM Integration
        AgentSystem -- LLM Calls --> LLM_Interface[LLM Interface (OpenAI/OpenRouter)]
        LLM_Interface -- API Requests --> ExternalLLM[External LLM Providers]
    end

    style Frontend fill:#f9f,stroke:#333,stroke-width:2px
    style Backend fill:#bbf,stroke:#333,stroke-width:2px
    style SimulationCore fill:#bfb,stroke:#333,stroke-width:2px
    style AgentSystem fill:#dfd,stroke:#333,stroke-width:2px
    style ExperimentExecution fill:#ffb,stroke:#333,stroke-width:2px
    style DistributedInfrastructure fill:#ccf,stroke:#333,stroke-width:2px
    style LLMIntegration fill:#fbc,stroke:#333,stroke-width:2px
```

**Key Interactions and Data Flow:**

1.  **Initialization**: The `experiment_cli.py` or the FastAPI application (`fba_bench_api/main.py`) initiates a simulation run, which involves the `ScenarioEngine` to load scenario configurations.
2.  **Simulation Loop**: `SimulationOrchestrator` acts as the clock, continuously emitting `TickEvent`s onto the `EventBus`.
3.  **Event Propagation**: The `EventBus` (potentially distributed via `RedisBroker` in `infrastructure/distributed_event_bus.py`) dispatches `TickEvent`s and other domain-specific `BaseEvent`s (e.g., `SaleOccurred`, `CompetitorPricesUpdated`) to all subscribed components.
4.  **Agent Decision Cycle**:
    *   `BaseSkill` modules within the Agent System receive relevant events.
    *   Skills process these events, analyze the `SkillContext` (current state, market data, etc.), possibly query LLMs via `llm_interface/`, and generate `SkillAction` proposals.
    *   The `SkillCoordinator` collects these `SkillAction`s, prioritizes them, and resolves potential conflicts.
    *   For sophisticated agents, the `MultiDomainController` applies high-level strategic alignment checks, business rules, and allocates resources before approving final actions.
    *   Approved actions are then translated into commands (e.g., `SetPriceCommand`) and sent to the `ToolboxAPIService` for execution.
5.  **World Interaction**: The `ToolboxAPIService` acts as an intermediary, receiving update commands from agents (e.g., price changes) and publishing them as new events to the `EventBus`. It also maintains a cached view of the `WorldState` (derived from `WorldStateSnapshotEvent`s or specific updates from the EventBus) that agents can "observe."
6.  **State Management & Auditing**: Core simulation state is updated by various internal services (implied by events) and the `FinancialAuditService` rigorously checks all financial transactions for integrity.
7.  **Observability & UI**: The FastAPI backend collects data and exposes it via API endpoints, including real-time WebSockets, which the React/TypeScript frontend consumes to provide live dashboards and user controls.

## 🛠️ Installation and Setup

### Prerequisites

*   **Python 3.9+**: For the backend services and simulation core.
*   **Node.js & npm (or Yarn)**: For the frontend application.
*   **Docker (Optional)**: For running Redis or other containerized services in a distributed setup.
*   **Redis (Optional)**: Required for Distributed Event Bus functionality.

### Backend Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/fba-bench.git
    cd fba-bench
    ```

2.  **Create and activate a Python virtual environment**:
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies with Poetry (recommended)**:
    ```bash
    pip install -U pip setuptools wheel
    pip install poetry
    poetry install --with dev
    ```

4.  **Database Initialization (if applicable)**:
    If using a persistent database (e.g., PostgreSQL configured in `fba_bench_api/core/database.py`), ensure it's running and apply any necessary migrations. For a default SQLite setup, no explicit initialization might be needed beyond running the application.

5.  **Environment Variables**:
    Create a `.env` file in the root directory based on `.env.example`.
    ```ini
    # .env
    OPENROUTER_API_KEY="your_openrouter_api_key_here"
    OPENAI_API_KEY="your_openai_api_key_here"
    FBA_BENCH_REDIS_URL="redis://localhost:6379/0" # Only needed for distributed mode
    FBA_BENCH_DB_URL="sqlite:///./fba_bench.db" # Default SQLite, can be postgresql/mysql
    ```

### Frontend Setup

1.  **Navigate to the frontend directory**:
    ```bash
    cd frontend
    ```

2.  **Install Node.js dependencies**:
    ```bash
    npm install
    # or yarn install
    ```

3.  **Environment Variables (Frontend)**:
    Create a `.env` file in the `frontend/` directory based on `frontend/.env.example`.
    ```ini
    # frontend/.env
    VITE_API_BASE_URL=http://localhost:8000 # Adjust if your backend runs on a different port
    VITE_WEBSOCKET_URL=ws://localhost:8000/ws # Adjust if your backend runs on a different host/port
    ```

## 🏃‍♀️ Usage

### Running the Backend API

From the root directory of the project, with your Python virtual environment activated:

```bash
poetry run uvicorn fba_bench_api.main:app --reload --port 8000
```
The API will be accessible at `http://localhost:8000`.

### Running the Frontend Application

From the `frontend/` directory:

```bash
npm run dev
# or yarn dev
```
The frontend application will typically open in your browser at `http://localhost:5173` (or similar).

### Running CLI Experiments

From the root directory of the project, with your Python virtual environment activated:

1.  **Execute an experiment sweep**:
    ```bash
    python experiment_cli.py run config/templates/benchmark_basic.yaml --parallel 4
    ```
    This command runs the experiment defined in `benchmark_basic.yaml` using 4 parallel processes. Results will be saved in a timestamped directory under `results/`.

2.  **Analyze experiment results**:
    ```bash
    python experiment_cli.py analyze results/your_experiment_name_timestamp/
    ```
    Replace `results/your_experiment_name_timestamp/` with the actual path to your experiment results directory.

### Running Tests

The project includes unit, integration, and accessibility tests for both backend and frontend components.

#### Backend Tests

From the root directory:

```bash
pytest
```
To run specific test modules or apply filters, refer to the `pytest` documentation.

#### Frontend Tests

From the `frontend/` directory:

```bash
npm test
# or yarn test
```
This will execute tests using Vitest/React Testing Library, as configured in `frontend/package.json`.

## ⚙️ Configuration

### Global Configuration

*   **`.env` files**: Manage API keys, database URLs, and other sensitive or environment-specific settings. Follow `.env.example` as a template.
*   **`config.yaml` / `configs/`**: Global application configurations, potentially including default parameters for simulations, agent behaviors, or infrastructure settings.

### Scenario Configuration

*   **`scenarios/`**: Defines the building blocks for simulation scenarios.
*   **`config/templates/` (e.g., `benchmark_basic.yaml`, `benchmark_advanced.yaml`)**: Examples of complete scenario configurations used by `experiment_cli.py`. These files define `scenario_name`, `difficulty_tier`, `expected_duration`, `success_criteria`, `market_conditions`, `external_events`, and `agent_constraints`.

### Agent Configuration

*   **`agents/`**: Contains various agent implementations. Agent-specific configurations are often defined within their respective modules or loaded from external YAML files. Parameters for LLM clients (models, temperatures) are configured in `llm_interface/` clients.

## 🤝 Contribution Guidelines

We welcome contributions to FBA-Bench! Please refer to `CONTRIBUTING.md` (to be created) for detailed guidelines on how to set up your development environment, propose changes, and submit pull requests.

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file (to be created or identified if existing) for details.

---
**FBA-Bench v3: Research Toolkit for Financial & Business Agent Benchmarking**