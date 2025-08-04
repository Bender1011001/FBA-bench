# Functional Benchmarking Application (FBA)

A comprehensive benchmarking framework for evaluating AI agents across various scenarios and metrics.

## Overview

The Functional Benchmarking Application (FBA) provides a robust platform for creating, running, and analyzing benchmarks for AI agents. It supports multiple agent frameworks, customizable scenarios, and detailed metrics collection and visualization.

## Features

- **Agent Integration**: Support for multiple agent frameworks with a unified interface
- **Scenario Builder**: Create and customize benchmark scenarios with a user-friendly interface
- **Metrics Collection**: Comprehensive metrics collection including cognitive, business, and technical metrics
- **Real-time Monitoring**: Live monitoring of benchmark execution with WebSocket support
- **Visualization**: Interactive charts and graphs for analyzing benchmark results
- **Report Generation**: Generate detailed reports in various formats
- **Configuration Management**: Save, load, and share benchmark configurations
- **Extensible Architecture**: Plugin-based system for adding new agent frameworks and metrics

## Architecture

The FBA consists of several key components:

### Backend Components

1. **Core Engine** (`benchmarking/core/`)
   - `engine.py`: Main benchmark execution engine
   - `models.py`: Data models for benchmarks, results, and configurations
   - `status.py`: Benchmark status enumeration

2. **Configuration Management** (`benchmarking/config/`)
   - `manager.py`: Configuration validation and management
   - `schema.py`: JSON schema definitions for configurations

3. **Integration Framework** (`benchmarking/integration/`)
   - `manager.py`: Integration manager for agent frameworks
   - `agent_adapter.py`: Adapter interface for agent frameworks
   - `metrics_adapter.py`: Adapter interface for metrics collection

4. **Metrics System** (`benchmarking/metrics/`)
   - `registry.py`: Registry for available metrics
   - `base.py`: Base classes for implementing metrics
   - `cognitive/`: Cognitive metrics implementations
   - `business/`: Business metrics implementations
   - `technical/`: Technical metrics implementations

5. **Scenarios** (`benchmarking/scenarios/`)
   - `registry.py`: Registry for available scenarios
   - `base.py`: Base classes for implementing scenarios
   - `test_scenarios/`: Test scenario implementations

6. **API Server** (`api/`)
   - `main.py`: FastAPI application entry point
   - `routes/`: API endpoints for various resources
   - `middleware/`: Custom middleware for authentication, logging, etc.

### Frontend Components

1. **React Application** (`frontend/src/`)
   - `components/`: React components for the UI
   - `hooks/`: Custom React hooks
   - `services/`: API service and other utilities
   - `utils/`: Utility functions and helpers

2. **Key UI Components**
   - `Dashboard/`: Main dashboard with overview of benchmarks
   - `BenchmarkRunner/`: Interface for running benchmarks
   - `ScenarioBuilder/`: Interface for creating scenarios
   - `ConfigurationEditor/`: Interface for editing configurations
   - `MetricsVisualization/`: Charts and graphs for results
   - `ReportGenerator/`: Interface for generating reports

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/fba.git
   cd fba
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the database:
   ```bash
   python -m api.database.setup
   ```

5. Run the API server:
   ```bash
   python -m api.main
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open your browser and navigate to `http://localhost:3000`

## Usage

### Creating a Benchmark

1. Navigate to the "Scenario Builder" in the web interface.
2. Select a configuration template or start from scratch.
3. Configure the benchmark settings:
   - General information (name, description)
   - Environment settings (deterministic, parallel execution)
   - Scenarios (type, configuration)
   - Agents (framework, configuration)
   - Metrics (categories, custom metrics)
   - Execution settings (runs, timeout, retries)
   - Output settings (format, path)
   - Validation settings (significance, confidence level)
4. Save the configuration.

### Running a Benchmark

1. Navigate to the "Benchmark Runner" in the web interface.
2. Select a saved configuration.
3. Click "Run Benchmark" to start the execution.
4. Monitor the progress in real-time.
5. View the results once the benchmark completes.

### Analyzing Results

1. Navigate to the "Metrics Visualization" in the web interface.
2. Select the benchmark results to analyze.
3. Choose the visualization type (overview, agents, trends, comparison).
4. Select the chart type (bar, line, radar, scatter).
5. Export the visualization or data as needed.

### Generating Reports

1. Navigate to the "Report Generator" in the web interface.
2. Select the benchmark results to include in the report.
3. Choose a report template.
4. Customize the report settings.
5. Generate and download the report.

## Development

### Adding New Agent Frameworks

1. Create a new agent adapter by implementing the `AgentAdapter` interface:
   ```python
   from benchmarking.integration.agent_adapter import AgentAdapter
   
   class MyAgentAdapter(AgentAdapter):
       def __init__(self, config):
           super().__init__(config)
           # Initialize your agent framework
       
       async def execute(self, scenario_config):
           # Execute the scenario with your agent
           pass
   ```

2. Register the adapter in the integration manager:
   ```python
   from benchmarking.integration.manager import IntegrationManager
   
   manager = IntegrationManager()
   manager.register_agent_framework('my-framework', MyAgentAdapter)
   ```

3. Add the framework to the frontend configuration in `frontend/src/config/agentFrameworks.js`.

### Adding New Metrics

1. Create a new metric by implementing the `BaseMetric` class:
   ```python
   from benchmarking.metrics.base import BaseMetric
   
   class MyMetric(BaseMetric):
       def __init__(self, config):
           super().__init__(config)
           # Initialize your metric
       
       async def calculate(self, events, agent_data, scenario_data):
           # Calculate the metric value
           pass
   ```

2. Register the metric in the metrics registry:
   ```python
   from benchmarking.metrics.registry import metrics_registry
   
   metrics_registry.register('my-metric', MyMetric, 'my-category')
   ```

3. Add the metric to the frontend configuration in `frontend/src/config/metrics.js`.

### Adding New Scenarios

1. Create a new scenario by implementing the `BaseScenario` class:
   ```python
   from benchmarking.scenarios.base import BaseScenario
   
   class MyScenario(BaseScenario):
       def __init__(self, config):
           super().__init__(config)
           # Initialize your scenario
       
       async def run(self, agent):
           # Run the scenario with the agent
           pass
   ```

2. Register the scenario in the scenarios registry:
   ```python
   from benchmarking.scenarios.registry import scenario_registry
   
   scenario_registry.register('my-scenario', MyScenario)
   ```

3. Add the scenario to the frontend configuration in `frontend/src/config/scenarioTypes.js`.

## Testing

### Backend Tests

Run the backend tests:
```bash
pytest tests/
```

### Frontend Tests

Run the frontend tests:
```bash
cd frontend
npm test
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Add tests for your changes.
5. Ensure all tests pass.
6. Submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on the GitHub repository or contact the development team.