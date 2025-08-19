## Frontend UI Overview

- Layout and shell: [`tsx.function Layout()`](frontend/src/components/layout/Layout.tsx:1)
- Global error boundary: [`tsx.function ErrorBoundary()`](frontend/src/components/error/ErrorBoundary.tsx:1)
- Centralized routes: [`tsx.function AppRoutes()`](frontend/src/routes/index.tsx:1)

Run commands:
- Install deps: `cd frontend && npm i`
- Dev server: `npm run dev` (opens http://localhost:5173)
- Tests: `npm test`

Notes:
- Responsive, accessible app shell with collapsible sidebar, keyboard shortcuts:
  - 's' toggles sidebar, '/' focuses header search
- Toasts available via Toast context/helpers
- Pages are lazy-loaded with Suspense and have skeletons/spinners as fallbacks

## Scenarios (New)

A set of complex, deterministic, and runner-agnostic scenarios are available and discoverable by key via the scenario registry. See the dedicated reference for usage, configuration, and runner contracts:

- [docs/api-reference/scenarios.md](docs/api-reference/scenarios.md:1)

Built-in keys:
- complex_marketplace
- research_summarization
- multiturn_tool_use

Registered via [`python.class ScenarioRegistry`](benchmarking/scenarios/registry.py:1). Each scenario exposes:
- [`python.class ComplexMarketplaceConfig(BaseModel)`](benchmarking/scenarios/complex_marketplace.py:1)
- [`python.def generate_input(seed: int|None, params: dict|None) -> dict`](benchmarking/scenarios/complex_marketplace.py:1)
- [`python.async def run(input_payload: dict, runner_callable: Callable[[dict], Awaitable[dict]], timeout_seconds: int|None=None) -> dict`](benchmarking/scenarios/complex_marketplace.py:1)
- [`python.def postprocess(raw_output: dict) -> dict`](benchmarking/scenarios/complex_marketplace.py:1)

- [`python.class ResearchSummarizationConfig(BaseModel)`](benchmarking/scenarios/research_summarization.py:1)
- [`python.def generate_input(seed: int|None, params: dict|None) -> dict`](benchmarking/scenarios/research_summarization.py:1)
- [`python.async def run(input_payload: dict, runner_callable: Callable[[dict], Awaitable[dict]], timeout_seconds: int|None=None) -> dict`](benchmarking/scenarios/research_summarization.py:1)

- [`python.class MultiTurnToolUseConfig(BaseModel)`](benchmarking/scenarios/multiturn_tool_use.py:1)
- [`python.def generate_input(seed: int|None, params: dict|None) -> dict`](benchmarking/scenarios/multiturn_tool_use.py:1)
- [`python.async def run(input_payload: dict, runner_callable: Callable[[dict], Awaitable[dict]], timeout_seconds: int|None=None) -> dict`](benchmarking/scenarios/multiturn_tool_use.py:1)
# FBA-Bench Backend API (Agents, Experiments, Config, Simulation)

This section documents the newly implemented, production-ready endpoints. All routes include explicit response models in OpenAPI. Identifiers are UUID4 strings.

Base prefix: /api/v1

Agents
- List agents
  - GET /agents
  - 200 -> array of Agent
- Create agent
  - POST /agents
  - Body example:
    {
      "name": "My Agent",
      "framework": "baseline",
      "config": {"temp": 0.2}
    }
  - 201 -> Agent
- Get agent
  - GET /agents/{agent_id}
  - 200 -> Agent, 404 if not found
- Update agent
  - PATCH /agents/{agent_id}
  - Body example:
    {"name": "Updated Agent", "config": {"temp": 0.3}}
  - 200 -> Agent, 404 if not found, 400/422 validation
- Delete agent
  - DELETE /agents/{agent_id}
  - 204 on success, 404 if not found

Agent model
- id: string (uuid)
- name: string (non-empty)
- framework: enum ["baseline","langchain","crewai","custom"]
- config: object
- created_at: string (date-time)
- updated_at: string (date-time)

Experiments
- List experiments
  - GET /experiments
  - 200 -> array of Experiment
- Create experiment (starts as draft)
  - POST /experiments
  - Body example:
    {
      "name": "Exp1",
      "description": "Benchmark agent on scenario-abc",
      "agent_id": "7f3a3a2f-6f2b-4bfb-8b9b-2b7b0f5f8e12",
      "scenario_id": "scenario-abc",
      "params": {"k": 1, "seed": 42}
    }
  - 201 -> Experiment
- Get experiment
  - GET /experiments/{experiment_id}
  - 200 -> Experiment, 404 if not found
- Update experiment or transition status
  - PATCH /experiments/{experiment_id}
  - Body example:
    {"status": "running"}
  - 200 -> Experiment
  - 400 if invalid transition (allowed: draft->running; running->completed/failed)
  - 404 if not found
- Delete experiment
  - DELETE /experiments/{experiment_id}
  - 204 on success, 404 if not found

Experiment model
- id: string (uuid)
- name: string (non-empty)
- description: string|null
- agent_id: string (uuid)
- scenario_id: string|null
- params: object
- status: enum ["draft","running","completed","failed"]
- created_at: string (date-time)
- updated_at: string (date-time)

Runtime Config
- Get effective config (env + overrides)
  - GET /config
  - 200 -> ConfigResponse
- Patch runtime overrides (in-memory)
  - PATCH /config
  - Body example:
    {"enable_observability": true, "profile": "development", "telemetry_endpoint": "http://localhost:4318"}
  - 200 -> merged ConfigResponse

Config model
- enable_observability: boolean|null (env: ENABLE_OBSERVABILITY)
- profile: string|null (env: PROFILE)
- telemetry_endpoint: string|null (env: OTEL_EXPORTER_OTLP_ENDPOINT)

Simulation
- Create simulation record (pending)
  - POST /simulation
  - Body example:
    { "experiment_id": null, "metadata": {"note": "ad-hoc run"} }
  - 201 -> Simulation (includes websocket_topic simulation-progress:{id})
- Start simulation
  - POST /simulation/{id}/start
  - 200 -> Simulation with status=running
  - 400 if not in pending, 404 if not found
- Stop simulation
  - POST /simulation/{id}/stop
  - 200 -> Simulation with status=stopped
  - 400 if not running, 404 if not found
- Get simulation status
  - GET /simulation/{id}
  - 200 -> Simulation, 404 if not found

Simulation model
- id: string (uuid)
- experiment_id: string|null
- status: enum ["pending","running","stopped","completed","failed"]
- websocket_topic: string (e.g., "simulation-progress:{id}")
- metadata: object|null
- created_at: string (date-time)
- updated_at: string (date-time)

Error handling
- 400: Validation or invalid status transition
- 404: Not found
- 409: Conflict (duplicate identifiers)
- 201: Created
- 200: Success
- 204: No Content on delete

Note: These endpoints are backed by in-memory repositories with clear interfaces to support future SQLAlchemy persistence. Live update topic is included on simulation creation and control operations for frontend WebSocket subscriptions.
# API Documentation

This document provides detailed information about the REST API endpoints for the Functional Benchmarking Application (FBA).

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Response Format

All API responses follow a consistent JSON format:

```json
{
  "success": true,
  "data": {},
  "message": "Success message",
  "errors": []
}
```

## Error Handling

When an error occurs, the API returns a response with the following structure:

```json
{
  "success": false,
  "data": null,
  "message": "Error message",
  "errors": [
    {
      "field": "field_name",
      "message": "Error description"
    }
  ]
}
```

## Endpoints

### Authentication

#### Login

Authenticate a user and return a JWT token.

**Endpoint:** `POST /auth/login`

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "token": "jwt_token",
    "user": {
      "id": "user_id",
      "username": "username",
      "email": "email",
      "role": "user_role"
    }
  },
  "message": "Login successful",
  "errors": []
}
```

#### Logout

Invalidate the current JWT token.

**Endpoint:** `POST /auth/logout`

**Request Headers:**
```
Authorization: Bearer <your-jwt-token>
```

**Response:**
```json
{
  "success": true,
  "data": null,
  "message": "Logout successful",
  "errors": []
}
```

#### Refresh Token

Refresh an expired JWT token.

**Endpoint:** `POST /auth/refresh`

**Request Headers:**
```
Authorization: Bearer <your-jwt-token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "token": "new_jwt_token"
  },
  "message": "Token refreshed successfully",
  "errors": []
}
```

### Benchmark Configurations

#### Get All Configurations

Retrieve a list of all benchmark configurations.

**Endpoint:** `GET /benchmarking/configurations`

**Query Parameters:**
- `page` (integer, optional): Page number for pagination (default: 1)
- `limit` (integer, optional): Number of items per page (default: 10)
- `sort` (string, optional): Field to sort by (default: created_at)
- `order` (string, optional): Sort order (asc or desc, default: desc)
- `search` (string, optional): Search term for filtering results

**Response:**
```json
{
  "success": true,
  "data": {
    "configurations": [
      {
        "id": "config_id",
        "benchmark_id": "benchmark_id",
        "name": "Configuration Name",
        "description": "Configuration Description",
        "version": "1.0.0",
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
        "created_by": "user_id"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 100,
      "pages": 10
    }
  },
  "message": "Configurations retrieved successfully",
  "errors": []
}
```

#### Get Configuration by ID

Retrieve a specific benchmark configuration by ID.

**Endpoint:** `GET /benchmarking/configurations/{id}`

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "config_id",
    "benchmark_id": "benchmark_id",
    "name": "Configuration Name",
    "description": "Configuration Description",
    "version": "1.0.0",
    "environment": {
      "deterministic": true,
      "random_seed": 42,
      "parallel_execution": false,
      "max_workers": 1
    },
    "scenarios": [
      {
        "id": "scenario_id",
        "name": "Scenario Name",
        "type": "scenario_type",
        "enabled": true,
        "priority": 1,
        "config": {}
      }
    ],
    "agents": [
      {
        "id": "agent_id",
        "name": "Agent Name",
        "framework": "agent_framework",
        "enabled": true,
        "config": {}
      }
    ],
    "metrics": {
      "categories": ["cognitive", "business"],
      "custom_metrics": []
    },
    "execution": {
      "runs_per_scenario": 2,
      "max_duration": 0,
      "timeout": 300,
      "retry_on_failure": true,
      "max_retries": 3
    },
    "output": {
      "format": "json",
      "path": "./results",
      "include_detailed_logs": false,
      "include_audit_trail": true
    },
    "validation": {
      "enabled": true,
      "statistical_significance": true,
      "confidence_level": 0.95,
      "reproducibility_check": true
    },
    "metadata": {
      "author": "Author Name",
      "created": "2023-01-01T00:00:00Z",
      "tags": ["tag1", "tag2"]
    },
    "created_at": "2023-01-01T00:00:00Z",
    "updated_at": "2023-01-01T00:00:00Z",
    "created_by": "user_id"
  },
  "message": "Configuration retrieved successfully",
  "errors": []
}
```

#### Create Configuration

Create a new benchmark configuration.

**Endpoint:** `POST /benchmarking/configurations`

**Request Body:**
```json
{
  "benchmark_id": "benchmark_id",
  "name": "Configuration Name",
  "description": "Configuration Description",
  "version": "1.0.0",
  "environment": {
    "deterministic": true,
    "random_seed": 42,
    "parallel_execution": false,
    "max_workers": 1
  },
  "scenarios": [
    {
      "id": "scenario_id",
      "name": "Scenario Name",
      "type": "scenario_type",
      "enabled": true,
      "priority": 1,
      "config": {}
    }
  ],
  "agents": [
    {
      "id": "agent_id",
      "name": "Agent Name",
      "framework": "agent_framework",
      "enabled": true,
      "config": {}
    }
  ],
  "metrics": {
    "categories": ["cognitive", "business"],
    "custom_metrics": []
  },
  "execution": {
    "runs_per_scenario": 2,
    "max_duration": 0,
    "timeout": 300,
    "retry_on_failure": true,
    "max_retries": 3
  },
  "output": {
    "format": "json",
    "path": "./results",
    "include_detailed_logs": false,
    "include_audit_trail": true
  },
  "validation": {
    "enabled": true,
    "statistical_significance": true,
    "confidence_level": 0.95,
    "reproducibility_check": true
  },
  "metadata": {
    "author": "Author Name",
    "created": "2023-01-01T00:00:00Z",
    "tags": ["tag1", "tag2"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "config_id",
    "benchmark_id": "benchmark_id",
    "name": "Configuration Name",
    "description": "Configuration Description",
    "version": "1.0.0",
    "created_at": "2023-01-01T00:00:00Z",
    "updated_at": "2023-01-01T00:00:00Z",
    "created_by": "user_id"
  },
  "message": "Configuration created successfully",
  "errors": []
}
```

#### Update Configuration

Update an existing benchmark configuration.

**Endpoint:** `PUT /benchmarking/configurations/{id}`

**Request Body:** Same as Create Configuration

**Response:** Same as Create Configuration

#### Delete Configuration

Delete a benchmark configuration.

**Endpoint:** `DELETE /benchmarking/configurations/{id}`

**Response:**
```json
{
  "success": true,
  "data": null,
  "message": "Configuration deleted successfully",
  "errors": []
}
```

#### Validate Configuration

Validate a benchmark configuration without saving it.

**Endpoint:** `POST /benchmarking/configurations/validate`

**Request Body:** Same as Create Configuration

**Response:**
```json
{
  "success": true,
  "data": {
    "valid": true,
    "errors": []
  },
  "message": "Configuration validation completed",
  "errors": []
}
```

### Benchmark Runs

#### Get All Runs

Retrieve a list of all benchmark runs.

**Endpoint:** `GET /benchmarking/runs`

**Query Parameters:**
- `page` (integer, optional): Page number for pagination (default: 1)
- `limit` (integer, optional): Number of items per page (default: 10)
- `sort` (string, optional): Field to sort by (default: created_at)
- `order` (string, optional): Sort order (asc or desc, default: desc)
- `status` (string, optional): Filter by status (created, running, completed, failed, stopped, timeout)
- `configuration_id` (string, optional): Filter by configuration ID

**Response:**
```json
{
  "success": true,
  "data": {
    "runs": [
      {
        "id": "run_id",
        "benchmark_id": "benchmark_id",
        "configuration_id": "config_id",
        "configuration_name": "Configuration Name",
        "status": "completed",
        "start_time": "2023-01-01T00:00:00Z",
        "end_time": "2023-01-01T01:00:00Z",
        "duration_seconds": 3600,
        "created_at": "2023-01-01T00:00:00Z",
        "created_by": "user_id"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 100,
      "pages": 10
    }
  },
  "message": "Runs retrieved successfully",
  "errors": []
}
```

#### Get Run by ID

Retrieve a specific benchmark run by ID.

**Endpoint:** `GET /benchmarking/runs/{id}`

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "run_id",
    "benchmark_id": "benchmark_id",
    "configuration_id": "config_id",
    "configuration_name": "Configuration Name",
    "status": "completed",
    "start_time": "2023-01-01T00:00:00Z",
    "end_time": "2023-01-01T01:00:00Z",
    "duration_seconds": 3600,
    "scenario_results": [
      {
        "scenario_id": "scenario_id",
        "scenario_name": "Scenario Name",
        "agent_results": [
          {
            "agent_id": "agent_id",
            "agent_name": "Agent Name",
            "success": true,
            "metrics": [
              {
                "name": "metric_name",
                "value": 0.85,
                "unit": "score"
              }
            ],
            "execution_time": 1800,
            "error": null
          }
        ]
      }
    ],
    "created_at": "2023-01-01T00:00:00Z",
    "created_by": "user_id"
  },
  "message": "Run retrieved successfully",
  "errors": []
}
```

#### Start Benchmark Run

Start a new benchmark run.

**Endpoint:** `POST /benchmarking/run`

**Request Body:**
```json
{
  "configuration_id": "config_id",
  "options": {
    "async": true,
    "priority": "normal",
    "tags": ["tag1", "tag2"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "run_id",
    "benchmark_id": "benchmark_id",
    "configuration_id": "config_id",
    "status": "created",
    "start_time": null,
    "end_time": null,
    "duration_seconds": 0,
    "created_at": "2023-01-01T00:00:00Z",
    "created_by": "user_id"
  },
  "message": "Benchmark run started successfully",
  "errors": []
}
```

#### Stop Benchmark Run

Stop a running benchmark run.

**Endpoint:** `DELETE /benchmarking/run/{id}`

**Response:**
```json
{
  "success": true,
  "data": null,
  "message": "Benchmark run stopped successfully",
  "errors": []
}
```

#### Get Run Status

Get the current status of a benchmark run.

**Endpoint:** `GET /benchmarking/run/{id}/status`

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "run_id",
    "benchmark_id": "benchmark_id",
    "status": "running",
    "progress": {
      "total_scenarios": 5,
      "completed_scenarios": 2,
      "percentage": 40
    },
    "start_time": "2023-01-01T00:00:00Z",
    "estimated_end_time": "2023-01-01T02:00:00Z"
  },
  "message": "Run status retrieved successfully",
  "errors": []
}
```

### Benchmark Results

#### Get All Results

Retrieve a list of all benchmark results.

**Endpoint:** `GET /benchmarking/results`

**Query Parameters:**
- `page` (integer, optional): Page number for pagination (default: 1)
- `limit` (integer, optional): Number of items per page (default: 10)
- `sort` (string, optional): Field to sort by (default: created_at)
- `order` (string, optional): Sort order (asc or desc, default: desc)
- `configuration_id` (string, optional): Filter by configuration ID
- `run_id` (string, optional): Filter by run ID

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "result_id",
        "benchmark_id": "benchmark_id",
        "configuration_id": "config_id",
        "run_id": "run_id",
        "status": "completed",
        "overall_score": 0.85,
        "start_time": "2023-01-01T00:00:00Z",
        "end_time": "2023-01-01T01:00:00Z",
        "duration_seconds": 3600,
        "created_at": "2023-01-01T00:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 100,
      "pages": 10
    }
  },
  "message": "Results retrieved successfully",
  "errors": []
}
```

#### Get Result by ID

Retrieve a specific benchmark result by ID.

**Endpoint:** `GET /benchmarking/results/{id}`

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "result_id",
    "benchmark_id": "benchmark_id",
    "configuration_id": "config_id",
    "run_id": "run_id",
    "status": "completed",
    "overall_score": 0.85,
    "start_time": "2023-01-01T00:00:00Z",
    "end_time": "2023-01-01T01:00:00Z",
    "duration_seconds": 3600,
    "scenario_results": [
      {
        "scenario_id": "scenario_id",
        "scenario_name": "Scenario Name",
        "agent_results": [
          {
            "agent_id": "agent_id",
            "agent_name": "Agent Name",
            "success": true,
            "metrics": [
              {
                "name": "metric_name",
                "value": 0.85,
                "unit": "score"
              }
            ],
            "execution_time": 1800,
            "error": null
          }
        ]
      }
    ],
    "created_at": "2023-01-01T00:00:00Z"
  },
  "message": "Result retrieved successfully",
  "errors": []
}
```

#### Compare Results

Compare multiple benchmark results.

**Endpoint:** `POST /benchmarking/results/compare`

**Request Body:**
```json
{
  "result_ids": ["result_id_1", "result_id_2"],
  "metrics": ["overall_score", "cognitive_score", "business_score"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "comparison": {
      "result_id_1": {
        "benchmark_id": "benchmark_id_1",
        "overall_score": 0.85,
        "cognitive_score": 0.9,
        "business_score": 0.8
      },
      "result_id_2": {
        "benchmark_id": "benchmark_id_2",
        "overall_score": 0.75,
        "cognitive_score": 0.8,
        "business_score": 0.7
      }
    },
    "differences": {
      "overall_score": 0.1,
      "cognitive_score": 0.1,
      "business_score": 0.1
    }
  },
  "message": "Results compared successfully",
  "errors": []
}
```

### Configuration Templates

#### Get All Templates

Retrieve a list of all configuration templates.

**Endpoint:** `GET /benchmarking/configuration-templates`

**Query Parameters:**
- `category` (string, optional): Filter by category

**Response:**
```json
{
  "success": true,
  "data": {
    "templates": [
      {
        "id": "template_id",
        "name": "Template Name",
        "description": "Template Description",
        "category": "general",
        "config": {
          "benchmark_id": "template_benchmark",
          "name": "Template Benchmark",
          "description": "Template Description",
          "version": "1.0.0",
          "environment": {
            "deterministic": true,
            "random_seed": 42,
            "parallel_execution": false,
            "max_workers": 1
          },
          "scenarios": [],
          "agents": [],
          "metrics": {
            "categories": ["cognitive", "business"],
            "custom_metrics": []
          },
          "execution": {
            "runs_per_scenario": 2,
            "max_duration": 0,
            "timeout": 300,
            "retry_on_failure": true,
            "max_retries": 3
          },
          "output": {
            "format": "json",
            "path": "./results",
            "include_detailed_logs": false,
            "include_audit_trail": true
          },
          "validation": {
            "enabled": true,
            "statistical_significance": true,
            "confidence_level": 0.95,
            "reproducibility_check": true
          },
          "metadata": {
            "author": "Template Author",
            "created": "2023-01-01T00:00:00Z",
            "tags": ["template"]
          }
        }
      }
    ]
  },
  "message": "Templates retrieved successfully",
  "errors": []
}
```

#### Get Template by ID

Retrieve a specific configuration template by ID.

**Endpoint:** `GET /benchmarking/configuration-templates/{id}`

**Response:** Same as Get All Templates, but with a single template

### Agent Frameworks

#### Get All Frameworks

Retrieve a list of all available agent frameworks.

**Endpoint:** `GET /benchmarking/agent-frameworks`

**Response:**
```json
{
  "success": true,
  "data": {
    "frameworks": [
      {
        "id": "framework_id",
        "name": "Framework Name",
        "description": "Framework Description",
        "version": "1.0.0",
        "capabilities": ["reasoning", "planning"],
        "config_schema": {
          "type": "object",
          "properties": {
            "model": {
              "type": "string",
              "description": "Model name"
            },
            "temperature": {
              "type": "number",
              "description": "Temperature setting",
              "minimum": 0,
              "maximum": 1
            }
          },
          "required": ["model"]
        }
      }
    ]
  },
  "message": "Frameworks retrieved successfully",
  "errors": []
}
```

#### Get Framework by ID

Retrieve a specific agent framework by ID.

**Endpoint:** `GET /benchmarking/agent-frameworks/{id}`

**Response:** Same as Get All Frameworks, but with a single framework

### Scenario Types

#### Get All Types

Retrieve a list of all available scenario types.

**Endpoint:** `GET /benchmarking/scenario-types`

**Response:**
```json
{
  "success": true,
  "data": {
    "types": [
      {
        "id": "type_id",
        "name": "Type Name",
        "description": "Type Description",
        "config_schema": {
          "type": "object",
          "properties": {
            "duration": {
              "type": "integer",
              "description": "Duration in seconds",
              "minimum": 1
            },
            "complexity": {
              "type": "string",
              "description": "Complexity level",
              "enum": ["low", "medium", "high"]
            }
          },
          "required": ["duration"]
        }
      }
    ]
  },
  "message": "Types retrieved successfully",
  "errors": []
}
```

#### Get Type by ID

Retrieve a specific scenario type by ID.

**Endpoint:** `GET /benchmarking/scenario-types/{id}`

**Response:** Same as Get All Types, but with a single type

### Metrics

#### Get All Metrics

Retrieve a list of all available metrics.

**Endpoint:** `GET /benchmarking/metrics`

**Query Parameters:**
- `category` (string, optional): Filter by category

**Response:**
```json
{
  "success": true,
  "data": {
    "metrics": [
      {
        "id": "metric_id",
        "name": "Metric Name",
        "description": "Metric Description",
        "category": "cognitive",
        "unit": "score",
        "range": {
          "min": 0,
          "max": 1
        },
        "config_schema": {
          "type": "object",
          "properties": {},
          "required": []
        }
      }
    ]
  },
  "message": "Metrics retrieved successfully",
  "errors": []
}
```

#### Get Metric by ID

Retrieve a specific metric by ID.

**Endpoint:** `GET /benchmarking/metrics/{id}`

**Response:** Same as Get All Metrics, but with a single metric

### Reports

#### Generate Report

Generate a report for benchmark results.

**Endpoint:** `POST /benchmarking/reports`

**Request Body:**
```json
{
  "result_ids": ["result_id_1", "result_id_2"],
  "template": "default",
  "format": "pdf",
  "options": {
    "include_charts": true,
    "include_detailed_metrics": true,
    "include_raw_data": false
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "report_id": "report_id",
    "download_url": "https://example.com/api/v1/benchmarking/reports/report_id/download",
    "expires_at": "2023-01-01T01:00:00Z"
  },
  "message": "Report generated successfully",
  "errors": []
}
```

#### Download Report

Download a generated report.

**Endpoint:** `GET /benchmarking/reports/{id}/download`

**Response:** The report file in the specified format

### WebSocket Events

The API provides WebSocket endpoints for real-time updates on benchmark runs.

#### Connect to WebSocket

**Endpoint:** `ws://localhost:8000/api/v1/benchmarking/ws`

**Authentication:** Include the JWT token as a query parameter:
```
ws://localhost:8000/api/v1/benchmarking/ws?token=<your-jwt-token>
```

#### Events

##### Run Started

Emitted when a benchmark run starts.

```json
{
  "type": "run_started",
  "data": {
    "run_id": "run_id",
    "benchmark_id": "benchmark_id",
    "configuration_id": "config_id",
    "start_time": "2023-01-01T00:00:00Z"
  }
}
```

##### Run Progress

Emitted periodically during a benchmark run to provide progress updates.

```json
{
  "type": "run_progress",
  "data": {
    "run_id": "run_id",
    "progress": {
      "total_scenarios": 5,
      "completed_scenarios": 2,
      "percentage": 40
    },
    "current_scenario": {
      "scenario_id": "scenario_id",
      "scenario_name": "Scenario Name",
      "current_agent": {
        "agent_id": "agent_id",
        "agent_name": "Agent Name"
      }
    }
  }
}
```

##### Run Completed

Emitted when a benchmark run completes successfully.

```json
{
  "type": "run_completed",
  "data": {
    "run_id": "run_id",
    "benchmark_id": "benchmark_id",
    "configuration_id": "config_id",
    "status": "completed",
    "start_time": "2023-01-01T00:00:00Z",
    "end_time": "2023-01-01T01:00:00Z",
    "duration_seconds": 3600,
    "overall_score": 0.85
  }
}
```

##### Run Failed

Emitted when a benchmark run fails.

```json
{
  "type": "run_failed",
  "data": {
    "run_id": "run_id",
    "benchmark_id": "benchmark_id",
    "configuration_id": "config_id",
    "status": "failed",
    "start_time": "2023-01-01T00:00:00Z",
    "end_time": "2023-01-01T00:30:00Z",
    "duration_seconds": 1800,
    "error": "Error message"
  }
}
```

##### Run Stopped

Emitted when a benchmark run is stopped by the user.

```json
{
  "type": "run_stopped",
  "data": {
    "run_id": "run_id",
    "benchmark_id": "benchmark_id",
    "configuration_id": "config_id",
    "status": "stopped",
    "start_time": "2023-01-01T00:00:00Z",
    "end_time": "2023-01-01T00:30:00Z",
    "duration_seconds": 1800
  }
}
```

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 400 | Bad Request | The request is invalid or cannot be served |
| 401 | Unauthorized | Authentication is required and has failed or has not yet been provided |
| 403 | Forbidden | The user does not have permission to access the requested resource |
| 404 | Not Found | The requested resource could not be found |
| 422 | Unprocessable Entity | The request was well-formed but contains semantic errors |
| 500 | Internal Server Error | An unexpected error occurred on the server |

## Skills System (Lightweight Modules)

A lightweight, deterministic skills system is available for agents/runners to call small, dependency-light capabilities with a uniform API.

- Overview and usage: [`docs/api-reference/skills.md`](docs/api-reference/skills.md:1)
- Registry entry points: [`agents/skill_modules/registry.py`](agents/skill_modules/registry.py:1)
- Core interface: [`agents/skill_modules/base.py`](agents/skill_modules/base.py:1)

Quick example:

```python
from agents.skill_modules.registry import create

calc = create("calculator")
out = calc.run({"expression": "12*(3+4)"})
# -> {"result": 84.0, "steps": ["parsed AST", "evaluated nodes safely"]}
```

## Benchmarking Engine (Lightweight Orchestration API)

This repository includes a production-ready, lightweight benchmarking engine that orchestrates end-to-end runs across scenarios, agent runners, metrics, and validators with robust error handling, concurrency limits, and optional Redis Pub/Sub progress.

Key classes and helpers (import from `benchmarking.core.engine`):
- [`python.class Engine(config: EngineConfig)`](benchmarking/core/engine.py:1)
- [`python.async def Engine.run(self) -> EngineReport`](benchmarking/core/engine.py:1)
- [`python.def run_benchmark(config: dict|EngineConfig) -> EngineReport`](benchmarking/core/engine.py:1)
- Models:
  - [`python.class RunnerSpec(BaseModel)`](benchmarking/core/engine.py:1)
  - [`python.class ScenarioSpec(BaseModel)`](benchmarking/core/engine.py:1)
  - [`python.class EngineConfig(BaseModel)`](benchmarking/core/engine.py:1)
  - [`python.class RunResult(BaseModel)`](benchmarking/core/engine.py:1)
  - [`python.class ScenarioReport(BaseModel)`](benchmarking/core/engine.py:1)
  - [`python.class EngineReport(BaseModel)`](benchmarking/core/engine.py:1)
- Reporting helpers:
  - [`python.def summarize_scenario(report: ScenarioReport) -> dict`](benchmarking/core/engine.py:1)
  - [`python.def compute_totals(scenario_reports: list[ScenarioReport]) -> dict`](benchmarking/core/engine.py:1)

Configuration models (Pydantic v2):
- RunnerSpec: `key: str`, `config: dict`
- ScenarioSpec: `key: str`, `params: dict|None`, `repetitions: int=1`, `seeds: list[int]|None`, `timeout_seconds: int|None`
- EngineConfig:
  - `scenarios: list[ScenarioSpec]`
  - `runners: list[RunnerSpec]`
  - `metrics: list[str] = []`
  - `validators: list[str] = []`
  - `parallelism: int = 1`
  - `retries: int = 0`
  - `observation_topic_prefix: str = "benchmark"`

Minimal example
```python
from benchmarking.core.engine import EngineConfig, RunnerSpec, ScenarioSpec, run_benchmark

config = EngineConfig(
    scenarios=[ScenarioSpec(key="my_pkg.scenarios:my_scenario", repetitions=2, seeds=[1, 2], timeout_seconds=5)],
    runners=[RunnerSpec(key="diy", config={"agent_id": "baseline-1"})],
    metrics=["technical_performance"],    # via benchmarking.metrics.registry
    validators=["dummy_validator"],       # via benchmarking.validators.registry
    parallelism=2,
    retries=1,
)

report = run_benchmark(config)
print(report.totals)
```

Orchestration pipeline
1) Resolve scenario by key via scenario registry or dotted import `module:callable_or_Class`.
2) For each runner x repetition/seed:
   - Create runner via agent runner registry [`python.def create_runner`](agent_runners/registry.py:1)
   - Execute scenario with optional timeout; apply retries for failures/errors.
3) Apply per-run metrics; failures logged and do not abort.
4) Compute scenario aggregates with [`python.def summarize_scenario`](benchmarking/core/engine.py:1).
5) Run validators and attach validation results to scenario aggregates.
6) Compute global totals with [`python.def compute_totals`](benchmarking/core/engine.py:1).
7) Optionally publish progress to Redis (topic prefix `observation_topic_prefix`).

Metrics and validators
- Metrics are resolved using [`benchmarking.metrics.registry.MetricRegistry`](benchmarking/metrics/registry.py:1). Use metric keys (e.g., `"technical_performance"`).
- Validators are resolved using [`benchmarking.validators.registry.ValidatorRegistry`](benchmarking/validators/registry.py:1). Use validator keys you’ve registered.

Metrics (function-style)
- The engine now supports a function-style metrics registry in addition to legacy class-based metrics.
- Register and use via:
  - [`python.def register_metric(key: str, fn: Callable[[dict, dict|None], dict]) -> None`](benchmarking/metrics/registry.py:1)
  - [`python.def get_metric(key: str) -> Callable`](benchmarking/metrics/registry.py:1)
  - [`python.def list_metrics() -> list[str]`](benchmarking/metrics/registry.py:1)
- Interface for each metric function:
  - [`python.def evaluate(run: dict, context: dict|None=None) -> dict`](benchmarking/metrics/technical_performance_v2.py:1)
    - run is a dict representation of [`python.class RunResult`](benchmarking/core/engine.py:793) (e.g., `RunResult.model_dump()`).
    - context may include scenario_key, params, or scenario outputs if needed.
- Built-in metrics (auto-registered on import) with typed Pydantic v2 schemas and deterministic behavior:
  - [`benchmarking/metrics/technical_performance_v2.py`](benchmarking/metrics/technical_performance_v2.py:1) key: "technical_performance"
  - [`benchmarking/metrics/accuracy_score.py`](benchmarking/metrics/accuracy_score.py:1) key: "accuracy_score"
  - [`benchmarking/metrics/keyword_coverage.py`](benchmarking/metrics/keyword_coverage.py:1) key: "keyword_coverage"
  - [`benchmarking/metrics/policy_compliance.py`](benchmarking/metrics/policy_compliance.py:1) key: "policy_compliance"
  - [`benchmarking/metrics/robustness.py`](benchmarking/metrics/robustness.py:1) key: "robustness"
  - [`benchmarking/metrics/cost_efficiency.py`](benchmarking/metrics/cost_efficiency.py:1) key: "cost_efficiency"
  - [`benchmarking/metrics/completeness.py`](benchmarking/metrics/completeness.py:1) key: "completeness"
  - [`benchmarking/metrics/custom_scriptable.py`](benchmarking/metrics/custom_scriptable.py:1) key: "custom_scriptable"

Aggregation utilities
- Per-metric aggregation and helpers:
  - [`python.def aggregate_metric_values(runs: list[dict], metric_key: str) -> dict`](benchmarking/metrics/aggregate.py:1)
  - [`python.def aggregate_all(runs: list[dict], metric_keys: list[str]) -> dict`](benchmarking/metrics/aggregate.py:1)
- These compute mean/median/stddev for numeric outputs, boolean success rates where applicable, and field-wise stats for dict-valued metrics. They handle missing values safely.

Concurrency and isolation
- Concurrency is limited by `parallelism` via `asyncio.Semaphore`.
- Per-run isolation: deterministic seeds passed into scenario payload; inputs deep-copied; no mutable state is shared.

Timeouts and retries
- `timeout_seconds` per ScenarioSpec enforces `asyncio.wait_for` around scenario execution.
- `retries` reattempts failed/error runs with deterministic behavior. Timeouts are marked `"timeout"` and are not retried.

Pub/Sub progress (optional)
- If `REDIS_URL` is present and `fba_bench_api.core.redis_client` is importable:
  - Topic: `"{observation_topic_prefix}:scenario:{scenario_key}"`
  - Events: `{"type":"run_started"|"run_finished","runner":runner_key,"seed":seed,"status":...}`

Registering components
- Scenarios: `from benchmarking.scenarios.registry import scenario_registry; scenario_registry.register("my_scenario", MyScenarioOrFunction)`
- Metrics: register via MetricRegistry or use built-ins (e.g., `technical_performance`).
- Validators: register via ValidatorRegistry.

Example EngineConfig JSON (inline with Pydantic examples)
```json
{
  "scenarios":[{"key":"example_scenario","params":{"difficulty":"easy"},"repetitions":2,"seeds":[1,2],"timeout_seconds":5}],
  "runners":[{"key":"diy","config":{"agent_id":"baseline-1"}}],
  "metrics":["technical_performance"],
  "validators":["basic_consistency"],
  "parallelism":2,
  "retries":1,
  "observation_topic_prefix":"benchmark"
}
```

## Validators

Validators provide post-run and scenario-level quality checks with deterministic behavior and safe defaults. They are referenced by key in EngineConfig, discovered via the function-style validator registry, and attached to each ScenarioReport under aggregates["validations"].

- Registry helpers:
  - [`python.def register_validator(key: str, fn: Callable[[dict, dict|None], dict]) -> None`](benchmarking/validators/registry.py:1)
  - [`python.def get_validator(key: str) -> Callable`](benchmarking/validators/registry.py:1)
  - [`python.def list_validators() -> list[str]`](benchmarking/validators/registry.py:1)

- Standard callable interface:
  - [`python.def validate(report: dict, context: dict|None=None) -> dict`](benchmarking/validators/types.py:1)
    - report: dict representation of [`python.class ScenarioReport`](benchmarking/core/engine.py:823) (i.e., model_dump())
    - context: optional dict including scenario_key, params, expected_seeds, config_digest, etc.
  - Normalized return:
    - {"issues": list[dict], "summary": dict}
      - issue: {"id": str, "severity": "info"|"warning"|"error", "message": str, "path": list[str]|None}

Engine usage (summary):
1) Engine runs scenarios and collects RunResult entries.
2) Engine aggregates scenario metrics with [`python.def summarize_scenario`](benchmarking/core/engine.py:1).
3) Engine applies validators by key and attaches results to scenario_report["aggregates"]["validations"].

Built-in validators (auto-registered on import):
- Structural: "structural_consistency" → [`benchmarking/validators/structural_consistency.py`](benchmarking/validators/structural_consistency.py:1)
- Determinism: "determinism_check" → [`benchmarking/validators/determinism_check.py`](benchmarking/validators/determinism_check.py:1)
- Reproducibility metadata: "reproducibility_metadata" → [`benchmarking/validators/reproducibility_metadata.py`](benchmarking/validators/reproducibility_metadata.py:1)
- Schema adherence: "schema_adherence" → [`benchmarking/validators/schema_adherence.py`](benchmarking/validators/schema_adherence.py:1)
- Outlier detection: "outlier_detection" → [`benchmarking/validators/outlier_detection.py`](benchmarking/validators/outlier_detection.py:1)
- Fairness/balance: "fairness_balance" → [`benchmarking/validators/fairness_balance.py`](benchmarking/validators/fairness_balance.py:1)

See the dedicated reference for details and examples: [docs/api-reference/validators.md](docs/api-reference/validators.md:1)


## Frontend Features

This frontend implements Experiments, Agents, and Results with realtime updates and accessible UI.

Key pages and modules:
- Experiments
  - List, create, start simulation, and live status updates:
    - [`tsx.function Experiments()`](frontend/src/pages/Experiments.tsx:1)
    - Detail view with metrics and validators:
      - [`tsx.function ExperimentDetail()`](frontend/src/pages/ExperimentDetail.tsx:1)
    - Components:
      - [`tsx.function ExperimentForm(props)`](frontend/src/components/experiments/ExperimentForm.tsx:1)
      - [`tsx.function ExperimentList(props)`](frontend/src/components/experiments/ExperimentList.tsx:1)
- Agents
  - List, create, edit, delete with optimistic UI:
    - [`tsx.function Agents()`](frontend/src/pages/Agents.tsx:1)
    - Form:
      - [`tsx.function AgentForm(props)`](frontend/src/components/agents/AgentForm.tsx:1)
- Results
  - Select experiment, fetch EngineReport, metrics and validators, refresh, and realtime-triggered refresh:
    - [`tsx.function Results()`](frontend/src/pages/Results.tsx:1)

Shared services and realtime:
- API client core + typed helpers:
  - Core fetch helper:
    - [`ts.function apiFetch()`](frontend/src/services/api.ts:1)
  - Typed endpoints:
    - [`ts.function getExperiments()`](frontend/src/services/api.ts:1)
    - [`ts.function createExperiment()`](frontend/src/services/api.ts:1)
    - [`ts.function getExperiment()`](frontend/src/services/api.ts:1)
    - [`ts.function startSimulation()`](frontend/src/services/api.ts:1)
    - [`ts.function getAgents()`](frontend/src/services/api.ts:1)
    - [`ts.function createAgent()`](frontend/src/services/api.ts:1)
    - [`ts.function updateAgent()`](frontend/src/services/api.ts:1)
    - [`ts.function deleteAgent()`](frontend/src/services/api.ts:1)
    - [`ts.function getEngineReport()`](frontend/src/services/api.ts:1)
- Realtime client:
  - Connect and manage subscriptions/publish/heartbeat:
    - [`ts.function connectRealtime()`](frontend/src/services/realtime.ts:1)
  - Topic constants:
    - SIMULATION_TOPIC_PREFIX, EXPERIMENT_TOPIC_PREFIX in [`ts.module realtime`](frontend/src/services/realtime.ts:1)

UI primitives:
- Accessible modal, badges, responsive table:
  - [`tsx.function Modal()`](frontend/src/components/ui/Modal.tsx:1)
  - [`tsx.function Badge()`](frontend/src/components/ui/Badge.tsx:1)
  - [`tsx.function Table()`](frontend/src/components/ui/Table.tsx:1)

Environment configuration:
- Create a .env file in frontend/ with:
  - VITE_API_URL=http://localhost:8000
  - VITE_WS_URL=ws://localhost:8000
  - VITE_REALTIME_URL=ws://localhost:8000/ws/realtime
- Optionally set:
  - VITE_API_TIMEOUT, VITE_API_RETRY_ATTEMPTS, VITE_API_RETRY_DELAY_MS, VITE_ALLOW_API_KEY_AUTH

Routing:
- App routes include detail view:
  - [`tsx.function AppRoutes()`](frontend/src/routes/index.tsx:1)
  - Paths:
    - /experiments
    - /experiments/:id
    - /agents
    - /results

How to run:
- Install: cd frontend && npm i
- Dev server: npm run dev (http://localhost:5173)
- Tests: npm test

Mock-friendly fallbacks:
- If a backend route returns 404, the API client translates this to a typed RouteNotAvailableError, and pages surface actionable toasts while remaining usable (e.g., manual refresh, realtime optional).
- Realtime is resilient: if WebSocket is unavailable, the client degrades to no-op; UI still functions with manual refresh.

## Frontend Authentication (Mock)

This project includes a lightweight, mock-friendly authentication scaffold that can be toggled via environment variables. It protects feature routes in development and provides a simple login flow with accessibility and test coverage.

Environment toggle:
- VITE_AUTH_ENABLED: "true" to enable guard and login, "false" to disable and auto-authenticate
- VITE_AUTH_DEFAULT_USER: Default dev user email when auth is disabled; used as login prefill
- VITE_AUTH_DEFAULT_ROLE: Optional role string for the mock user

Example .env (see frontend/.env.example):
- VITE_AUTH_ENABLED=true
- VITE_AUTH_DEFAULT_USER=devuser@example.com
- VITE_AUTH_DEFAULT_ROLE=admin

Protected routes (when VITE_AUTH_ENABLED=true):
- /experiments
- /experiments/:id
- /agents
- /results

Public route:
- /login

Behavior:
- When disabled (VITE_AUTH_ENABLED=false): The app behaves as before; routes are accessible without login. A "Dev Mode" badge is shown in the header.
- When enabled (VITE_AUTH_ENABLED=true): Visiting a protected route while unauthenticated redirects to /login?redirect=..., logging in authenticates and then navigates to the redirect target (or "/" by default). Sign-out returns to /login.

Code locations:
- Auth provider and hook:
  - [tsx.function AuthProvider()](frontend/src/contexts/AuthContext.tsx:1)
  - [tsx.function useAuth()](frontend/src/contexts/AuthContext.tsx:1)
- Route guard:
  - [tsx.function PrivateRoute()](frontend/src/routes/PrivateRoute.tsx:1)
- Login page:
  - [tsx.function Login()](frontend/src/pages/Login.tsx:1)

API client token passthrough:
- The API client supports an optional Authorization header via a provider the Auth layer can register:
  - [ts.function setAuthTokenProvider()](frontend/src/services/api.ts:1)
- When auth is enabled and a user is logged in, the scaffold registers a static "dev-token". When disabled or logged out, the provider is cleared. If backend ignores Authorization, no behavior changes occur.

Upgrading to real auth (JWT/OIDC):
- Replace mock login with real auth (e.g., OIDC/OAuth redirect or embedded login)
- Store and refresh a real access token in the AuthProvider
- Update setAuthTokenProvider(() =&gt; actualAccessToken) after login/refresh
- Consider secure storage patterns and CSRF protections if moving to cookie-based session flows

How to run with auth scaffold:
- Set VITE_AUTH_ENABLED in frontend/.env
- npm i (if new files only, no new deps expected)
- npm run dev
- npm test
