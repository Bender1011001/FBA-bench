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