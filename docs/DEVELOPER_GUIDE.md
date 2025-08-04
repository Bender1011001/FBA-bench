# Developer Guide

This guide provides detailed information for developers who want to contribute to the Functional Benchmarking Application (FBA) or extend its functionality.

## Architecture Overview

The FBA is built using a microservices architecture with a clear separation between the backend API server and the frontend React application. The backend is built with Python using FastAPI, while the frontend is built with React and TypeScript.

### Backend Architecture

The backend consists of several key modules:

1. **Core Engine**: The heart of the benchmarking system, responsible for executing benchmarks and collecting results.
2. **Configuration Management**: Handles validation and management of benchmark configurations.
3. **Integration Framework**: Provides adapters for integrating with different agent frameworks.
4. **Metrics System**: Collects and calculates various metrics for evaluating agent performance.
5. **Scenarios System**: Defines and manages different benchmark scenarios.
6. **API Server**: Exposes RESTful endpoints for the frontend to interact with.

### Frontend Architecture

The frontend is a single-page application built with React and TypeScript, using modern React patterns and best practices:

1. **Component-Based Architecture**: The UI is built using reusable components.
2. **State Management**: Uses React hooks for state management.
3. **Service Layer**: Abstracts API calls and other business logic.
4. **Routing**: Uses React Router for navigation.

## Development Setup

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Git

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

## Code Structure

### Backend Code Structure

```
benchmarking/
├── core/                  # Core benchmarking engine
│   ├── engine.py         # Main benchmark execution engine
│   ├── models.py         # Data models for benchmarks, results, etc.
│   └── status.py         # Benchmark status enumeration
├── config/               # Configuration management
│   ├── manager.py        # Configuration validation and management
│   └── schema.py         # JSON schema definitions
├── integration/          # Integration framework
│   ├── manager.py        # Integration manager for agent frameworks
│   ├── agent_adapter.py  # Adapter interface for agent frameworks
│   └── metrics_adapter.py # Adapter interface for metrics collection
├── metrics/              # Metrics system
│   ├── registry.py       # Registry for available metrics
│   ├── base.py           # Base classes for implementing metrics
│   ├── cognitive/        # Cognitive metrics implementations
│   ├── business/         # Business metrics implementations
│   └── technical/        # Technical metrics implementations
├── scenarios/            # Scenarios system
│   ├── registry.py       # Registry for available scenarios
│   ├── base.py           # Base classes for implementing scenarios
│   └── test_scenarios/   # Test scenario implementations
└── utils/                # Utility functions and helpers
    ├── logging.py        # Logging utilities
    ├── validation.py     # Validation utilities
    └── helpers.py        # General helper functions

api/
├── main.py               # FastAPI application entry point
├── routes/               # API endpoints
│   ├── auth.py           # Authentication endpoints
│   ├── configurations.py # Configuration endpoints
│   ├── runs.py           # Benchmark run endpoints
│   ├── results.py        # Benchmark result endpoints
│   ├── templates.py      # Configuration template endpoints
│   ├── frameworks.py     # Agent framework endpoints
│   ├── scenarios.py      # Scenario type endpoints
│   ├── metrics.py        # Metric endpoints
│   └── reports.py        # Report endpoints
├── middleware/           # Custom middleware
│   ├── auth.py           # Authentication middleware
│   ├── logging.py        # Logging middleware
│   └── cors.py           # CORS middleware
├── database/             # Database setup and models
│   ├── setup.py          # Database setup script
│   ├── models.py         # Database models
│   └── migrations/       # Database migrations
└── websocket/            # WebSocket endpoints
    ├── main.py           # WebSocket server
    └── events.py         # WebSocket event handlers
```

### Frontend Code Structure

```
frontend/src/
├── components/           # React components
│   ├── Dashboard/        # Dashboard components
│   ├── BenchmarkRunner/  # Benchmark runner components
│   ├── ScenarioBuilder/  # Scenario builder components
│   ├── ConfigurationEditor/ # Configuration editor components
│   ├── MetricsVisualization/ # Metrics visualization components
│   ├── ReportGenerator/  # Report generator components
│   ├── common/           # Common components (buttons, inputs, etc.)
│   └── layout/           # Layout components (header, sidebar, etc.)
├── hooks/               # Custom React hooks
│   ├── useApi.js        # Hook for API calls
│   ├── useAuth.js       # Hook for authentication
│   ├── useWebSocket.js  # Hook for WebSocket connections
│   └── useLocalStorage.js # Hook for local storage
├── services/            # Service layer
│   ├── apiService.js    # API service for HTTP requests
│   ├── authService.js   # Authentication service
│   ├── websocketService.js # WebSocket service
│   └── notificationService.js # Notification service
├── utils/               # Utility functions
│   ├── validation.js    # Validation utilities
│   ├── formatting.js    # Formatting utilities
│   ├── dateHelpers.js   # Date helper functions
│   └── constants.js     # Application constants
├── config/              # Configuration files
│   ├── api.js           # API configuration
│   ├── routes.js        # Route configuration
│   ├── agentFrameworks.js # Agent framework configuration
│   ├── scenarioTypes.js # Scenario type configuration
│   └── metrics.js       # Metric configuration
├── styles/              # CSS styles
│   ├── global.css       # Global styles
│   ├── variables.css    # CSS variables
│   └── components/      # Component-specific styles
├── types/               # TypeScript type definitions
│   ├── api.ts           # API response types
│   ├── benchmark.ts     # Benchmark-related types
│   ├── configuration.ts # Configuration-related types
│   └── common.ts        # Common types
├── App.tsx              # Main App component
├── index.tsx            # Entry point
└── setupTests.ts        # Test setup
```

## Development Guidelines

### Backend Development Guidelines

#### Python Code Style

- Follow PEP 8 style guidelines.
- Use type hints for all function signatures and variables.
- Use docstrings for all public functions, classes, and modules.
- Use meaningful variable and function names.
- Keep functions small and focused on a single responsibility.
- Use async/await for asynchronous operations.

#### Error Handling

- Use custom exception classes for different types of errors.
- Handle exceptions at the appropriate level.
- Provide meaningful error messages.
- Log errors with sufficient context.
- Use HTTP status codes appropriately in API responses.

#### Testing

- Write unit tests for all functions and classes.
- Write integration tests for API endpoints.
- Use pytest for testing.
- Mock external dependencies in tests.
- Aim for high test coverage.

#### Database

- Use SQLAlchemy ORM for database operations.
- Define clear database models with relationships.
- Use Alembic for database migrations.
- Avoid N+1 query problems.
- Use database transactions appropriately.

### Frontend Development Guidelines

#### TypeScript and React

- Use TypeScript for all React components and utilities.
- Use functional components with hooks.
- Use the useState, useEffect, and useContext hooks appropriately.
- Create custom hooks for reusable logic.
- Use React.memo for performance optimization when necessary.

#### Component Structure

- Keep components small and focused on a single responsibility.
- Use composition over inheritance.
- Use props and callbacks for communication between components.
- Use context for global state that is needed by many components.
- Use state management libraries (like Redux) only for complex global state.

#### Styling

- Use CSS modules or styled-components for component-specific styles.
- Use CSS variables for theming.
- Use responsive design with media queries.
- Use a consistent design system.
- Use Tailwind CSS for utility classes.

#### Testing

- Write unit tests for all components and utilities.
- Write integration tests for user flows.
- Use React Testing Library for testing components.
- Mock API calls in tests.
- Aim for high test coverage.

## Extending the Framework

### Adding New Agent Frameworks

To add support for a new agent framework, you need to create an adapter that implements the `AgentAdapter` interface:

1. Create a new adapter class that inherits from `AgentAdapter`:
   ```python
   from benchmarking.integration.agent_adapter import AgentAdapter, AgentExecutionResult
   
   class MyAgentAdapter(AgentAdapter):
       def __init__(self, config):
           super().__init__(config)
           # Initialize your agent framework
       
       async def execute(self, scenario_config):
           try:
               # Execute the scenario with your agent
               # ...
               
               # Return the result
               return AgentExecutionResult(
                   success=True,
                   metrics={
                       "overall_score": 0.85,
                       "cognitive_score": 0.9,
                       "business_score": 0.8,
                       "technical_score": 0.75
                   },
                   execution_time=1800,
                   events=[],
                   error=None
               )
           except Exception as e:
               return AgentExecutionResult(
                   success=False,
                   metrics={},
                   execution_time=0,
                   events=[],
                   error=str(e)
               )
   ```

2. Register the adapter in the integration manager:
   ```python
   from benchmarking.integration.manager import IntegrationManager
   
   manager = IntegrationManager()
   manager.register_agent_framework('my-framework', MyAgentAdapter)
   ```

3. Add the framework to the frontend configuration in `frontend/src/config/agentFrameworks.js`:
   ```javascript
   export const agentFrameworks = [
     // ... existing frameworks
     {
       id: 'my-framework',
       name: 'My Framework',
       description: 'Description of my framework',
       version: '1.0.0',
       capabilities: ['reasoning', 'planning'],
       configSchema: {
         type: 'object',
         properties: {
           model: {
             type: 'string',
             description: 'Model name'
           },
           temperature: {
             type: 'number',
             description: 'Temperature setting',
             minimum: 0,
             maximum: 1
           }
         },
         required: ['model']
       }
     }
   ];
   ```

### Adding New Metrics

To add a new metric, you need to create a class that implements the `BaseMetric` interface:

1. Create a new metric class that inherits from `BaseMetric`:
   ```python
   from benchmarking.metrics.base import BaseMetric, MetricResult
   
   class MyMetric(BaseMetric):
       def __init__(self, config):
           super().__init__(config)
           # Initialize your metric
       
       async def calculate(self, events, agent_data, scenario_data):
           try:
               # Calculate the metric value
               value = self._calculate_metric(events, agent_data, scenario_data)
               
               # Return the result
               return MetricResult(
                   name="my_metric",
                   value=value,
                   unit="score",
                   min_value=0,
                   max_value=1,
                   metadata={}
               )
           except Exception as e:
               return MetricResult(
                   name="my_metric",
                   value=0,
                   unit="score",
                   min_value=0,
                   max_value=1,
                   metadata={"error": str(e)}
               )
       
       def _calculate_metric(self, events, agent_data, scenario_data):
           # Implement your metric calculation logic here
           # ...
           return 0.85
   ```

2. Register the metric in the metrics registry:
   ```python
   from benchmarking.metrics.registry import metrics_registry
   
   metrics_registry.register('my-metric', MyMetric, 'my-category')
   ```

3. Add the metric to the frontend configuration in `frontend/src/config/metrics.js`:
   ```javascript
   export const metrics = [
     // ... existing metrics
     {
       id: 'my-metric',
       name: 'My Metric',
       description: 'Description of my metric',
       category: 'my-category',
       unit: 'score',
       range: {
         min: 0,
         max: 1
       },
       configSchema: {
         type: 'object',
         properties: {},
         required: []
       }
     }
   ];
   ```

### Adding New Scenarios

To add a new scenario type, you need to create a class that implements the `BaseScenario` interface:

1. Create a new scenario class that inherits from `BaseScenario`:
   ```python
   from benchmarking.scenarios.base import BaseScenario, ScenarioResult
   
   class MyScenario(BaseScenario):
       def __init__(self, config):
           super().__init__(config)
           # Initialize your scenario
       
       async def run(self, agent):
           try:
               # Run the scenario with the agent
               events = []
               start_time = time.time()
               
               # Implement your scenario logic here
               # ...
               
               end_time = time.time()
               execution_time = end_time - start_time
               
               # Return the result
               return ScenarioResult(
                   success=True,
                   events=events,
                   execution_time=execution_time,
                   error=None
               )
           except Exception as e:
               return ScenarioResult(
                   success=False,
                   events=[],
                   execution_time=0,
                   error=str(e)
               )
   ```

2. Register the scenario in the scenarios registry:
   ```python
   from benchmarking.scenarios.registry import scenario_registry
   
   scenario_registry.register('my-scenario', MyScenario)
   ```

3. Add the scenario to the frontend configuration in `frontend/src/config/scenarioTypes.js`:
   ```javascript
   export const scenarioTypes = [
     // ... existing scenario types
     {
       id: 'my-scenario',
       name: 'My Scenario',
       description: 'Description of my scenario',
       configSchema: {
         type: 'object',
         properties: {
           duration: {
             type: 'integer',
             description: 'Duration in seconds',
             minimum: 1
           },
           complexity: {
             type: 'string',
             description: 'Complexity level',
             enum: ['low', 'medium', 'high']
           }
         },
         required: ['duration']
       }
     }
   ];
   ```

### Adding New API Endpoints

To add a new API endpoint:

1. Create a new route file in `api/routes/` or add to an existing one:
   ```python
   from fastapi import APIRouter, Depends, HTTPException
   from typing import List
   from api.middleware.auth import get_current_user
   
   router = APIRouter()
   
   @router.get("/my-endpoint")
   async def get_my_endpoint(current_user: dict = Depends(get_current_user)):
       # Implement your endpoint logic here
       return {"message": "Hello, World!"}
   ```

2. Add the router to the main FastAPI app in `api/main.py`:
   ```python
   from api.routes.my_route import router as my_router
   
   app.include_router(my_router, prefix="/api/v1/my-route", tags=["my-route"])
   ```

3. Add TypeScript types for the API response in `frontend/src/types/api.ts`:
   ```typescript
   export interface MyEndpointResponse {
     message: string;
   }
   ```

4. Add the API call to the API service in `frontend/src/services/apiService.js`:
   ```javascript
   export const getMyEndpoint = async () => {
     const response = await apiService.get('/my-route/my-endpoint');
     return response.data;
   };
   ```

### Adding New Frontend Components

To add a new React component:

1. Create a new component file in the appropriate directory under `frontend/src/components/`:
   ```typescript
   import React from 'react';
   import './MyComponent.css';
   
   interface MyComponentProps {
     title: string;
     onButtonClick?: () => void;
   }
   
   const MyComponent: React.FC<MyComponentProps> = ({ title, onButtonClick }) => {
     return (
       <div className="my-component">
         <h1>{title}</h1>
         <button onClick={onButtonClick}>Click me</button>
       </div>
     );
   };
   
   export default MyComponent;
   ```

2. Create a CSS file for the component in the same directory:
   ```css
   .my-component {
     padding: 1rem;
     border: 1px solid #ccc;
     border-radius: 4px;
   }
   
   .my-component h1 {
     font-size: 1.5rem;
     margin-bottom: 1rem;
   }
   
   .my-component button {
     padding: 0.5rem 1rem;
     background-color: #007bff;
     color: white;
     border: none;
     border-radius: 4px;
     cursor: pointer;
   }
   
   .my-component button:hover {
     background-color: #0069d9;
   }
   ```

3. Create a test file for the component in `frontend/src/components/__tests__/`:
   ```typescript
   import React from 'react';
   import { render, screen, fireEvent } from '@testing-library/react';
   import MyComponent from '../MyComponent';
   
   describe('MyComponent', () => {
     test('renders with title', () => {
       render(<MyComponent title="Test Title" />);
       expect(screen.getByText('Test Title')).toBeInTheDocument();
     });
   
     test('calls onButtonClick when button is clicked', () => {
       const mockOnClick = jest.fn();
       render(<MyComponent title="Test Title" onButtonClick={mockOnClick} />);
       
       fireEvent.click(screen.getByText('Click me'));
       expect(mockOnClick).toHaveBeenCalled();
     });
   });
   ```

## Testing

### Backend Testing

The backend uses pytest for testing. To run the tests:

```bash
pytest tests/
```

To run tests with coverage:

```bash
pytest --cov=benchmarking tests/
```

To run a specific test file:

```bash
pytest tests/unit/test_engine.py
```

To run a specific test:

```bash
pytest tests/unit/test_engine.py::TestBenchmarkEngine::test_init
```

### Frontend Testing

The frontend uses React Testing Library and Jest for testing. To run the tests:

```bash
cd frontend
npm test
```

To run tests with coverage:

```bash
npm run test:coverage
```

To run a specific test file:

```bash
npm test -- MyComponent.test.tsx
```

To run tests in watch mode:

```bash
npm run test:watch
```

## Deployment

### Backend Deployment

The backend can be deployed using Docker:

1. Build the Docker image:
   ```bash
   docker build -t fba-backend .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 fba-backend
   ```

For production deployment, consider using a container orchestration platform like Kubernetes or a cloud service like AWS ECS or Google Cloud Run.

### Frontend Deployment

The frontend can be deployed as a static website:

1. Build the production version:
   ```bash
   cd frontend
   npm run build
   ```

2. Serve the built files using a web server like Nginx or Apache, or deploy to a static hosting service like Netlify, Vercel, or AWS S3.

For a complete deployment guide, see the deployment documentation.

## Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Make your changes.
4. Add tests for your changes.
5. Ensure all tests pass:
   ```bash
   # Backend tests
   pytest tests/
   
   # Frontend tests
   cd frontend
   npm test
   ```
6. Commit your changes:
   ```bash
   git commit -am "Add my feature"
   ```
7. Push to the branch:
   ```bash
   git push origin feature/my-feature
   ```
8. Submit a pull request.

## Code Review Process

All pull requests go through a code review process to ensure quality and consistency. The review process includes:

1. Automated checks (linting, formatting, tests).
2. Review by at least one maintainer.
3. Approval by at least one maintainer.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.