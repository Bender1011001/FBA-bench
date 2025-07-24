# FBA-Bench Integrated Analysis Dashboard

A comprehensive, real-time dashboard for FBA-Bench simulation analysis with multi-tab interface, interactive charts, and live data updates.

## 🏗️ Architecture Overview

The dashboard follows a modern, scalable architecture:

- **Backend**: FastAPI with Pydantic models for automatic validation
- **Frontend**: React + TypeScript with Apache ECharts for visualization
- **Real-time**: WebSocket connections for live updates
- **State Management**: Zustand for efficient React state management
- **Styling**: Tailwind CSS with custom component library

## 📁 Project Structure

```
fba_bench_repo/
├── dashboard/                    # Backend API
│   ├── __init__.py
│   ├── models.py                # Pydantic data models
│   ├── data_exporter.py         # Data extraction layer
│   └── api.py                   # FastAPI backend
├── frontend/                    # React frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   └── src/
│       ├── types/dashboard.ts   # TypeScript types
│       ├── store/dashboardStore.ts  # Zustand state management
│       ├── services/api.ts      # API service layer
│       ├── components/          # React components (to be created)
│       ├── App.tsx             # Main app component
│       ├── index.tsx           # React entry point
│       └── index.css           # Tailwind CSS styles
├── dashboard_example.py         # Integration example
├── dashboard_architecture_plan.md  # Detailed architecture plan
└── requirements.txt            # Updated with dashboard dependencies
```

## 🚀 Quick Start

### Backend Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard API**
   ```bash
   python dashboard_example.py
   ```
   
   Or programmatically:
   ```python
   from fba_bench.simulation import Simulation
   from fba_bench.advanced_agent import AdvancedAgent
   from dashboard import run_dashboard_server
   
   # Create simulation
   sim = Simulation()
   sim.launch_product("B000TEST", "Electronics", cost=5.0, price=19.99, qty=100)
   
   # Create agent
   agent = AdvancedAgent(days=30)
   
   # Run dashboard server
   run_dashboard_server(host="127.0.0.1", port=8000, simulation=sim, agent=agent)
   ```

3. **API Endpoints Available**
   - `GET /` - API status and health
   - `GET /api/health` - Health check
   - `GET /api/dashboard/executive-summary` - Tab 1 data
   - `GET /api/dashboard/financial` - Tab 2 data
   - `GET /api/dashboard/product-market` - Tab 3 data
   - `GET /api/dashboard/supply-chain` - Tab 4 data
   - `GET /api/dashboard/agent-cognition` - Tab 5 data
   - `GET /api/kpis` - Real-time KPI metrics
   - `WS /ws` - WebSocket for real-time updates

### Frontend Setup

1. **Navigate to Frontend Directory**
   ```bash
   cd frontend
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Start Development Server**
   ```bash
   npm start
   ```

4. **Access Dashboard**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📊 Dashboard Features

### Tab 1: Executive Summary
- **KPI Header**: Resilient Net Worth, Daily/Total Profit, Cash Balance, Trust Score, Distress Status
- **Performance Chart**: Time-series visualization with toggleable metrics
- **Agent Status**: Current goals, budget usage, strategic coherence
- **Event Log**: Real-time filtered event stream

### Tab 2: Financial Deep Dive
- **P&L Statement**: Revenue, COGS, expenses by period
- **Fee Breakdown**: Pie chart of Amazon fee types
- **Balance Sheet**: Assets, liabilities, equity overview

### Tab 3: Product & Market Analysis
- **Product Dashboard**: ASIN details, inventory status, turnover
- **BSR Components**: Multi-line chart of BSR calculation inputs
- **Competitor Analysis**: Comparative table with agent highlighting

### Tab 4: Supply Chain & Operations
- **Supplier Overview**: Status, reputation, MOQ, lead times
- **Order Pipeline**: Kanban-style order tracking

### Tab 5: Agent Cognition & Strategy
- **Goal Stack**: Hierarchical task visualization
- **Memory Explorer**: Searchable episodic/semantic/procedural memory
- **Strategic Plan**: Mission, objectives, coherence scoring

## 🔧 Technical Implementation

### Backend Architecture

**Data Export Layer** (`dashboard/data_exporter.py`):
- Non-invasive integration with existing FBA-Bench components
- Extracts data from Simulation, Agent, Ledger, etc.
- Converts to Pydantic models for validation

**API Layer** (`dashboard/api.py`):
- FastAPI with automatic OpenAPI documentation
- In-memory caching with 1-second TTL
- WebSocket support for real-time updates
- CORS enabled for frontend integration

**Data Models** (`dashboard/models.py`):
- Comprehensive Pydantic models for all dashboard data
- Automatic validation and serialization
- Type-safe API contracts

### Frontend Architecture

**State Management** (`frontend/src/store/dashboardStore.ts`):
- Zustand store with TypeScript support
- Async state management for API calls
- WebSocket event handling
- Filter and tab state management

**API Service** (`frontend/src/services/api.ts`):
- Centralized API client with error handling
- WebSocket service for real-time updates
- Retry logic and connection management

**Component Structure**:
- Modular tab-based architecture
- Shared component library
- Responsive design with Tailwind CSS
- Apache ECharts for interactive visualizations

## 🎯 Key Features

### Real-time Updates
- WebSocket connections for live data streaming
- Automatic cache invalidation
- Event-driven UI updates

### Interactive Visualizations
- Apache ECharts for powerful, responsive charts
- Time-series data with zoom and pan
- Toggleable metrics and filtering

### Responsive Design
- Mobile-first approach with Tailwind CSS
- Adaptive layouts for different screen sizes
- Touch-friendly interactions

### Error Handling
- Comprehensive error boundaries
- API error handling with retry logic
- Loading states and user feedback

## 🔌 Integration Examples

### Basic Integration
```python
from dashboard import run_dashboard_server, dashboard_api
from fba_bench.simulation import Simulation
from fba_bench.advanced_agent import AdvancedAgent

# Create simulation components
sim = Simulation()
agent = AdvancedAgent(days=30)

# Connect to dashboard
dashboard_api.set_simulation_components(sim, agent)

# Run server
run_dashboard_server(simulation=sim, agent=agent)
```

### Custom Data Export
```python
from dashboard.data_exporter import DashboardDataExporter

# Create custom exporter
exporter = DashboardDataExporter(simulation, agent)

# Extract specific data
kpis = exporter.extract_kpi_metrics()
financial_data = exporter.extract_financial_deep_dive()
```

### WebSocket Integration
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.event_type === 'kpi_update') {
    // Handle real-time KPI updates
    updateDashboard(data.data);
  }
};
```

## 🧪 Testing

### Backend Testing
```bash
# Test API endpoints
python dashboard_example.py test

# Manual API testing
curl http://localhost:8000/api/health
curl http://localhost:8000/api/kpis
```

### Frontend Testing
```bash
cd frontend
npm test
```

## 📈 Performance Considerations

- **Backend Caching**: 1-second TTL for single-user scenarios
- **Frontend Optimization**: Virtual scrolling for large datasets
- **Real-time Throttling**: Debounced WebSocket updates
- **Lazy Loading**: Non-visible tab content loaded on demand

## 🔮 Future Enhancements

- **Multi-user Support**: Redis caching for scalability
- **Advanced Analytics**: Machine learning insights
- **Export Functionality**: PDF/Excel report generation
- **Custom Dashboards**: User-configurable layouts
- **Dark Mode**: Theme switching support
- **Mobile App**: React Native companion app

## 🐛 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check if backend is running on port 8000
   - Verify CORS settings in API configuration

2. **Frontend Build Errors**
   - Ensure all dependencies are installed: `npm install`
   - Check TypeScript configuration in `tsconfig.json`

3. **API Connection Issues**
   - Verify backend health: `curl http://localhost:8000/api/health`
   - Check proxy configuration in `package.json`

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug
run_dashboard_server(debug=True)
```

## 📝 Development Status

### ✅ Completed
- [x] Pydantic data models and validation
- [x] FastAPI backend with caching
- [x] WebSocket real-time updates
- [x] TypeScript types and interfaces
- [x] Zustand state management
- [x] API service layer
- [x] Project structure and configuration

### 🚧 In Progress
- [ ] React component implementation
- [ ] Apache ECharts integration
- [ ] Tab-specific functionality

### 📋 Pending
- [ ] Interactive filtering
- [ ] Data export functionality
- [ ] Responsive design optimization
- [ ] Comprehensive error handling
- [ ] Documentation and deployment guides

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add TypeScript types for all new interfaces
3. Include error handling and loading states
4. Test both backend and frontend changes
5. Update documentation for new features

## 📄 License

This dashboard is part of the FBA-Bench project and follows the same licensing terms.