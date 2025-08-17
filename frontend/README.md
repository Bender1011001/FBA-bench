# FBA-Bench Frontend

The FBA-Bench frontend is a modern, responsive web application built with React and TypeScript that provides an intuitive interface for managing and monitoring AI agent benchmarks.

## 🎯 Overview

The frontend serves as the primary user interface for FBA-Bench, offering real-time monitoring, interactive visualizations, and comprehensive management tools for benchmark execution and analysis.

## ✨ Key Features

### 📊 Real-time Monitoring
- **Live Dashboard**: Real-time visualization of benchmark execution
- **WebSocket Integration**: Instant updates for agent actions and metrics
- **Performance Metrics**: Live charts and graphs for all 7 evaluation domains
- **Status Tracking**: Real-time status updates for all running benchmarks

### 🎮 Interactive Controls
- **Scenario Builder**: Drag-and-drop interface for creating custom scenarios
- **Configuration Editor**: Visual configuration management with validation
- **Agent Management**: Interface for managing different agent frameworks
- **Benchmark Runner**: Start, stop, and monitor benchmark execution

### 📈 Visualization & Analysis
- **Metrics Visualization**: Interactive charts for cognitive, business, and technical metrics
- **Comparative Analysis**: Side-by-side comparison of agent performance
- **Historical Trends**: Track performance over time
- **Export Capabilities**: Generate reports in multiple formats

### 🔧 Development Tools
- **Developer Mode**: Advanced debugging and development features
- **API Explorer**: Interactive API documentation and testing
- **Performance Profiling**: Built-in performance monitoring tools
- **Error Tracking**: Comprehensive error reporting and debugging

## 🏗️ Architecture

### Component Structure
```
src/
├── components/           # Reusable UI components
│   ├── charts/          # Data visualization components
│   ├── observability/   # Monitoring and debugging components
│   ├── ui/              # Base UI components
│   └── widgets/         # Specialized UI widgets
├── pages/               # Main application pages
├── services/            # API services and utilities
├── hooks/               # Custom React hooks
├── contexts/            # React contexts for state management
├── utils/               # Utility functions
└── types/               # TypeScript type definitions
```

### Key Components

#### Dashboard (`src/pages/Dashboard/`)
- **Overview**: High-level summary of all benchmarks
- **Quick Actions**: Rapid access to common operations
- **System Status**: Health monitoring of all system components

#### Benchmark Runner (`src/pages/BenchmarkRunner/`)
- **Execution Control**: Start, stop, pause benchmarks
- **Real-time Logs**: Live streaming of execution logs
- **Progress Tracking**: Visual progress indicators

#### Scenario Builder (`src/pages/ScenarioBuilder/`)
- **Visual Editor**: Drag-and-drop scenario creation
- **Validation**: Real-time validation of scenario configurations
- **Templates**: Pre-built scenario templates

#### Configuration Editor (`src/pages/ConfigurationEditor/`)
- **Form Management**: Dynamic form generation for configurations
- **Schema Validation**: JSON Schema-based validation
- **Version Control**: Track configuration changes

#### Metrics Visualization (`src/pages/MetricsVisualization/`)
- **Chart Components**: Interactive charts using Recharts
- **Data Processing**: Client-side data processing and aggregation
- **Export Tools**: Generate reports and export data

## 🚀 Getting Started

### Prerequisites
- Node.js 16+
- npm or yarn
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to configure your API endpoint and other settings.

### Development

1. **Start the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

2. **Open your browser:**
   Navigate to `http://localhost:5173` (or the URL shown in the terminal)

### Production Build

1. **Build the application:**
   ```bash
   npm run build
   # or
   yarn build
   ```

2. **Preview the build:**
   ```bash
   npm run preview
   # or
   yarn preview
   ```

## 🔧 Configuration

### Environment Variables
- `VITE_API_URL`: Backend API endpoint
- `VITE_WS_URL`: WebSocket endpoint for real-time updates
- `VITE_APP_ENV`: Application environment (development, production)
- `VITE_ENABLE_DEV_TOOLS`: Enable development tools

### API Integration
The frontend integrates with the FBA-Bench backend through:
- **REST API**: For configuration management and data retrieval
- **WebSocket**: For real-time updates and live monitoring
- **Authentication**: JWT-based authentication

## 🧪 Testing

### Running Tests
```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Test Structure
```
src/
├── __tests__/           # Test utilities and setup
├── components/
│   └── __tests__/       # Component tests
├── hooks/
│   └── __tests__/       # Hook tests
├── services/
│   └── __tests__/       # Service tests
└── utils/
    └── __tests__/       # Utility tests
```

## 🎨 Styling

### CSS Architecture
- **CSS Modules**: Component-scoped CSS for better maintainability
- **Tailwind CSS**: Utility-first CSS framework for rapid development
- **Responsive Design**: Mobile-first responsive design approach

### Theme System
- **Light/Dark Mode**: Built-in theme switching
- **Customizable Colors**: CSS variables for easy theming
- **Consistent Spacing**: Design system for consistent UI

## 📊 Performance

### Optimization Features
- **Code Splitting**: Automatic code splitting for better loading performance
- **Lazy Loading**: Components loaded on demand
- **Caching**: Efficient caching strategies
- **Bundle Analysis**: Built-in bundle analysis tools

### Performance Targets
- **First Contentful Paint**: <1.5s
- **Largest Contentful Paint**: <2.5s
- **Time to Interactive**: <3.0s
- **Cumulative Layout Shift**: <0.1

## 🔍 Debugging

### Development Tools
- **React DevTools**: Browser extension for React debugging
- **Redux DevTools**: State management debugging
- **Network Tab**: API call monitoring
- **Console Logging**: Structured logging throughout the application

### Common Issues
- **CORS Issues**: Ensure backend CORS is properly configured
- **WebSocket Connection**: Check WebSocket server status
- **API Errors**: Review network requests and responses
- **Build Errors**: Check TypeScript types and dependencies

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Standards
- **TypeScript**: Strict type checking enabled
- **ESLint**: Code linting with TypeScript support
- **Prettier**: Code formatting for consistency
- **Component Structure**: Follow established patterns

## 📚 Documentation

### Related Documentation
- **[Main README](../README.md)**: Project overview and setup
- **[API Documentation](../docs/api-reference/)**: Backend API reference
- **[Developer Guide](../docs/development/)**: Development guidelines
- **[Configuration Guide](../docs/configuration/)**: Configuration management

### Component Documentation
- **Storybook**: Interactive component documentation (if available)
- **JSDoc**: Inline code documentation
- **Type Definitions**: TypeScript type documentation

## 🚀 Deployment

### Production Deployment
1. **Build the application:**
   ```bash
   npm run build
   ```

2. **Serve the build:**
   ```bash
   npm run preview
   ```

3. **Deploy to static hosting:**
   - Copy `dist/` folder to your web server
   - Configure routing for single-page application
   - Set up proper caching headers

### Docker Deployment
```dockerfile
FROM node:16-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 📞 Support

### Getting Help
- **Documentation**: Comprehensive guides and API references
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions and ask questions

### Reporting Issues
When reporting issues, please include:
- Browser and version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Console errors (if any)

---

**FBA-Bench Frontend** - Providing an intuitive interface for tier-1 LLM agent benchmarking and evaluation.
