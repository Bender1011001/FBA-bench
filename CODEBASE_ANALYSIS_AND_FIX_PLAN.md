# Codebase Analysis and Fix Plan

## Overview
This document provides a comprehensive analysis of missing and simulated components in the FBA-Bench codebase and outlines a plan to implement them.

## Missing Code Components

### 1. Agent Registry System
**File**: `benchmarking/agents/registry.py` (MISSING)
**Issue**: Referenced in `benchmarking/core/engine.py` but doesn't exist
**Impact**: Critical - prevents agent management in the benchmarking system
**Fix**: Create a new file with AgentRegistry class similar to MetricRegistry and ScenarioRegistry

#### Implementation Details:
- Create `AgentRegistration` dataclass for agent metadata
- Create `AgentRegistry` class with methods for:
  - Registering agents
  - Discovering agents by category/tag
  - Creating agent instances
  - Enabling/disabling agents
- Define global `agent_registry` variable
- Add support for agent categories and tags
- Include validation for agent configurations

### 2. Configuration Schema Classes
**File**: `benchmarking/config/schema.py` (INCOMPLETE)
**Issue**: Several schema classes referenced but not implemented
**Impact**: High - prevents proper configuration validation

#### 2.1 BenchmarkConfigurationSchema
**Issue**: Referenced but not implemented
**Fix**: Implement comprehensive validation for benchmark configurations

#### 2.2 ScenarioConfigurationSchema
**Issue**: Referenced but not implemented
**Fix**: Implement validation for scenario-specific configurations

#### 2.3 AgentConfigurationSchema
**Issue**: Referenced but not implemented
**Fix**: Implement validation for agent-specific configurations

#### Implementation Details:
- Define proper JSON schemas for each configuration type
- Add validation methods with detailed error reporting
- Support nested configuration structures
- Include default values and type checking

### 3. Global Registry Variables
**Issue**: Global registry variables referenced but not properly defined/exported
**Impact**: High - causes import errors and runtime failures

#### 3.1 agent_registry
**File**: `benchmarking/agents/registry.py` (MISSING)
**Issue**: Referenced in `benchmarking/core/engine.py` but not defined
**Fix**: Define global `agent_registry` variable in the new registry file

#### 3.2 scenario_registry
**File**: `benchmarking/scenarios/registry.py` (EXISTS)
**Issue**: May not be properly exported
**Fix**: Ensure proper export of global `scenario_registry` variable

#### 3.3 metrics_registry
**File**: `benchmarking/metrics/registry.py` (EXISTS)
**Issue**: May not be properly exported
**Fix**: Ensure proper export of global `metrics_registry` variable

#### Implementation Details:
- Check if existing registries are properly exported
- Add proper imports in files that use these registries
- Ensure consistent naming and access patterns

## Simulated/Mocked Components

### 1. External Dependencies
**Issue**: Several external modules are imported with try/except but don't exist
**Impact**: Medium - system falls back to mock implementations

#### 1.1 agent_runners Module
**Files Referenced**:
- `agent_runners/runner_factory.py` (MISSING)
- `agent_runners/base_runner.py` (MISSING)

**Usage**: Referenced in:
- `benchmarking/integration/manager.py`
- `benchmarking/integration/agent_adapter.py`

**Fix**: Create minimal mock implementations:
- `AgentRunner` base class
- `RunnerFactory` class
- `SimulationState` class
- `ToolCall` class

#### 1.2 metrics Module
**Files Referenced**:
- `metrics/metric_suite.py` (EXISTS but may be incomplete)
- `metrics/cognitive_metrics.py` (EXISTS)

**Usage**: Referenced in:
- `benchmarking/integration/manager.py`

**Fix**: Ensure existing metrics module has required classes:
- `MetricSuite` class
- `CognitiveMetrics` class

#### 1.3 infrastructure Module
**Files Referenced**:
- `infrastructure/deployment.py` (MISSING)

**Usage**: Referenced in:
- `benchmarking/integration/manager.py`

**Fix**: Create minimal mock implementation:
- `DeploymentManager` class

#### 1.4 models Module
**Files Referenced**:
- `models/product.py` (EXISTS)

**Usage**: Referenced in:
- `benchmarking/integration/agent_adapter.py`

**Fix**: Ensure existing models module has required classes:
- `Product` class

#### 1.5 memory_experiments Module
**Files Referenced**:
- Multiple files in `memory_experiments/` directory (EXIST)

**Usage**: Referenced in:
- `benchmarking/integration/manager.py`

**Fix**: Ensure existing memory_experiments module has required integration points

### 2. Integration Components
**Issue**: Integration components depend on external modules that may not exist
**Impact**: Medium - integration features may not work properly

#### 2.1 Agent Runners Integration
**Location**: `benchmarking/integration/manager.py`
**Issue**: Depends on non-existent agent_runners module
**Fix**: Create mock implementations or update integration logic

#### 2.2 Legacy Metrics Integration
**Location**: `benchmarking/integration/manager.py`
**Issue**: Depends on incomplete metrics module
**Fix**: Ensure metrics module has all required classes

#### 2.3 Infrastructure Integration
**Location**: `benchmarking/integration/manager.py`
**Issue**: Depends on non-existent infrastructure module
**Fix**: Create mock implementations or update integration logic

#### 2.4 Memory Systems Integration
**Location**: `benchmarking/integration/manager.py`
**Issue**: Integration logic may be incomplete
**Fix**: Review and complete integration implementation

## Implementation Priority

### Priority 1: Core Infrastructure (High Impact)
1. Create Agent Registry System
2. Implement Configuration Schema Classes
3. Define Global Registry Variables

### Priority 2: Integration Components (Medium Impact)
4. Create Mock Implementations for External Dependencies
5. Update Integration Logic

### Priority 3: Testing and Validation (Low Impact)
6. Update Tests
7. Add Tests for New Components
8. Ensure All Tests Pass

## Implementation Plan Details

### Phase 1: Agent Registry System
1. Create `benchmarking/agents/` directory
2. Create `__init__.py` in the new directory
3. Create `registry.py` with:
   - `AgentRegistration` dataclass
   - `AgentRegistry` class
   - Global `agent_registry` variable
4. Add imports in `benchmarking/core/engine.py`

### Phase 2: Configuration Schema Classes
1. Extend `benchmarking/config/schema.py` with:
   - `BenchmarkConfigurationSchema` class
   - `ScenarioConfigurationSchema` class
   - `AgentConfigurationSchema` class
2. Add validation methods
3. Update configuration manager to use new schemas

### Phase 3: Global Registry Variables
1. Check existing registry exports
2. Add missing imports where needed
3. Ensure consistent naming and access patterns

### Phase 4: Mock Implementations
1. Create minimal `agent_runners` module
2. Ensure `metrics` module has all required classes
3. Create minimal `infrastructure` module
4. Ensure `models` module has all required classes

### Phase 5: Integration Updates
1. Update integration logic to work with mock implementations
2. Test integration components
3. Fix any remaining issues

## Files to Create or Modify

### New Files:
1. `benchmarking/agents/__init__.py`
2. `benchmarking/agents/registry.py`
3. `agent_runners/__init__.py` (if needed)
4. `agent_runners/base_runner.py` (mock)
5. `agent_runners/runner_factory.py` (mock)
6. `infrastructure/deployment.py` (mock)

### Modified Files:
1. `benchmarking/config/schema.py`
2. `benchmarking/core/engine.py`
3. `benchmarking/integration/manager.py`
4. `benchmarking/integration/agent_adapter.py`
5. `benchmarking/scenarios/registry.py` (if needed)
6. `benchmarking/metrics/registry.py` (if needed)

## Testing Considerations

1. Update existing tests to work with new implementations
2. Add tests for new components
3. Ensure all tests pass
4. Test integration components
5. Test configuration validation

## Risk Assessment

### High Risk:
- Agent Registry System (critical for system functionality)
- Configuration Schema Classes (affects system validation)

### Medium Risk:
- Global Registry Variables (may cause import errors)
- Mock Implementations (may affect integration)

### Low Risk:
- Testing and Validation (quality assurance)

## Success Criteria

1. All missing components are implemented
2. All simulated components have proper mock implementations
3. All imports resolve correctly
4. Configuration validation works properly
5. Integration components function correctly
6. All tests pass
7. System runs without errors

## Notes for Implementation

- Keep implementations minimal but functional
- Follow existing code patterns and conventions
- Add appropriate logging and error handling
- Ensure proper documentation for new components
- Maintain backward compatibility where possible
- Test each component individually before integration