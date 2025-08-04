"""
Benchmark configuration management.

This module provides configuration classes for defining benchmark parameters,
agent configurations, and execution settings.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for an agent in a benchmark."""
    
    agent_id: str
    framework: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create AgentConfig from dictionary."""
        return cls(
            agent_id=data['agent_id'],
            framework=data['framework'],
            config=data.get('config', {}),
            enabled=data.get('enabled', True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AgentConfig to dictionary."""
        return {
            'agent_id': self.agent_id,
            'framework': self.framework,
            'config': self.config,
            'enabled': self.enabled
        }


@dataclass
class ScenarioConfig:
    """Configuration for a benchmark scenario."""
    
    name: str
    description: str
    duration_ticks: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioConfig':
        """Create ScenarioConfig from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            duration_ticks=data['duration_ticks'],
            parameters=data.get('parameters', {}),
            enabled=data.get('enabled', True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ScenarioConfig to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'duration_ticks': self.duration_ticks,
            'parameters': self.parameters,
            'enabled': self.enabled
        }


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    
    enabled_metrics: List[str] = field(default_factory=lambda: [
        'cognitive', 'business', 'technical', 'ethical'
    ])
    collection_interval: int = 10  # ticks
    statistical_significance_threshold: float = 0.95
    confidence_interval: float = 0.95
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsConfig':
        """Create MetricsConfig from dictionary."""
        return cls(
            enabled_metrics=data.get('enabled_metrics', cls.enabled_metrics),
            collection_interval=data.get('collection_interval', 10),
            statistical_significance_threshold=data.get('statistical_significance_threshold', 0.95),
            confidence_interval=data.get('confidence_interval', 0.95)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MetricsConfig to dictionary."""
        return {
            'enabled_metrics': self.enabled_metrics,
            'collection_interval': self.collection_interval,
            'statistical_significance_threshold': self.statistical_significance_threshold,
            'confidence_interval': self.confidence_interval
        }


@dataclass
class ExecutionConfig:
    """Configuration for benchmark execution."""
    
    num_runs: int = 3
    parallel_execution: bool = True
    max_workers: int = 4
    random_seed: Optional[int] = None
    timeout_seconds: int = 3600  # 1 hour
    output_dir: str = "benchmark_results"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionConfig':
        """Create ExecutionConfig from dictionary."""
        return cls(
            num_runs=data.get('num_runs', 3),
            parallel_execution=data.get('parallel_execution', True),
            max_workers=data.get('max_workers', 4),
            random_seed=data.get('random_seed'),
            timeout_seconds=data.get('timeout_seconds', 3600),
            output_dir=data.get('output_dir', 'benchmark_results')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ExecutionConfig to dictionary."""
        return {
            'num_runs': self.num_runs,
            'parallel_execution': self.parallel_execution,
            'max_workers': self.max_workers,
            'random_seed': self.random_seed,
            'timeout_seconds': self.timeout_seconds,
            'output_dir': self.output_dir
        }


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    
    name: str
    description: str
    agents: List[AgentConfig] = field(default_factory=list)
    scenarios: List[ScenarioConfig] = field(default_factory=list)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    environment: str = "development"  # development, testing, production
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'BenchmarkConfig':
        """Load BenchmarkConfig from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkConfig':
        """Create BenchmarkConfig from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            agents=[AgentConfig.from_dict(agent_data) for agent_data in data.get('agents', [])],
            scenarios=[ScenarioConfig.from_dict(scenario_data) for scenario_data in data.get('scenarios', [])],
            metrics=MetricsConfig.from_dict(data.get('metrics', {})),
            execution=ExecutionConfig.from_dict(data.get('execution', {})),
            environment=data.get('environment', 'development')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert BenchmarkConfig to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'agents': [agent.to_dict() for agent in self.agents],
            'scenarios': [scenario.to_dict() for scenario in self.scenarios],
            'metrics': self.metrics.to_dict(),
            'execution': self.execution.to_dict(),
            'environment': self.environment
        }
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save BenchmarkConfig to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate the configuration and return list of errors."""
        errors = []
        
        if not self.name:
            errors.append("Benchmark name is required")
        
        if not self.description:
            errors.append("Benchmark description is required")
        
        if not self.agents:
            errors.append("At least one agent must be configured")
        
        if not self.scenarios:
            errors.append("At least one scenario must be configured")
        
        # Validate agent configurations
        agent_ids = set()
        for agent in self.agents:
            if not agent.agent_id:
                errors.append("Agent ID is required for all agents")
            elif agent.agent_id in agent_ids:
                errors.append(f"Duplicate agent ID: {agent.agent_id}")
            else:
                agent_ids.add(agent.agent_id)
            
            if not agent.framework:
                errors.append(f"Framework is required for agent {agent.agent_id}")
        
        # Validate scenario configurations
        scenario_names = set()
        for scenario in self.scenarios:
            if not scenario.name:
                errors.append("Scenario name is required for all scenarios")
            elif scenario.name in scenario_names:
                errors.append(f"Duplicate scenario name: {scenario.name}")
            else:
                scenario_names.add(scenario.name)
            
            if scenario.duration_ticks <= 0:
                errors.append(f"Duration ticks must be positive for scenario {scenario.name}")
        
        # Validate execution configuration
        if self.execution.num_runs <= 0:
            errors.append("Number of runs must be positive")
        
        if self.execution.max_workers <= 0:
            errors.append("Max workers must be positive")
        
        if self.execution.timeout_seconds <= 0:
            errors.append("Timeout seconds must be positive")
        
        return errors