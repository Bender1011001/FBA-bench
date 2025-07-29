"""
Experiment Runner

A/B testing framework for memory experiments, enabling systematic comparison
of different memory configurations and their impact on agent performance.
"""

import asyncio
import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

from .memory_config import MemoryConfig, MemoryMode
from .memory_enforcer import MemoryEnforcer
from .statistical_analyzer import StatisticalAnalyzer
from constraints.constraint_config import ConstraintConfig
from event_bus import EventBus


logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a memory experiment."""
    experiment_id: str
    name: str
    description: str
    
    # Experiment parameters
    memory_configs: List[MemoryConfig]
    baseline_config: MemoryConfig
    agent_type: str
    scenario_config: str
    
    # Statistical parameters
    sample_size_per_condition: int = 30
    confidence_level: float = 0.95
    min_effect_size: float = 0.1
    
    # Experiment controls
    randomization_seed: Optional[int] = None
    max_simulation_ticks: int = 1000
    parallel_runs: int = 1
    
    # Output settings
    output_directory: str = "memory_experiment_results"
    save_detailed_logs: bool = True
    save_memory_traces: bool = False


@dataclass
class ExperimentRun:
    """Results from a single experimental run."""
    run_id: str
    experiment_id: str
    memory_config_name: str
    agent_id: str
    
    # Performance metrics
    overall_score: float
    memory_dependent_score: float
    memory_independent_score: float
    
    # Memory usage metrics
    memory_retrievals: int
    memory_promotions: int
    reflection_count: int
    avg_memory_tokens: float
    
    # Timing metrics
    start_time: datetime
    end_time: datetime
    total_ticks: int
    
    # Additional metrics
    cognitive_metrics: Dict[str, float]
    memory_efficiency: float
    consolidation_quality: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResults:
    """Aggregated results from a memory experiment."""
    experiment_id: str
    config: ExperimentConfig
    
    # Statistical results
    statistical_significance: Dict[str, float]  # p-values for each comparison
    effect_sizes: Dict[str, float]             # Effect sizes for each comparison
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Performance comparisons
    memory_impact_score: float  # How much memory helps vs baseline
    reasoning_vs_recall: str    # "reasoning_dominant", "memory_dominant", "balanced"
    optimal_memory_mode: str
    
    # Individual run results
    individual_runs: List[ExperimentRun]
    
    # Summary statistics
    summary_stats: Dict[str, Any]
    
    # Research conclusions
    conclusions: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result_dict = asdict(self)
        result_dict['individual_runs'] = [run.to_dict() for run in self.individual_runs]
        return result_dict


class ExperimentRunner:
    """
    A/B testing framework for memory experiments.
    
    Enables systematic comparison of different memory configurations
    and their impact on agent performance with statistical validation.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Experiment tracking
        self.current_experiment: Optional[ExperimentConfig] = None
        self.active_runs: Dict[str, ExperimentRun] = {}
        self.completed_experiments: List[ExperimentResults] = []
        
        # Agent factory and metrics (to be injected)
        self.agent_factory: Optional[Callable] = None
        self.metrics_calculator: Optional[Callable] = None
        
        logger.info("ExperimentRunner initialized")
    
    def set_agent_factory(self, factory: Callable):
        """Set the agent factory for creating test agents."""
        self.agent_factory = factory
    
    def set_metrics_calculator(self, calculator: Callable):
        """Set the metrics calculator for evaluating performance."""
        self.metrics_calculator = calculator
    
    async def run_experiment(self, config: ExperimentConfig) -> ExperimentResults:
        """
        Run a complete memory experiment with statistical validation.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Comprehensive experiment results with statistical analysis
        """
        logger.info(f"Starting memory experiment: {config.experiment_id}")
        self.current_experiment = config
        
        # Set random seed for reproducibility
        if config.randomization_seed:
            random.seed(config.randomization_seed)
        
        # Create output directory
        output_dir = Path(config.output_directory) / config.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results tracking
        all_runs: List[ExperimentRun] = []
        
        try:
            # Run experiments for each memory configuration
            for memory_config in config.memory_configs:
                logger.info(f"Testing memory configuration: {memory_config.memory_mode.value}")
                
                config_runs = await self._run_configuration_trials(
                    config, memory_config, output_dir
                )
                all_runs.extend(config_runs)
            
            # Also run baseline configuration
            logger.info(f"Testing baseline configuration: {config.baseline_config.memory_mode.value}")
            baseline_runs = await self._run_configuration_trials(
                config, config.baseline_config, output_dir
            )
            all_runs.extend(baseline_runs)
            
            # Perform statistical analysis
            statistical_results = await self._analyze_results(config, all_runs)
            
            # Create comprehensive results
            experiment_results = ExperimentResults(
                experiment_id=config.experiment_id,
                config=config,
                individual_runs=all_runs,
                **statistical_results
            )
            
            # Save results
            await self._save_experiment_results(experiment_results, output_dir)
            
            self.completed_experiments.append(experiment_results)
            
            logger.info(f"Memory experiment completed: {config.experiment_id}")
            
            return experiment_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            self.current_experiment = None
    
    async def _run_configuration_trials(self, 
                                      experiment_config: ExperimentConfig,
                                      memory_config: MemoryConfig,
                                      output_dir: Path) -> List[ExperimentRun]:
        """Run multiple trials for a single memory configuration."""
        
        runs = []
        config_name = memory_config.memory_mode.value
        
        for trial in range(experiment_config.sample_size_per_condition):
            run_id = f"{experiment_config.experiment_id}_{config_name}_{trial:03d}"
            
            logger.info(f"Running trial {trial + 1}/{experiment_config.sample_size_per_condition} for {config_name}")
            
            try:
                run_result = await self._run_single_trial(
                    run_id, experiment_config, memory_config, output_dir
                )
                runs.append(run_result)
                
            except Exception as e:
                logger.error(f"Trial {run_id} failed: {e}")
                # Continue with other trials
                continue
        
        logger.info(f"Completed {len(runs)} trials for {config_name}")
        return runs
    
    async def _run_single_trial(self,
                              run_id: str,
                              experiment_config: ExperimentConfig,
                              memory_config: MemoryConfig,
                              output_dir: Path) -> ExperimentRun:
        """Run a single experimental trial."""
        
        start_time = datetime.now()
        
        # Create agent with memory configuration
        agent_id = f"memory_test_agent_{run_id}"
        
        if not self.agent_factory:
            raise ValueError("Agent factory not set. Call set_agent_factory() first.")
        
        agent = self.agent_factory(agent_id, memory_config)
        
        # Initialize memory enforcer
        memory_enforcer = MemoryEnforcer(memory_config, agent_id, self.event_bus)
        
        # Run simulation
        simulation_results = await self._run_simulation(
            agent, memory_enforcer, experiment_config
        )
        
        end_time = datetime.now()
        
        # Calculate metrics
        performance_metrics = await self._calculate_performance_metrics(
            simulation_results, memory_enforcer
        )
        
        # Create run result
        run_result = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_config.experiment_id,
            memory_config_name=memory_config.memory_mode.value,
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time,
            total_ticks=simulation_results.get('total_ticks', 0),
            **performance_metrics
        )
        
        # Save detailed logs if requested
        if experiment_config.save_detailed_logs:
            await self._save_run_details(run_result, simulation_results, output_dir)
        
        return run_result
    
    async def _run_simulation(self,
                            agent,
                            memory_enforcer: MemoryEnforcer,
                            config: ExperimentConfig) -> Dict[str, Any]:
        """Run the actual simulation for the trial."""
        
        # This is a placeholder for the actual simulation logic
        # In practice, this would integrate with FBA-Bench's simulation orchestrator
        
        results = {
            'total_ticks': config.max_simulation_ticks,
            'events_processed': random.randint(100, 500),
            'decisions_made': random.randint(50, 200),
            'simulation_seed': random.randint(1000, 9999)
        }
        
        # Simulate memory usage during the run
        for tick in range(config.max_simulation_ticks // 10):  # Sample every 10 ticks
            memory_enforcer.update_tick(tick * 10)
            
            # Simulate reflection triggers
            if tick % 24 == 0:  # Daily reflection
                await memory_enforcer.check_reflection_trigger()
        
        return results
    
    async def _calculate_performance_metrics(self,
                                           simulation_results: Dict[str, Any],
                                           memory_enforcer: MemoryEnforcer) -> Dict[str, Any]:
        """Calculate performance metrics for the trial."""
        
        if not self.metrics_calculator:
            # Placeholder metrics for testing
            memory_stats = await memory_enforcer.get_memory_statistics()
            
            return {
                'overall_score': random.uniform(50, 90),
                'memory_dependent_score': random.uniform(40, 80),
                'memory_independent_score': random.uniform(60, 95),
                'memory_retrievals': memory_stats['memory_usage']['total_retrievals'],
                'memory_promotions': memory_stats.get('reflection', {}).get('total_promotions', 0),
                'reflection_count': memory_stats.get('reflection', {}).get('total_reflections', 0),
                'avg_memory_tokens': memory_stats['memory_usage']['current_memory_tokens'],
                'cognitive_metrics': {
                    'attention_span': random.uniform(0.7, 1.0),
                    'decision_quality': random.uniform(0.6, 0.9),
                    'strategic_coherence': random.uniform(0.5, 0.95)
                },
                'memory_efficiency': random.uniform(0.4, 0.8),
                'consolidation_quality': random.uniform(0.5, 0.9)
            }
        
        # Use actual metrics calculator
        return await self.metrics_calculator(simulation_results, memory_enforcer)
    
    async def _analyze_results(self,
                             config: ExperimentConfig,
                             all_runs: List[ExperimentRun]) -> Dict[str, Any]:
        """Perform statistical analysis on experiment results."""
        
        # Group runs by memory configuration
        config_groups = {}
        for run in all_runs:
            config_name = run.memory_config_name
            if config_name not in config_groups:
                config_groups[config_name] = []
            config_groups[config_name].append(run)
        
        baseline_name = config.baseline_config.memory_mode.value
        baseline_runs = config_groups.get(baseline_name, [])
        
        # Perform statistical comparisons
        statistical_significance = {}
        effect_sizes = {}
        confidence_intervals = {}
        
        for config_name, runs in config_groups.items():
            if config_name == baseline_name:
                continue
            
            # Compare against baseline
            comparison_key = f"{config_name}_vs_{baseline_name}"
            
            baseline_scores = [run.overall_score for run in baseline_runs]
            config_scores = [run.overall_score for run in runs]
            
            if baseline_scores and config_scores:
                p_value = self.statistical_analyzer.ttest_independent(
                    baseline_scores, config_scores
                )
                effect_size = self.statistical_analyzer.cohens_d(
                    baseline_scores, config_scores
                )
                ci = self.statistical_analyzer.confidence_interval(
                    config_scores, config.confidence_level
                )
                
                statistical_significance[comparison_key] = p_value
                effect_sizes[comparison_key] = effect_size
                confidence_intervals[comparison_key] = ci
        
        # Determine memory impact and optimal configuration
        memory_impact_score = self._calculate_memory_impact(config_groups, baseline_name)
        reasoning_vs_recall = self._analyze_reasoning_vs_recall(config_groups)
        optimal_memory_mode = self._find_optimal_memory_mode(config_groups)
        
        # Generate summary statistics
        summary_stats = self._calculate_summary_statistics(config_groups)
        
        # Research conclusions
        conclusions = self._generate_conclusions(
            statistical_significance, effect_sizes, memory_impact_score
        )
        
        return {
            'statistical_significance': statistical_significance,
            'effect_sizes': effect_sizes,
            'confidence_intervals': confidence_intervals,
            'memory_impact_score': memory_impact_score,
            'reasoning_vs_recall': reasoning_vs_recall,
            'optimal_memory_mode': optimal_memory_mode,
            'summary_stats': summary_stats,
            'conclusions': conclusions
        }
    
    def _calculate_memory_impact(self, config_groups: Dict[str, List[ExperimentRun]], baseline_name: str) -> float:
        """Calculate overall impact of memory on performance."""
        baseline_runs = config_groups.get(baseline_name, [])
        if not baseline_runs:
            return 0.0
        
        baseline_avg = sum(run.overall_score for run in baseline_runs) / len(baseline_runs)
        
        memory_improvements = []
        for config_name, runs in config_groups.items():
            if config_name == baseline_name or not runs:
                continue
            
            config_avg = sum(run.overall_score for run in runs) / len(runs)
            improvement = (config_avg - baseline_avg) / baseline_avg
            memory_improvements.append(improvement)
        
        return sum(memory_improvements) / len(memory_improvements) if memory_improvements else 0.0
    
    def _analyze_reasoning_vs_recall(self, config_groups: Dict[str, List[ExperimentRun]]) -> str:
        """Analyze whether reasoning or memory recall is more important."""
        
        # Simple heuristic based on memory-dependent vs independent scores
        reasoning_scores = []
        recall_scores = []
        
        for runs in config_groups.values():
            for run in runs:
                reasoning_scores.append(run.memory_independent_score)
                recall_scores.append(run.memory_dependent_score)
        
        if not reasoning_scores or not recall_scores:
            return "insufficient_data"
        
        avg_reasoning = sum(reasoning_scores) / len(reasoning_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)
        
        difference = abs(avg_reasoning - avg_recall)
        
        if difference < 5:  # Within 5 points
            return "balanced"
        elif avg_reasoning > avg_recall:
            return "reasoning_dominant"
        else:
            return "memory_dominant"
    
    def _find_optimal_memory_mode(self, config_groups: Dict[str, List[ExperimentRun]]) -> str:
        """Find the memory configuration with best performance."""
        
        config_averages = {}
        for config_name, runs in config_groups.items():
            if runs:
                avg_score = sum(run.overall_score for run in runs) / len(runs)
                config_averages[config_name] = avg_score
        
        if not config_averages:
            return "unknown"
        
        return max(config_averages, key=config_averages.get)
    
    def _calculate_summary_statistics(self, config_groups: Dict[str, List[ExperimentRun]]) -> Dict[str, Any]:
        """Calculate summary statistics across all configurations."""
        
        all_runs = [run for runs in config_groups.values() for run in runs]
        
        if not all_runs:
            return {}
        
        return {
            'total_runs': len(all_runs),
            'configurations_tested': len(config_groups),
            'avg_overall_score': sum(run.overall_score for run in all_runs) / len(all_runs),
            'avg_memory_retrievals': sum(run.memory_retrievals for run in all_runs) / len(all_runs),
            'avg_reflection_count': sum(run.reflection_count for run in all_runs) / len(all_runs),
            'score_variance': self.statistical_analyzer.variance([run.overall_score for run in all_runs])
        }
    
    def _generate_conclusions(self,
                            statistical_significance: Dict[str, float],
                            effect_sizes: Dict[str, float],
                            memory_impact_score: float) -> Dict[str, Any]:
        """Generate research conclusions from the experiment."""
        
        significant_results = [
            comparison for comparison, p_value in statistical_significance.items()
            if p_value < 0.05
        ]
        
        large_effects = [
            comparison for comparison, effect_size in effect_sizes.items()
            if abs(effect_size) > 0.5
        ]
        
        conclusions = {
            'memory_matters': abs(memory_impact_score) > 0.05,
            'significant_differences': len(significant_results) > 0,
            'large_effect_sizes': len(large_effects) > 0,
            'memory_impact_magnitude': abs(memory_impact_score),
            'statistical_power': 'adequate' if len(significant_results) > 0 else 'insufficient',
            'key_findings': []
        }
        
        # Generate key findings
        if conclusions['memory_matters']:
            if memory_impact_score > 0:
                conclusions['key_findings'].append("Memory systems improve agent performance")
            else:
                conclusions['key_findings'].append("Memory systems may hurt agent performance")
        else:
            conclusions['key_findings'].append("Memory has minimal impact on performance (supporting VendingBench findings)")
        
        if large_effects:
            conclusions['key_findings'].append(f"Large effect sizes found in {len(large_effects)} comparisons")
        
        return conclusions
    
    async def _save_experiment_results(self,
                                     results: ExperimentResults,
                                     output_dir: Path):
        """Save comprehensive experiment results."""
        
        # Save main results as JSON
        results_file = output_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        # Save summary report
        await self._generate_summary_report(results, output_dir)
        
        logger.info(f"Experiment results saved to {output_dir}")
    
    async def _save_run_details(self,
                              run_result: ExperimentRun,
                              simulation_results: Dict[str, Any],
                              output_dir: Path):
        """Save detailed information for a single run."""
        
        run_dir = output_dir / "detailed_runs" / run_result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run metadata
        run_file = run_dir / "run_results.json"
        with open(run_file, 'w') as f:
            json.dump(run_result.to_dict(), f, indent=2, default=str)
        
        # Save simulation details
        sim_file = run_dir / "simulation_details.json"
        with open(sim_file, 'w') as f:
            json.dump(simulation_results, f, indent=2, default=str)
    
    async def _generate_summary_report(self,
                                     results: ExperimentResults,
                                     output_dir: Path):
        """Generate a human-readable summary report."""
        
        report_file = output_dir / "summary_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Memory Experiment Report: {results.experiment_id}\n\n")
            f.write(f"**Experiment Name:** {results.config.name}\n")
            f.write(f"**Description:** {results.config.description}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Runs:** {results.summary_stats.get('total_runs', 0)}\n")
            f.write(f"- **Configurations Tested:** {results.summary_stats.get('configurations_tested', 0)}\n")
            f.write(f"- **Memory Impact Score:** {results.memory_impact_score:.3f}\n")
            f.write(f"- **Reasoning vs Recall:** {results.reasoning_vs_recall}\n")
            f.write(f"- **Optimal Memory Mode:** {results.optimal_memory_mode}\n\n")
            
            f.write("## Statistical Results\n\n")
            for comparison, p_value in results.statistical_significance.items():
                effect_size = results.effect_sizes.get(comparison, 0.0)
                significance = "✓" if p_value < 0.05 else "✗"
                f.write(f"- **{comparison}:** p={p_value:.4f}, d={effect_size:.3f} {significance}\n")
            
            f.write("\n## Key Findings\n\n")
            for finding in results.conclusions.get('key_findings', []):
                f.write(f"- {finding}\n")
            
            f.write("\n## Research Implications\n\n")
            if results.conclusions.get('memory_matters'):
                f.write("This experiment provides evidence that memory systems significantly impact agent performance in FBA-Bench scenarios.\n")
            else:
                f.write("This experiment supports VendingBench's finding that memory may not be the primary bottleneck for agent performance.\n")
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get current experiment status."""
        return {
            'current_experiment': self.current_experiment.experiment_id if self.current_experiment else None,
            'active_runs': len(self.active_runs),
            'completed_experiments': len(self.completed_experiments),
            'last_experiment_results': self.completed_experiments[-1].to_dict() if self.completed_experiments else None
        }