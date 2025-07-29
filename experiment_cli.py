#!/usr/bin/env python3
"""
FBA-Bench v3 Experiment CLI - High-Throughput Research Platform

Command-line tool for defining, running, and aggregating large-scale 
automated simulation experiments with parameter sweeps.

Usage:
    python experiment_cli.py run sweep.yaml
    python experiment_cli.py run sweep.yaml --parallel 4
    python experiment_cli.py analyze results/price_strategy_test_20250728-123456/
"""

import argparse
import asyncio
import json
import os
import sys
import time
import hashlib
import itertools
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
from money import Money

# Import FBA-Bench components
try:
    from simulation_orchestrator import SimulationOrchestrator, SimulationConfig
    from event_bus import EventBus
    from services.competitor_manager import CompetitorManager
    from services.dashboard_api_service import DashboardAPIService
    from personas import IrrationalSlasher, SlowFollower, CompetitorPersona
    from events import TickEvent, SaleOccurred, CompetitorPricesUpdated
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This CLI requires FBA-Bench v3 components. Make sure you're running from the project root.")
    sys.exit(1)


class ExperimentConfig:
    """Configuration for an experiment loaded from sweep.yaml."""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.experiment_name = config_data['experiment_name']
        self.description = config_data.get('description', '')
        self.base_parameters = config_data['base_parameters']
        self.parameter_sweep = config_data['parameter_sweep']
        self.output_config = config_data.get('output', {})
        
    def generate_parameter_combinations(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Generate all parameter combinations from the sweep configuration.
        
        Returns:
            Iterator of (run_number, parameters) tuples
        """
        # Extract sweep parameters
        sweep_params = {}
        for param_name, param_values in self.parameter_sweep.items():
            if param_name == 'competitor_persona_distribution':
                # Special handling for persona distribution configs
                sweep_params[param_name] = param_values
            else:
                sweep_params[param_name] = param_values
        
        # Generate all combinations using itertools.product
        param_names = list(sweep_params.keys())
        param_value_lists = [sweep_params[name] for name in param_names]
        
        run_number = 1
        for combination in itertools.product(*param_value_lists):
            parameters = dict(zip(param_names, combination))
            
            # Merge with base parameters
            final_params = self.base_parameters.copy()
            final_params.update(parameters)
            
            yield run_number, final_params
            run_number += 1
    
    def get_total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        total = 1
        for param_values in self.parameter_sweep.values():
            total *= len(param_values)
        return total


class SimulationRunner:
    """Executes individual simulations with specified parameters."""
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config
        
    async def run_single_simulation(self, run_number: int, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single simulation with the given parameters.
        
        Args:
            run_number: Unique identifier for this run
            parameters: Complete parameter set for this simulation
            
        Returns:
            Dict containing complete simulation results and metrics
        """
        print(f"🚀 Starting simulation run {run_number}")
        print(f"   Parameters: {self._format_key_parameters(parameters)}")
        
        # Convert duration_hours to max_ticks (assuming 1 tick per second)
        duration_hours = parameters.get('duration_hours', 72)
        max_ticks = int(duration_hours * 3600 / parameters.get('tick_interval_seconds', 1.0))
        
        # Create simulation configuration
        sim_config = SimulationConfig(
            tick_interval_seconds=parameters.get('tick_interval_seconds', 1.0),
            max_ticks=max_ticks,
            time_acceleration=parameters.get('time_acceleration', 1.0),
            auto_start=False
        )
        
        # Set up simulation environment
        event_bus = EventBus()
        orchestrator = SimulationOrchestrator(sim_config)
        
        # Execute simulation with event-driven architecture
        start_time = time.time()
        
        try:
            # Start event bus first
            await event_bus.start()
            
            # Configure services with parameters
            competitor_manager = self._setup_competitor_manager(parameters, event_bus)
            dashboard_service = DashboardAPIService(event_bus)
            
            # Subscribe services to event bus
            await self._setup_event_subscriptions(event_bus, competitor_manager, dashboard_service)
            
            # Start orchestrator
            await orchestrator.start(event_bus)
            
            # Wait for simulation to complete (with timeout for CLI testing)
            max_duration = parameters.get('duration_hours', 72) * 3600
            start_time_check = time.time()
            
            while orchestrator.is_running:
                await asyncio.sleep(0.1)
                
                # Add timeout to prevent infinite loops in testing
                if time.time() - start_time_check > min(max_duration, 60):  # Max 60 seconds for CLI testing
                    print(f"    ⏰ Simulation timeout reached (60s), stopping...")
                    break
                
        except Exception as e:
            print(f"❌ Simulation run {run_number} failed: {e}")
            raise
        finally:
            # Ensure orchestrator is stopped
            if orchestrator.is_running:
                await orchestrator.stop()
            
            # Stop event bus
            await event_bus.stop()
        
        execution_time = time.time() - start_time
        print(f"✅ Simulation run {run_number} completed in {execution_time:.1f}s")
        
        # Create simplified audit record from simulation results
        return self._create_simulation_result(run_number, parameters, dashboard_service, execution_time)
    
    def _setup_competitor_manager(self, parameters: Dict[str, Any], event_bus: EventBus) -> CompetitorManager:
        """Set up CompetitorManager with experiment parameters."""
        # Extract competitor manager config
        cm_config = parameters.get('competitor_manager', {})
        
        # Handle persona distribution
        persona_dist = parameters.get('competitor_persona_distribution', {})
        if isinstance(persona_dist, dict) and 'distribution' in persona_dist:
            cm_config['persona_distribution'] = persona_dist['distribution']
        
        # Create competitor manager with event bus
        cm = CompetitorManager(cm_config)
        cm.event_bus = event_bus  # Set event bus for publishing events
        
        return cm
    
    async def _setup_event_subscriptions(self, event_bus: EventBus, 
                                       competitor_manager: CompetitorManager,
                                       dashboard_service: DashboardAPIService) -> None:
        """Subscribe all services to relevant events."""
        # Subscribe services to events they need
        # Note: This would normally be done in a proper initialization routine
        # For now, we'll set up the basic subscriptions manually
        
        from events import TickEvent, SaleOccurred, CompetitorPricesUpdated
        
        # CompetitorManager subscribes to TickEvents
        await event_bus.subscribe(TickEvent, competitor_manager._handle_tick_event)
        
        # DashboardAPIService subscribes to all events for state tracking
        # Note: Using simplified event handlers for CLI integration
        await event_bus.subscribe(TickEvent, self._dashboard_tick_handler(dashboard_service))
        await event_bus.subscribe(SaleOccurred, self._dashboard_sales_handler(dashboard_service))
        await event_bus.subscribe(CompetitorPricesUpdated, self._dashboard_competitor_handler(dashboard_service))
    
    def _create_simulation_result(self, run_number: int, parameters: Dict[str, Any],
                                dashboard_service: DashboardAPIService, execution_time: float) -> Dict[str, Any]:
        """Create simulation result record from dashboard data."""
        # Get final simulation state
        snapshot = dashboard_service.get_simulation_snapshot()
        
        # Create simplified result record
        return {
            'run_number': run_number,
            'parameters': parameters,
            'execution_time_seconds': execution_time,
            'config_hash': self._hash_dict(parameters),
            'timestamp': datetime.now().isoformat(),
            'simulation_data': {
                'final_snapshot': snapshot,
                'total_ticks': snapshot.get('current_tick', 0),
                'simulation_duration_hours': parameters.get('duration_hours', 72),
                'cost_basis': parameters.get('cost_basis', 0),
                'initial_price': parameters.get('initial_price', 0)
            },
            'metrics': {
                'total_revenue': snapshot.get('sales_summary', {}).get('total_revenue', 0),
                'total_profit': snapshot.get('financial_summary', {}).get('total_profit', 0),
                'final_price': snapshot.get('current_price', 0),
                'competitor_count': len(snapshot.get('competitors', [])),
                'sales_count': len(snapshot.get('recent_sales', []))
            }
        }
    
    def _format_key_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format key parameters for display."""
        key_params = []
        
        if 'initial_price' in parameters:
            key_params.append(f"price=${parameters['initial_price']}")
        
        if 'competitor_persona_distribution' in parameters:
            dist = parameters['competitor_persona_distribution']
            if isinstance(dist, dict) and 'name' in dist:
                key_params.append(f"market={dist['name']}")
        
        if 'market_sensitivity' in parameters:
            key_params.append(f"sensitivity={parameters['market_sensitivity']}")
        
        return ", ".join(key_params)
    
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Create deterministic hash of dictionary."""
        return hashlib.sha256(str(sorted(data.items())).encode()).hexdigest()[:16]
    
    def _dashboard_tick_handler(self, dashboard_service: DashboardAPIService):
        """Create tick event handler for dashboard service."""
        async def handler(event):
            # Update dashboard state with tick information
            dashboard_service.simulation_state['current_tick'] = event.tick_number
            dashboard_service.simulation_state['simulation_time'] = event.timestamp
            dashboard_service.simulation_state['last_update'] = datetime.now()
        return handler
    
    def _dashboard_sales_handler(self, dashboard_service: DashboardAPIService):
        """Create sales event handler for dashboard service."""
        async def handler(event):
            # Update dashboard state with sales information
            if 'recent_sales' not in dashboard_service.simulation_state:
                dashboard_service.simulation_state['recent_sales'] = []
            dashboard_service.simulation_state['recent_sales'].append({
                'timestamp': event.timestamp,
                'amount': getattr(event, 'amount', 0),
                'quantity': getattr(event, 'quantity', 1)
            })
            # Keep only recent sales (last 100)
            dashboard_service.simulation_state['recent_sales'] = dashboard_service.simulation_state['recent_sales'][-100:]
        return handler
    
    def _dashboard_competitor_handler(self, dashboard_service: DashboardAPIService):
        """Create competitor prices updated handler for dashboard service."""
        async def handler(event):
            # Update dashboard state with competitor information
            dashboard_service.simulation_state['competitors'] = event.competitors
            dashboard_service.simulation_state['competitor_update_time'] = datetime.now()
        return handler


class ExperimentManager:
    """Manages experiment execution and result aggregation."""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.experiment_config = self._load_config()
        self.runner = SimulationRunner(self.experiment_config)
        self.results_dir = self._create_results_directory()
        
    def _load_config(self) -> ExperimentConfig:
        """Load and validate experiment configuration."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            return ExperimentConfig(config_data)
        except Exception as e:
            print(f"❌ Failed to load config file {self.config_file}: {e}")
            sys.exit(1)
    
    def _create_results_directory(self) -> Path:
        """Create timestamped results directory."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(f"results/{self.experiment_config.experiment_name}_{timestamp}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment configuration
        config_path = results_dir / "experiment_config.yaml"
        with open(config_path, 'w') as f:
            with open(self.config_file, 'r') as source:
                f.write(source.read())
        
        print(f"📁 Results directory: {results_dir}")
        return results_dir
    
    async def run_experiment(self, max_runs: Optional[int] = None, parallel_workers: int = 1) -> None:
        """
        Execute the full experiment with optional parallel processing.
        
        Args:
            max_runs: Maximum number of runs to execute (for testing)
            parallel_workers: Number of parallel processes to use
        """
        print(f"🧪 Starting experiment: {self.experiment_config.experiment_name}")
        print(f"📝 Description: {self.experiment_config.description}")
        
        total_combinations = self.experiment_config.get_total_combinations()
        if max_runs:
            total_combinations = min(total_combinations, max_runs)
        
        print(f"🔢 Total parameter combinations: {total_combinations}")
        
        if parallel_workers > 1:
            print(f"⚡ Parallel processing enabled: {parallel_workers} workers")
            await self._run_experiment_parallel(max_runs, parallel_workers)
        else:
            print(f"🔄 Sequential processing")
            await self._run_experiment_sequential(max_runs)
    
    async def _run_experiment_sequential(self, max_runs: Optional[int] = None) -> None:
        """Execute experiment runs sequentially."""
        run_count = 0
        for run_number, parameters in self.experiment_config.generate_parameter_combinations():
            if max_runs and run_count >= max_runs:
                break
                
            try:
                # Execute simulation
                result = await self.runner.run_single_simulation(run_number, parameters)
                
                # Save result
                self._save_run_result(run_number, parameters, result)
                
                run_count += 1
                
            except Exception as e:
                print(f"❌ Run {run_number} failed: {e}")
                continue
        
        print(f"🎉 Experiment completed! {run_count} successful runs saved to {self.results_dir}")
    
    def _save_run_result(self, run_number: int, parameters: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Save individual run result to JSON file."""
        result_file = self.results_dir / f"run_{run_number}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"💾 Saved run {run_number} results to {result_file}")
        
        # Print key metrics for immediate feedback
        metrics = result.get('metrics', {})
        print(f"    📊 Key metrics: Revenue=${metrics.get('total_revenue', 0):.2f}, "
              f"Profit=${metrics.get('total_profit', 0):.2f}, "
              f"Final Price=${metrics.get('final_price', 0):.2f}")
    
    async def _run_experiment_parallel(self, max_runs: Optional[int] = None, parallel_workers: int = 4) -> None:
        """Execute experiment runs in parallel using multiprocessing."""
        # Collect all parameter combinations
        all_combinations = []
        for run_number, parameters in self.experiment_config.generate_parameter_combinations():
            if max_runs and len(all_combinations) >= max_runs:
                break
            all_combinations.append((run_number, parameters))
        
        # Create worker arguments
        worker_args = [
            (run_number, parameters, str(self.results_dir), self.config_file)
            for run_number, parameters in all_combinations
        ]
        
        print(f"🚀 Launching {len(worker_args)} simulations across {parallel_workers} workers...")
        
        # Execute in parallel
        successful_runs = 0
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            # Submit all jobs
            future_to_run = {
                executor.submit(_run_single_simulation_worker, args): args[0]
                for args in worker_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_run):
                run_number = future_to_run[future]
                try:
                    success = future.result()
                    if success:
                        successful_runs += 1
                        print(f"✅ Run {run_number} completed successfully")
                    else:
                        print(f"❌ Run {run_number} failed")
                except Exception as e:
                    print(f"❌ Run {run_number} crashed: {e}")
        
        print(f"🎉 Parallel experiment completed! {successful_runs}/{len(worker_args)} runs successful")


def _run_single_simulation_worker(args: tuple) -> bool:
    """
    Worker function for parallel processing of individual simulation runs.
    
    This function runs in a separate process and cannot use shared state.
    """
    run_number, parameters, results_dir, config_file = args
    
    try:
        # Import here to avoid issues with multiprocessing
        import asyncio
        import json
        from pathlib import Path
        from datetime import datetime
        
        print(f"🔄 Worker starting simulation run {run_number}")
        
        # Create a new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Load experiment config
            with open(config_file, 'r') as f:
                import yaml
                config_data = yaml.safe_load(f)
            
            exp_config = ExperimentConfig(config_data)
            runner = SimulationRunner(exp_config)
            
            # Execute the simulation
            result = loop.run_until_complete(
                runner.run_single_simulation(run_number, parameters)
            )
            
            # Save result
            result_file = Path(results_dir) / f"run_{run_number}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"💾 Worker saved run {run_number} to {result_file}")
            return True
            
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ Worker failed for run {run_number}: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FBA-Bench v3 Experiment CLI - High-Throughput Research Platform"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Execute experiment from configuration file')
    run_parser.add_argument('config_file', help='Path to sweep.yaml configuration file')
    run_parser.add_argument('--max-runs', type=int, help='Maximum number of runs to execute (for testing)')
    run_parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes (future)')
    
    # Analyze command (placeholder for future)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment results')
    analyze_parser.add_argument('results_dir', help='Path to experiment results directory')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        # Validate config file exists
        if not os.path.exists(args.config_file):
            print(f"❌ Configuration file not found: {args.config_file}")
            sys.exit(1)
        
        # Execute experiment
        manager = ExperimentManager(args.config_file)
        parallel_workers = getattr(args, 'parallel', 1)
        asyncio.run(manager.run_experiment(max_runs=args.max_runs, parallel_workers=parallel_workers))
        
    elif args.command == 'analyze':
        print(f"📊 Analysis functionality coming soon for {args.results_dir}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()