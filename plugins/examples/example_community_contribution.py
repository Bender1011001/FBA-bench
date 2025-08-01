"""
Community Contribution Example for FBA-Bench

This example demonstrates the complete workflow for contributing to the FBA-Bench
community, including plugin development, validation, submission, and quality assurance.

This example shows:
- How to develop a community plugin
- Plugin validation and testing procedures
- Submission process to the community repository
- Quality assurance and review workflows
- Best practices for community contributions

For more information, see the Community Guidelines in plugins/README.md
"""

import asyncio
import json
import logging
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import community contribution tools
from community.contribution_tools import ContributionManager, QualityAssessment
from plugins.plugin_framework import PluginManager
from plugins.scenario_plugins.base_scenario_plugin import ScenarioPlugin
from plugins.agent_plugins.base_agent_plugin import AgentPlugin

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleCommunityScenario(ScenarioPlugin):
    """Example scenario plugin for community contribution."""
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Seasonal Sales Rush Scenario",
            "description": "Simulates high-demand seasonal periods like Black Friday or holiday sales",
            "version": "1.2.0",
            "author": "Community Contributor",
            "email": "contributor@example.com",
            "category": "seasonal",
            "difficulty": "intermediate",
            "tags": ["seasonal", "high-demand", "competitive", "time-pressure"],
            "framework_compatibility": ["3.0.0+"],
            "features": ["dynamic_demand", "competitor_behavior", "time_constraints"]
        }
    
    async def initialize_scenario(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the seasonal sales rush scenario."""
        logger.info("Initializing Seasonal Sales Rush Scenario")
        
        # Extract configuration
        rush_duration = config.get("rush_duration_days", 3)
        demand_multiplier = config.get("peak_demand_multiplier", 5.0)
        competitor_aggressiveness = config.get("competitor_aggressiveness", 0.8)
        
        return {
            "scenario_type": "seasonal_rush",
            "rush_duration": rush_duration,
            "peak_demand_multiplier": demand_multiplier,
            "competitor_count": config.get("competitor_count", 15),
            "price_volatility": config.get("price_volatility", 0.6),
            "inventory_constraints": True,
            "customer_urgency": 0.9,
            "market_events": [
                {"time": 0, "type": "rush_start", "magnitude": demand_multiplier},
                {"time": rush_duration * 0.5, "type": "peak_demand", "magnitude": demand_multiplier * 1.2},
                {"time": rush_duration, "type": "rush_end", "magnitude": 0.3}
            ]
        }
    
    async def inject_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle dynamic events during the sales rush."""
        if event_type == "rush_start":
            logger.info("Sales rush has begun! Demand is spiking.")
        elif event_type == "peak_demand":
            logger.info("Peak demand reached! Maximum customer activity.")
        elif event_type == "inventory_shortage":
            logger.info("Inventory shortage detected - affecting availability.")
        elif event_type == "competitor_price_war":
            logger.info("Price war initiated by aggressive competitors.")
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate scenario configuration."""
        errors = []
        
        rush_duration = config.get("rush_duration_days")
        if rush_duration and (rush_duration < 1 or rush_duration > 14):
            errors.append("rush_duration_days must be between 1 and 14 days")
        
        demand_multiplier = config.get("peak_demand_multiplier")
        if demand_multiplier and (demand_multiplier < 1.0 or demand_multiplier > 10.0):
            errors.append("peak_demand_multiplier must be between 1.0 and 10.0")
        
        return errors


class ExampleCommunityAgent(AgentPlugin):
    """Example agent plugin for community contribution."""
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Momentum Trading Agent",
            "description": "Uses price momentum and volume indicators for trading decisions",
            "version": "1.1.0",
            "author": "Community Contributor",
            "email": "contributor@example.com",
            "agent_type": "ml_based",
            "trading_style": "momentum",
            "complexity": "intermediate",
            "tags": ["momentum", "technical-analysis", "volume-based", "adaptive"],
            "capabilities": ["momentum_analysis", "volume_tracking", "trend_following"],
            "performance_tier": "medium"
        }
    
    def __init__(self):
        super().__init__()
        self.momentum_window = 5
        self.volume_threshold = 1.5
        self.price_history = {}
        self.volume_history = {}
    
    async def initialize_agent(self, config: Dict[str, Any]) -> None:
        """Initialize the momentum trading agent."""
        logger.info("Initializing Momentum Trading Agent")
        
        self.momentum_window = config.get("momentum_window", 5)
        self.volume_threshold = config.get("volume_threshold", 1.5)
        self.risk_tolerance = config.get("risk_tolerance", 0.4)
    
    async def make_decision(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Make trading decision based on momentum analysis."""
        current_prices = observation.get("current_prices", {})
        volume_data = observation.get("volume_data", {})
        
        # Update price and volume history
        for product_id, price in current_prices.items():
            if product_id not in self.price_history:
                self.price_history[product_id] = []
            self.price_history[product_id].append(price)
            
            # Keep only recent history
            if len(self.price_history[product_id]) > self.momentum_window * 2:
                self.price_history[product_id] = self.price_history[product_id][-self.momentum_window * 2:]
        
        # Analyze momentum for best opportunity
        best_action = {"action_type": "hold", "parameters": {}, "confidence": 0.3}
        
        for product_id, prices in self.price_history.items():
            if len(prices) >= self.momentum_window:
                momentum = self._calculate_momentum(prices)
                volume_signal = self._analyze_volume(product_id, volume_data)
                
                confidence = min(abs(momentum) * volume_signal, 1.0)
                
                if momentum > 0.1 and confidence > best_action["confidence"]:
                    best_action = {
                        "action_type": "buy",
                        "parameters": {"product_id": product_id, "quantity": 1},
                        "confidence": confidence,
                        "reasoning": f"Strong upward momentum ({momentum:.3f}) with volume confirmation"
                    }
                elif momentum < -0.1 and confidence > best_action["confidence"]:
                    best_action = {
                        "action_type": "sell",
                        "parameters": {"product_id": product_id, "quantity": 1},
                        "confidence": confidence,
                        "reasoning": f"Strong downward momentum ({momentum:.3f}) with volume confirmation"
                    }
        
        return best_action
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum."""
        if len(prices) < self.momentum_window:
            return 0.0
        
        recent_avg = sum(prices[-self.momentum_window:]) / self.momentum_window
        older_avg = sum(prices[-self.momentum_window*2:-self.momentum_window]) / self.momentum_window
        
        return (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
    
    def _analyze_volume(self, product_id: str, volume_data: Dict[str, Any]) -> float:
        """Analyze volume for confirmation signal."""
        current_volume = volume_data.get(product_id, 1.0)
        avg_volume = volume_data.get(f"{product_id}_avg", 1.0)
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        return min(volume_ratio / self.volume_threshold, 2.0)
    
    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """Update strategy based on feedback."""
        reward = feedback.get("reward", 0.0)
        success = feedback.get("success", False)
        
        # Adapt momentum window based on performance
        if success and reward > 0:
            # Good performance, maintain current settings
            pass
        elif not success:
            # Poor performance, adjust momentum window
            self.momentum_window = max(3, min(10, self.momentum_window + (1 if reward < -5 else -1)))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return momentum-specific performance metrics."""
        return {
            "momentum_window": self.momentum_window,
            "volume_threshold": self.volume_threshold,
            "tracked_products": len(self.price_history),
            "avg_history_length": sum(len(h) for h in self.price_history.values()) / len(self.price_history) if self.price_history else 0
        }


class CommunityContributionExample:
    """Demonstrates the complete community contribution workflow."""
    
    def __init__(self):
        """Initialize the contribution example."""
        self.contribution_manager = ContributionManager()
        self.plugin_manager = PluginManager()
        self.temp_dir = None
        
    async def run_complete_workflow(self) -> None:
        """Run the complete community contribution workflow."""
        logger.info("=== Community Contribution Workflow Example ===\n")
        
        try:
            # Step 1: Package the plugins
            await self._step_1_package_plugins()
            
            # Step 2: Validate plugins locally
            await self._step_2_validate_plugins()
            
            # Step 3: Run quality assurance
            await self._step_3_quality_assurance()
            
            # Step 4: Performance benchmarking
            await self._step_4_performance_benchmarking()
            
            # Step 5: Submit to community
            await self._step_5_submit_to_community()
            
            # Step 6: Review process simulation
            await self._step_6_review_process()
            
            # Step 7: Publication and sharing
            await self._step_7_publication()
            
            logger.info("\n=== Community Contribution Workflow Complete ===")
            logger.info("Your plugins are now part of the FBA-Bench community!")
            
        except Exception as e:
            logger.error(f"Error in contribution workflow: {e}")
            raise
        finally:
            # Cleanup temporary files
            await self._cleanup()
    
    async def _step_1_package_plugins(self) -> None:
        """Step 1: Package plugins for submission."""
        logger.info("Step 1: Packaging Plugins for Submission")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        plugin_dir = self.temp_dir / "my_plugin_contribution"
        plugin_dir.mkdir()
        
        # Create plugin structure
        scenario_dir = plugin_dir / "scenarios"
        scenario_dir.mkdir()
        agent_dir = plugin_dir / "agents"
        agent_dir.mkdir()
        
        # Save scenario plugin
        scenario_file = scenario_dir / "seasonal_sales_rush.py"
        with open(scenario_file, 'w') as f:
            f.write(self._get_scenario_plugin_code())
        
        # Save agent plugin
        agent_file = agent_dir / "momentum_trading_agent.py"
        with open(agent_file, 'w') as f:
            f.write(self._get_agent_plugin_code())
        
        # Create plugin configuration
        config_file = plugin_dir / "plugin_config.yaml"
        with open(config_file, 'w') as f:
            f.write(self._get_plugin_config())
        
        # Create README
        readme_file = plugin_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(self._get_plugin_readme())
        
        # Create tests
        test_dir = plugin_dir / "tests"
        test_dir.mkdir()
        test_file = test_dir / "test_plugins.py"
        with open(test_file, 'w') as f:
            f.write(self._get_plugin_tests())
        
        logger.info(f"Plugins packaged in: {plugin_dir}")
        logger.info("Package contents:")
        for file_path in plugin_dir.rglob("*"):
            if file_path.is_file():
                logger.info(f"  - {file_path.relative_to(plugin_dir)}")
    
    async def _step_2_validate_plugins(self) -> None:
        """Step 2: Validate plugins locally before submission."""
        logger.info("\nStep 2: Local Plugin Validation")
        
        plugin_path = self.temp_dir / "my_plugin_contribution"
        
        # Validate plugin structure
        validation_result = await self.contribution_manager.validate_plugin_submission(
            plugin_path=str(plugin_path),
            include_performance_tests=True
        )
        
        logger.info("Validation Results:")
        logger.info(f"  Valid: {'✓' if validation_result['valid'] else '✗'}")
        logger.info(f"  Structure Score: {validation_result['structure_score']:.1f}/10")
        logger.info(f"  Code Quality Score: {validation_result['code_quality_score']:.1f}/10")
        logger.info(f"  Documentation Score: {validation_result['documentation_score']:.1f}/10")
        
        if validation_result['errors']:
            logger.info("  Errors:")
            for error in validation_result['errors']:
                logger.info(f"    - {error}")
        
        if validation_result['warnings']:
            logger.info("  Warnings:")
            for warning in validation_result['warnings']:
                logger.info(f"    - {warning}")
        
        if validation_result['suggestions']:
            logger.info("  Suggestions:")
            for suggestion in validation_result['suggestions']:
                logger.info(f"    - {suggestion}")
    
    async def _step_3_quality_assurance(self) -> None:
        """Step 3: Run comprehensive quality assurance."""
        logger.info("\nStep 3: Quality Assurance")
        
        plugin_path = self.temp_dir / "my_plugin_contribution"
        
        qa_results = await self.contribution_manager.run_quality_assurance(
            plugin_path=str(plugin_path)
        )
        
        logger.info("Quality Assurance Results:")
        logger.info(f"  Overall QA Score: {qa_results['overall_score']:.1f}/100")
        logger.info(f"  Code Quality: {qa_results['code_quality_score']:.1f}/100")
        logger.info(f"  Test Coverage: {qa_results['test_coverage']:.1f}%")
        logger.info(f"  Documentation Quality: {qa_results['documentation_score']:.1f}/100")
        logger.info(f"  Security Score: {qa_results['security_score']:.1f}/100")
        logger.info(f"  Performance Score: {qa_results['performance_score']:.1f}/100")
        
        # Detailed breakdown
        if qa_results.get('detailed_results'):
            logger.info("  Detailed Analysis:")
            for category, details in qa_results['detailed_results'].items():
                logger.info(f"    {category}: {details.get('score', 'N/A')}")
                if details.get('issues'):
                    for issue in details['issues'][:3]:  # Show top 3 issues
                        logger.info(f"      - {issue}")
    
    async def _step_4_performance_benchmarking(self) -> None:
        """Step 4: Performance benchmarking against standard scenarios."""
        logger.info("\nStep 4: Performance Benchmarking")
        
        plugin_path = self.temp_dir / "my_plugin_contribution"
        
        # Define benchmark scenarios
        benchmark_scenarios = [
            {
                "name": "Standard Market Conditions",
                "config": {
                    "market_type": "standard",
                    "volatility": 0.3,
                    "duration_days": 10
                },
                "expected_performance": {"min_score": 60, "target_score": 80}
            },
            {
                "name": "High Volatility Market",
                "config": {
                    "market_type": "volatile",
                    "volatility": 0.8,
                    "duration_days": 10
                },
                "expected_performance": {"min_score": 50, "target_score": 70}
            },
            {
                "name": "Competitive Market",
                "config": {
                    "market_type": "competitive",
                    "competitor_count": 20,
                    "duration_days": 10
                },
                "expected_performance": {"min_score": 45, "target_score": 65}
            }
        ]
        
        benchmark_results = await self.contribution_manager.benchmark_plugin_performance(
            plugin_path=str(plugin_path),
            scenarios=benchmark_scenarios
        )
        
        logger.info("Performance Benchmark Results:")
        
        for scenario_name, results in benchmark_results.items():
            logger.info(f"  {scenario_name}:")
            logger.info(f"    Average Score: {results['average_score']:.1f}")
            logger.info(f"    Standard Deviation: {results['std_dev']:.1f}")
            logger.info(f"    Best Performance: {results['best_score']:.1f}")
            logger.info(f"    Worst Performance: {results['worst_score']:.1f}")
            
            # Performance rating
            avg_score = results['average_score']
            if avg_score >= 80:
                rating = "Excellent ⭐⭐⭐⭐⭐"
            elif avg_score >= 70:
                rating = "Good ⭐⭐⭐⭐"
            elif avg_score >= 60:
                rating = "Acceptable ⭐⭐⭐"
            elif avg_score >= 50:
                rating = "Below Average ⭐⭐"
            else:
                rating = "Poor ⭐"
            
            logger.info(f"    Rating: {rating}")
    
    async def _step_5_submit_to_community(self) -> None:
        """Step 5: Submit plugins to community repository."""
        logger.info("\nStep 5: Community Submission")
        
        plugin_path = self.temp_dir / "my_plugin_contribution"
        
        # Submission metadata
        submission_metadata = {
            "category": "trading_strategies",
            "subcategory": "momentum_based",
            "tags": ["momentum", "seasonal", "intermediate", "volume-analysis"],
            "license": "MIT",
            "target_audience": ["researchers", "practitioners", "students"],
            "use_cases": [
                "Seasonal market analysis",
                "Momentum trading research",
                "Educational demonstrations"
            ],
            "dependencies": ["numpy", "pandas"],
            "compatibility": {
                "fba_bench_version": ">=3.0.0",
                "python_version": ">=3.8"
            }
        }
        
        # Submit plugin
        submission_result = await self.contribution_manager.submit_plugin(
            plugin_path=str(plugin_path),
            metadata=submission_metadata
        )
        
        logger.info("Submission Results:")
        logger.info(f"  Submission ID: {submission_result['submission_id']}")
        logger.info(f"  Status: {submission_result['status']}")
        logger.info(f"  Estimated Review Time: {submission_result['estimated_review_days']} days")
        logger.info(f"  Submission URL: {submission_result.get('submission_url', 'N/A')}")
        
        if submission_result.get('next_steps'):
            logger.info("  Next Steps:")
            for step in submission_result['next_steps']:
                logger.info(f"    - {step}")
    
    async def _step_6_review_process(self) -> None:
        """Step 6: Simulate the community review process."""
        logger.info("\nStep 6: Community Review Process (Simulation)")
        
        # Simulate review timeline
        review_stages = [
            {
                "stage": "Automated Initial Review",
                "duration_hours": 1,
                "description": "Automated validation and testing",
                "status": "completed",
                "result": "passed"
            },
            {
                "stage": "Technical Review",
                "duration_hours": 24,
                "description": "Code quality and architecture review by maintainers",
                "status": "completed",
                "result": "approved",
                "feedback": "Well-structured code with good documentation"
            },
            {
                "stage": "Performance Review",
                "duration_hours": 12,
                "description": "Performance testing across multiple scenarios",
                "status": "completed",
                "result": "approved",
                "feedback": "Meets performance requirements for intermediate complexity"
            },
            {
                "stage": "Community Feedback",
                "duration_hours": 72,
                "description": "Open feedback period from community members",
                "status": "completed",
                "result": "positive",
                "feedback": "Useful for seasonal trading research, clear documentation"
            },
            {
                "stage": "Final Approval",
                "duration_hours": 6,
                "description": "Final review and approval by core team",
                "status": "completed",
                "result": "approved"
            }
        ]
        
        logger.info("Review Process Timeline:")
        total_hours = 0
        for stage in review_stages:
            total_hours += stage["duration_hours"]
            logger.info(f"  {stage['stage']}: {stage['status']} ({stage['result']})")
            logger.info(f"    Duration: {stage['duration_hours']} hours")
            logger.info(f"    Description: {stage['description']}")
            if stage.get("feedback"):
                logger.info(f"    Feedback: {stage['feedback']}")
        
        logger.info(f"  Total Review Time: {total_hours} hours ({total_hours/24:.1f} days)")
    
    async def _step_7_publication(self) -> None:
        """Step 7: Publication and community sharing."""
        logger.info("\nStep 7: Publication and Community Sharing")
        
        # Publication details
        publication_info = {
            "publication_id": "contrib_2024_001",
            "plugin_names": ["Seasonal Sales Rush Scenario", "Momentum Trading Agent"],
            "publication_date": datetime.now().isoformat(),
            "download_url": "https://community.fba-bench.org/plugins/seasonal-momentum-pack",
            "documentation_url": "https://docs.fba-bench.org/community/seasonal-momentum-pack",
            "citation": "Community Contributor (2024). Seasonal-Momentum Trading Plugin Pack. FBA-Bench Community Repository.",
            "version": "1.0.0",
            "downloads": 0,
            "rating": None,  # Will be updated as users rate it
            "featured": False  # May become featured based on popularity
        }
        
        logger.info("Publication Details:")
        logger.info(f"  Publication ID: {publication_info['publication_id']}")
        logger.info(f"  Plugin Names: {', '.join(publication_info['plugin_names'])}")
        logger.info(f"  Download URL: {publication_info['download_url']}")
        logger.info(f"  Documentation: {publication_info['documentation_url']}")
        logger.info(f"  Citation Format: {publication_info['citation']}")
        
        # Community impact tracking
        logger.info("\nCommunity Impact Tracking:")
        logger.info("  - Plugin will be indexed for search and discovery")
        logger.info("  - Usage analytics will be collected (anonymized)")
        logger.info("  - Community ratings and feedback will be displayed")
        logger.info("  - Performance metrics will be tracked across scenarios")
        logger.info("  - Plugin may be featured if it gains popularity")
        
        # Usage examples
        logger.info("\nHow Community Members Can Use Your Plugin:")
        logger.info("  1. Download via CLI: `fba-bench install seasonal-momentum-pack`")
        logger.info("  2. Load in experiments: `--load-plugin seasonal-momentum-pack`")
        logger.info("  3. Reference in research papers using provided citation")
        logger.info("  4. Extend and build upon for new research")
        logger.info("  5. Report issues or suggest improvements via GitHub")
    
    # Helper methods for generating plugin content
    
    def _get_scenario_plugin_code(self) -> str:
        """Get the scenario plugin code as a string."""
        # In a real implementation, this would read from the actual plugin file
        return '''# Seasonal Sales Rush Scenario Plugin
# Implementation details would be here...
'''
    
    def _get_agent_plugin_code(self) -> str:
        """Get the agent plugin code as a string."""
        # In a real implementation, this would read from the actual plugin file
        return '''# Momentum Trading Agent Plugin
# Implementation details would be here...
'''
    
    def _get_plugin_config(self) -> str:
        """Get plugin configuration YAML."""
        return '''# Plugin Configuration
name: "Seasonal Momentum Trading Pack"
version: "1.0.0"
description: "A combination of seasonal scenario and momentum-based agent"
author: "Community Contributor"
license: "MIT"

plugins:
  scenarios:
    - name: "Seasonal Sales Rush"
      file: "scenarios/seasonal_sales_rush.py"
      class: "ExampleCommunityScenario"
  
  agents:
    - name: "Momentum Trading Agent"
      file: "agents/momentum_trading_agent.py"
      class: "ExampleCommunityAgent"

dependencies:
  - numpy>=1.20.0
  - pandas>=1.3.0

compatibility:
  fba_bench: ">=3.0.0"
  python: ">=3.8"
'''
    
    def _get_plugin_readme(self) -> str:
        """Get plugin README content."""
        return '''# Seasonal Momentum Trading Pack

A comprehensive plugin pack for FBA-Bench that combines seasonal market scenarios with momentum-based trading agents.

## Features

- **Seasonal Sales Rush Scenario**: Simulates high-demand periods like Black Friday
- **Momentum Trading Agent**: Uses price momentum and volume for decisions
- Full integration with FBA-Bench learning systems
- Comprehensive documentation and examples

## Installation

```bash
fba-bench install seasonal-momentum-pack
```

## Usage

```python
# Load in your experiments
python experiment_cli.py run --load-plugin seasonal-momentum-pack
```

## Citation

If you use this plugin in your research, please cite:

```
Community Contributor (2024). Seasonal-Momentum Trading Plugin Pack. 
FBA-Bench Community Repository.
```
'''
    
    def _get_plugin_tests(self) -> str:
        """Get plugin test code."""
        return '''# Plugin Test Suite
import pytest
import asyncio

# Test cases would be implemented here
def test_scenario_initialization():
    """Test scenario plugin initialization."""
    pass

def test_agent_decision_making():
    """Test agent plugin decision making."""
    pass

def test_plugin_integration():
    """Test plugin integration with FBA-Bench."""
    pass
'''
    
    async def _cleanup(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary files: {self.temp_dir}")


async def main():
    """Main function to run the community contribution example."""
    example = CommunityContributionExample()
    await example.run_complete_workflow()


if __name__ == "__main__":
    # Run the complete community contribution workflow
    asyncio.run(main())