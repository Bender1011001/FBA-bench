from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

class CurriculumValidator:
    """
    Validates the FBA-Bench curriculum by benchmarking agent performance
    across different tiers and scenarios, ensuring proper difficulty progression.
    """
    def __init__(self):
        self.performance_data: List[Dict[str, Any]] = []

    def benchmark_agent_performance(self, agent_model: str, tier: int, scenario_name: str, results: Dict[str, Any]):
        """
        Records the performance of a given agent model on a specific scenario within a tier.
        'results' should contain metrics like 'profit', 'market_share', 'success_status', etc.
        """
        self.performance_data.append({
            'agent_model': agent_model,
            'tier': tier,
            'scenario_name': scenario_name,
            **results
        })
        print(f"Benchmark recorded for agent '{agent_model}' on scenario '{scenario_name}' (Tier {tier}).")

    def validate_tier_progression(self, performance_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyzes collected performance data to ensure difficulty increases properly
        from T0 to T3. Checks success rates and key metric trends across tiers.
        """
        data = pd.DataFrame(performance_data if performance_data is not None else self.performance_data)
        if data.empty:
            print("No performance data available for tier progression validation.")
            return {}

        tier_summary = data.groupby('tier').agg(
            avg_profit=('profit', 'mean'),
            avg_success_rate=('success_status', lambda x: (x == 'success').mean()),
            avg_duration=('simulation_duration', 'mean')
        ).sort_values('tier')

        validation_report = {
            "tier_progression_check": True,
            "tier_summaries": tier_summary.to_dict('index'),
            "observations": []
        }

        # Check for decreasing success rate (or other difficulty indicators)
        if len(tier_summary) > 1:
            success_rates = tier_summary['avg_success_rate'].values
            for i in range(len(success_rates) - 1):
                if success_rates[i] < success_rates[i+1]:
                    validation_report["tier_progression_check"] = False
                    validation_report["observations"].append(
                        f"Warning: Success rate for Tier {tier_summary.index[i]} ({success_rates[i]:.2f}) "
                        f"is lower than Tier {tier_summary.index[i+1]} ({success_rates[i+1]:.2f})."
                    )
                else:
                    validation_report["observations"].append(
                        f"Success rate progression OK from Tier {tier_summary.index[i]} to {tier_summary.index[i+1]}."
                    )
        
        print("\nTier Progression Validation Report:")
        print(tier_summary)
        print("\n".join(validation_report["observations"]))
        return validation_report

    def analyze_success_rates(self, tier: Optional[int] = None, scenario_type: Optional[str] = None) -> Dict[str, float]:
        """
        Measures completion rates for scenarios, optionally filtered by tier or scenario type.
        """
        data = pd.DataFrame(self.performance_data)
        if data.empty:
            print("No performance data available for success rate analysis.")
            return {}

        if tier is not None:
            data = data[data['tier'] == tier]
        if scenario_type is not None:
            # This would require 'scenario_type' to be part of the collected results
            # For simplicity, let's assume scenario_name contains type info for now or we add it to benchmark_agent_performance
            data = data[data['scenario_name'].str.contains(scenario_type, case=False, na=False)]

        if data.empty:
            print(f"No data for selected filters (Tier: {tier}, Type: {scenario_type}).")
            return {}

        success_rates = data.groupby(['tier', 'scenario_name'])['success_status'].apply(lambda x: (x == 'success').mean()).to_dict()
        print(f"\nSuccess Rates (Tier: {tier}, Type: {scenario_type}):")
        for (t, s), rate in success_rates.items():
            print(f"  Tier {t}, Scenario '{s}': {rate:.2f}")
        return success_rates

    def generate_curriculum_report(self) -> Dict[str, Any]:
        """
        Creates a comprehensive validation analysis report, combining all metrics.
        """
        print("\n--- Generating Comprehensive Curriculum Report ---")
        report = {}
        report['overall_performance_summary'] = pd.DataFrame(self.performance_data).describe().to_dict()
        report['tier_progression_validation'] = self.validate_tier_progression()
        report['success_rate_by_tier'] = {}
        for tier_val in sorted(pd.DataFrame(self.performance_data)['tier'].unique()):
            report['success_rate_by_tier'][f'Tier_{tier_val}'] = self.analyze_success_rates(tier=tier_val)
        
        # Add skill assessment - Placeholder for future implementation
        report['skill_assessment'] = "Skill assessment requires defining specific skill metrics."

        print("\n--- Curriculum Report Generated ---")
        return report

    def recommend_tier_adjustments(self, performance_gaps: Dict[str, Any]) -> List[str]:
        """
        Suggests adjustments to scenario difficulty tiers based on observed performance gaps.
        'performance_gaps' would typically come from validate_tier_progression or analyze_success_rates.
        """
        recommendations = []
        # Example logic based on success rates
        if performance_gaps and 'tier_summaries' in performance_gaps:
            tier_summaries = performance_gaps['tier_summaries']
            for tier_num, summary in tier_summaries.items():
                avg_success = summary.get('avg_success_rate')
                if avg_success is not None:
                    if tier_num == 0 and avg_success < 0.80:
                        recommendations.append(f"Tier {tier_num}: Success rate ({avg_success:.2f}) is low. Consider simplifying T0 scenarios (e.g., fewer external events, more capital).")
                    elif tier_num == 0 and avg_success > 0.95:
                        recommendations.append(f"Tier {tier_num}: Success rate ({avg_success:.2f}) is very high. Consider slightly increasing T0 complexity.")
                    elif tier_num in [1, 2] and not (0.30 <= avg_success <= 0.70):
                        recommendations.append(f"Tier {tier_num}: Success rate ({avg_success:.2f}) is outside optimal range. Adjust complexity to target 30-70%.")
                    elif tier_num == 3 and avg_success > 0.30:
                        recommendations.append(f"Tier {tier_num}: Success rate ({avg_success:.2f}) is high for expert tier. Consider increasing T3 difficulty (e.g., more adversarial agents, severe shocks).")
                    elif tier_num == 3 and avg_success < 0.10:
                        recommendations.append(f"Tier {tier_num}: Success rate ({avg_success:.2f}) is very low for expert tier. Consider slightly reducing T3 difficulty or providing more initial resources.")

        if not recommendations:
            recommendations.append("No specific tier adjustments recommended based on current data.")
        
        print("\nCurriculum Adjustment Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")
        return recommendations

# Example Usage (after scenarios and engine are ready):
# from scenarios.scenario_engine import ScenarioEngine
# from some_agent_module import BasicAgent, AdvancedAgent

# validator = CurriculumValidator()
# engine = ScenarioEngine()

# # Simulate T0 scenario
# t0_scenario_config = engine.load_scenario('scenarios/tier_0_baseline.yaml')
# # Assuming engine.run_simulation returns results with 'success_status', 'profit', 'simulation_duration'
# # t0_results = engine.run_simulation(t0_scenario_config, agent=BasicAgent())
# # validator.benchmark_agent_performance("BasicAgent", 0, "tier_0_baseline", t0_results)

# # ... more simulation runs for different agents and tiers ...

# # After all benchmarks are run:
# # validator.generate_curriculum_report()
# # validator.recommend_tier_adjustments(validator.validate_tier_progression())