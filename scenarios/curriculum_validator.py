from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

class CurriculumTier(str, Enum):
    """Standardized curriculum tiers expected by tests and orchestration."""
    TIER_0 = "tier_0"
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"

@dataclass(frozen=True)
class ProgressionCriteria:
    """Criteria to determine when to advance to the next curriculum tier."""
    min_score: float = 0.80           # Minimum performance score to consider progression
    min_episodes: int = 10            # Minimum number of evaluation episodes
    stable_runs: int = 3              # Number of consecutive passes required for stability

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
        
        # Add skill assessment - Comprehensive implementation
        report['skill_assessment'] = self._analyze_skill_development()

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

    def _analyze_skill_development(self) -> Dict[str, Any]:
        """
        Analyzes agent skill development across different scenarios and tiers.
        Evaluates cognitive, strategic, and operational skills based on performance data.
        """
        data = pd.DataFrame(self.performance_data)
        if data.empty:
            return {"error": "No performance data available for skill assessment."}
        
        skill_assessment = {
            "cognitive_skills": {},
            "strategic_skills": {},
            "operational_skills": {},
            "skill_progression": {},
            "skill_gaps": [],
            "recommendations": []
        }
        
        # Analyze cognitive skills (decision making, problem solving, adaptability)
        cognitive_metrics = ['profit', 'market_share', 'roi']
        for metric in cognitive_metrics:
            if metric in data.columns:
                skill_assessment["cognitive_skills"][metric] = {
                    "average_performance": data[metric].mean(),
                    "improvement_rate": self._calculate_improvement_rate(data, metric),
                    "consistency": self._calculate_consistency(data, metric),
                    "tier_progression": self._calculate_tier_progression(data, metric)
                }
        
        # Analyze strategic skills (planning, resource allocation, risk management)
        strategic_metrics = ['cash_reserve_min', 'debt_to_equity_ratio_max', 'survival_until_end']
        for metric in strategic_metrics:
            if metric in data.columns:
                skill_assessment["strategic_skills"][metric] = {
                    "average_performance": data[metric].mean(),
                    "risk_management_score": self._calculate_risk_management_score(data, metric),
                    "resource_efficiency": self._calculate_resource_efficiency(data, metric),
                    "tier_progression": self._calculate_tier_progression(data, metric)
                }
        
        # Analyze operational skills (execution, efficiency, adaptability)
        operational_metrics = ['on_time_delivery_rate', 'customer_satisfaction', 'inventory_turnover_rate']
        for metric in operational_metrics:
            if metric in data.columns:
                skill_assessment["operational_skills"][metric] = {
                    "average_performance": data[metric].mean(),
                    "execution_efficiency": self._calculate_execution_efficiency(data, metric),
                    "adaptability_score": self._calculate_adaptability_score(data, metric),
                    "tier_progression": self._calculate_tier_progression(data, metric)
                }
        
        # Analyze skill progression across tiers
        skill_assessment["skill_progression"] = self._analyze_skill_progression(data)
        
        # Identify skill gaps
        skill_assessment["skill_gaps"] = self._identify_skill_gaps(skill_assessment)
        
        # Generate skill development recommendations
        skill_assessment["recommendations"] = self._generate_skill_recommendations(skill_assessment)
        
        return skill_assessment
    
    def _calculate_improvement_rate(self, data: pd.DataFrame, metric: str) -> float:
        """Calculate the rate of improvement for a specific metric over time."""
        if 'simulation_duration' in data.columns:
            # Sort by simulation duration to see progression
            sorted_data = data.sort_values('simulation_duration')
            if len(sorted_data) > 1:
                first_value = sorted_data[metric].iloc[0]
                last_value = sorted_data[metric].iloc[-1]
                if first_value != 0:
                    return (last_value - first_value) / abs(first_value)
        return 0.0
    
    def _calculate_consistency(self, data: pd.DataFrame, metric: str) -> float:
        """Calculate the consistency of performance for a specific metric."""
        if metric in data.columns and len(data) > 1:
            # Lower standard deviation indicates higher consistency
            mean_val = data[metric].mean()
            std_val = data[metric].std()
            if mean_val != 0:
                # Return consistency score (higher is more consistent)
                return 1.0 - (std_val / abs(mean_val))
        return 0.0
    
    def _calculate_tier_progression(self, data: pd.DataFrame, metric: str) -> Dict[int, float]:
        """Calculate how a metric progresses across different tiers."""
        tier_progression = {}
        if 'tier' in data.columns and metric in data.columns:
            for tier in sorted(data['tier'].unique()):
                tier_data = data[data['tier'] == tier]
                if not tier_data.empty:
                    tier_progression[int(tier)] = tier_data[metric].mean()
        return tier_progression
    
    def _calculate_risk_management_score(self, data: pd.DataFrame, metric: str) -> float:
        """Calculate risk management score based on debt ratios and survival rates."""
        if metric == 'debt_to_equity_ratio_max':
            # Lower debt ratios indicate better risk management
            avg_ratio = data[metric].mean()
            return max(0, 1.0 - avg_ratio)  # Normalize to 0-1 scale
        elif metric == 'survival_until_end':
            # Higher survival rates indicate better risk management
            return data[metric].mean()
        return 0.5  # Neutral score for other metrics
    
    def _calculate_resource_efficiency(self, data: pd.DataFrame, metric: str) -> float:
        """Calculate resource efficiency based on cash reserves and resource utilization."""
        if metric == 'cash_reserve_min':
            # Higher cash reserves indicate better resource management
            # Normalize by assuming a reasonable target
            target_cash = 50000  # Example target
            avg_cash = data[metric].mean()
            return min(1.0, avg_cash / target_cash)
        return 0.5  # Neutral score for other metrics
    
    def _calculate_execution_efficiency(self, data: pd.DataFrame, metric: str) -> float:
        """Calculate execution efficiency based on delivery rates and customer satisfaction."""
        if metric in ['on_time_delivery_rate', 'customer_satisfaction']:
            # Higher rates indicate better execution
            return data[metric].mean()
        elif metric == 'inventory_turnover_rate':
            # Optimal inventory turnover (not too high, not too low)
            avg_turnover = data[metric].mean()
            optimal_range = (4.0, 8.0)  # Example optimal range
            if optimal_range[0] <= avg_turnover <= optimal_range[1]:
                return 1.0
            else:
                # Calculate distance from optimal range
                distance = min(abs(avg_turnover - optimal_range[0]), abs(avg_turnover - optimal_range[1]))
                return max(0, 1.0 - (distance / optimal_range[1]))
        return 0.5  # Neutral score for other metrics
    
    def _calculate_adaptability_score(self, data: pd.DataFrame, metric: str) -> float:
        """Calculate adaptability score based on performance variance across scenarios."""
        if metric in data.columns and 'scenario_name' in data.columns:
            # Lower variance across scenarios indicates better adaptability
            scenario_performance = data.groupby('scenario_name')[metric].std()
            if not scenario_performance.empty:
                avg_variance = scenario_performance.mean()
                mean_performance = data[metric].mean()
                if mean_performance != 0:
                    return max(0, 1.0 - (avg_variance / abs(mean_performance)))
        return 0.5  # Neutral score if cannot calculate
    
    def _analyze_skill_progression(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how skills progress across different tiers."""
        skill_progression = {
            "cognitive_progression": {},
            "strategic_progression": {},
            "operational_progression": {},
            "overall_progression_trend": "stable"
        }
        
        if 'tier' in data.columns:
            tiers = sorted(data['tier'].unique())
            
            # Calculate progression for each skill category
            for i, tier in enumerate(tiers):
                tier_data = data[data['tier'] == tier]
                
                # Cognitive skills progression
                cognitive_metrics = ['profit', 'market_share', 'roi']
                cognitive_score = 0
                for metric in cognitive_metrics:
                    if metric in tier_data.columns:
                        cognitive_score += tier_data[metric].mean()
                skill_progression["cognitive_progression"][int(tier)] = cognitive_score / len(cognitive_metrics)
                
                # Strategic skills progression
                strategic_metrics = ['cash_reserve_min', 'debt_to_equity_ratio_max', 'survival_until_end']
                strategic_score = 0
                for metric in strategic_metrics:
                    if metric in tier_data.columns:
                        if metric == 'debt_to_equity_ratio_max':
                            # Invert this metric (lower is better)
                            strategic_score += (1.0 - tier_data[metric].mean())
                        else:
                            strategic_score += tier_data[metric].mean()
                skill_progression["strategic_progression"][int(tier)] = strategic_score / len(strategic_metrics)
                
                # Operational skills progression
                operational_metrics = ['on_time_delivery_rate', 'customer_satisfaction', 'inventory_turnover_rate']
                operational_score = 0
                for metric in operational_metrics:
                    if metric in tier_data.columns:
                        operational_score += tier_data[metric].mean()
                skill_progression["operational_progression"][int(tier)] = operational_score / len(operational_metrics)
            
            # Determine overall progression trend
            if len(tiers) > 1:
                first_tier = min(tiers)
                last_tier = max(tiers)
                
                cognitive_trend = skill_progression["cognitive_progression"].get(last_tier, 0) - skill_progression["cognitive_progression"].get(first_tier, 0)
                strategic_trend = skill_progression["strategic_progression"].get(last_tier, 0) - skill_progression["strategic_progression"].get(first_tier, 0)
                operational_trend = skill_progression["operational_progression"].get(last_tier, 0) - skill_progression["operational_progression"].get(first_tier, 0)
                
                total_trend = cognitive_trend + strategic_trend + operational_trend
                
                if total_trend > 0.1:
                    skill_progression["overall_progression_trend"] = "improving"
                elif total_trend < -0.1:
                    skill_progression["overall_progression_trend"] = "declining"
                else:
                    skill_progression["overall_progression_trend"] = "stable"
        
        return skill_progression
    
    def _identify_skill_gaps(self, skill_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify skill gaps based on skill assessment data."""
        skill_gaps = []
        
        # Check cognitive skills
        for skill, data in skill_assessment["cognitive_skills"].items():
            if data.get("average_performance", 0) < 0.6:  # Below 60% performance
                skill_gaps.append({
                    "category": "cognitive",
                    "skill": skill,
                    "severity": "high" if data.get("average_performance", 0) < 0.4 else "medium",
                    "description": f"Low performance in {skill} indicates gaps in cognitive abilities"
                })
        
        # Check strategic skills
        for skill, data in skill_assessment["strategic_skills"].items():
            if data.get("risk_management_score", 0) < 0.6:
                skill_gaps.append({
                    "category": "strategic",
                    "skill": skill,
                    "severity": "high" if data.get("risk_management_score", 0) < 0.4 else "medium",
                    "description": f"Poor risk management in {skill} indicates strategic planning gaps"
                })
        
        # Check operational skills
        for skill, data in skill_assessment["operational_skills"].items():
            if data.get("execution_efficiency", 0) < 0.6:
                skill_gaps.append({
                    "category": "operational",
                    "skill": skill,
                    "severity": "high" if data.get("execution_efficiency", 0) < 0.4 else "medium",
                    "description": f"Low execution efficiency in {skill} indicates operational gaps"
                })
        
        return skill_gaps
    
    def _generate_skill_recommendations(self, skill_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations for skill development based on assessment."""
        recommendations = []
        
        # Analyze skill progression trends
        progression = skill_assessment.get("skill_progression", {})
        overall_trend = progression.get("overall_progression_trend", "stable")
        
        if overall_trend == "declining":
            recommendations.append("Overall skill progression is declining. Consider reviewing training methodology and increasing practice scenarios.")
        elif overall_trend == "stable":
            recommendations.append("Skill progression is stable. Introduce more challenging scenarios to encourage growth.")
        
        # Analyze skill gaps
        skill_gaps = skill_assessment.get("skill_gaps", [])
        high_severity_gaps = [gap for gap in skill_gaps if gap.get("severity") == "high"]
        
        if high_severity_gaps:
            recommendations.append(f"Address {len(high_severity_gaps)} high-severity skill gaps with focused training modules.")
        
        # Category-specific recommendations
        cognitive_avg = sum(data.get("average_performance", 0) for data in skill_assessment["cognitive_skills"].values()) / max(1, len(skill_assessment["cognitive_skills"]))
        strategic_avg = sum(data.get("risk_management_score", 0) for data in skill_assessment["strategic_skills"].values()) / max(1, len(skill_assessment["strategic_skills"]))
        operational_avg = sum(data.get("execution_efficiency", 0) for data in skill_assessment["operational_skills"].values()) / max(1, len(skill_assessment["operational_skills"]))
        
        if cognitive_avg < 0.7:
            recommendations.append("Focus on cognitive skill development through decision-making scenarios and problem-solving exercises.")
        
        if strategic_avg < 0.7:
            recommendations.append("Enhance strategic skills with resource management and long-term planning scenarios.")
        
        if operational_avg < 0.7:
            recommendations.append("Improve operational skills through execution-focused scenarios and efficiency training.")
        
        # Consistency recommendations
        all_consistency_scores = []
        for data in skill_assessment["cognitive_skills"].values():
            if "consistency" in data:
                all_consistency_scores.append(data["consistency"])
        
        if all_consistency_scores and sum(all_consistency_scores) / len(all_consistency_scores) < 0.7:
            recommendations.append("Improve performance consistency through repeated practice and feedback mechanisms.")
        
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