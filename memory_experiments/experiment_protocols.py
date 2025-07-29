"""
Experiment Protocols

Standardized experiment configurations for systematic study of memory vs. reasoning
in agent performance, including protocols for testing consolidation effectiveness.
"""

import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

from .memory_config import MemoryConfig, MemoryMode, ConsolidationAlgorithm
from .experiment_runner import ExperimentConfig
from constraints.constraint_config import ConstraintConfig


@dataclass
class ExperimentProtocol:
    """Standardized protocol for memory experiments."""
    
    protocol_id: str
    name: str
    description: str
    research_questions: List[str]
    hypotheses: List[str]
    memory_configs: List[Dict[str, Any]]
    baseline_config: Dict[str, Any]
    experimental_controls: Dict[str, Any]
    expected_outcomes: List[str]
    
    def to_experiment_config(self, agent_type: str = "gpt_4o_mini", 
                           scenario_config: str = "standard") -> ExperimentConfig:
        """Convert protocol to executable experiment configuration."""
        
        # Load memory configurations
        memory_configs = []
        for config_dict in self.memory_configs:
            # Create base constraint config
            base_config = ConstraintConfig(
                max_tokens_per_action=config_dict["budget_constraints"]["max_tokens_per_action"],
                max_total_tokens=config_dict["budget_constraints"]["max_total_tokens"],
                token_cost_per_1k=config_dict["budget_constraints"]["token_cost_per_1k"],
                violation_penalty_weight=config_dict["budget_constraints"]["violation_penalty_weight"],
                grace_period_percentage=config_dict["budget_constraints"]["grace_period_percentage"],
                hard_fail_on_violation=config_dict["enforcement"]["hard_fail_on_violation"],
                inject_budget_status=config_dict["enforcement"]["inject_budget_status"],
                track_token_efficiency=config_dict["enforcement"]["track_token_efficiency"]
            )
            
            # Create memory config
            memory_settings = config_dict["memory_configuration"]
            memory_config = MemoryConfig(
                base_config=base_config,
                memory_mode=MemoryMode(memory_settings["memory_mode"]),
                short_term_capacity=memory_settings["short_term_capacity"],
                short_term_retention_days=memory_settings["short_term_retention_days"],
                long_term_capacity=memory_settings["long_term_capacity"],
                reflection_enabled=memory_settings["reflection_enabled"],
                consolidation_algorithm=ConsolidationAlgorithm(memory_settings["consolidation_algorithm"]),
                consolidation_percentage=memory_settings["consolidation_percentage"],
                enable_memory_injection=memory_settings["enable_memory_injection"],
                memory_budget_tokens=memory_settings["memory_budget_tokens"]
            )
            memory_configs.append(memory_config)
        
        # Create baseline config
        baseline_base = ConstraintConfig(
            max_tokens_per_action=self.baseline_config["budget_constraints"]["max_tokens_per_action"],
            max_total_tokens=self.baseline_config["budget_constraints"]["max_total_tokens"],
            token_cost_per_1k=self.baseline_config["budget_constraints"]["token_cost_per_1k"],
            violation_penalty_weight=self.baseline_config["budget_constraints"]["violation_penalty_weight"],
            grace_period_percentage=self.baseline_config["budget_constraints"]["grace_period_percentage"],
            hard_fail_on_violation=self.baseline_config["enforcement"]["hard_fail_on_violation"],
            inject_budget_status=self.baseline_config["enforcement"]["inject_budget_status"],
            track_token_efficiency=self.baseline_config["enforcement"]["track_token_efficiency"]
        )
        
        baseline_memory_settings = self.baseline_config["memory_configuration"]
        baseline_config = MemoryConfig(
            base_config=baseline_base,
            memory_mode=MemoryMode(baseline_memory_settings["memory_mode"]),
            short_term_capacity=baseline_memory_settings["short_term_capacity"],
            short_term_retention_days=baseline_memory_settings["short_term_retention_days"],
            long_term_capacity=baseline_memory_settings["long_term_capacity"],
            reflection_enabled=baseline_memory_settings["reflection_enabled"],
            consolidation_algorithm=ConsolidationAlgorithm(baseline_memory_settings["consolidation_algorithm"]),
            consolidation_percentage=baseline_memory_settings["consolidation_percentage"],
            enable_memory_injection=baseline_memory_settings["enable_memory_injection"],
            memory_budget_tokens=baseline_memory_settings["memory_budget_tokens"]
        )
        
        return ExperimentConfig(
            experiment_id=f"{self.protocol_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=self.name,
            description=self.description,
            memory_configs=memory_configs,
            baseline_config=baseline_config,
            agent_type=agent_type,
            scenario_config=scenario_config,
            sample_size_per_condition=self.experimental_controls.get("sample_size", 30),
            confidence_level=self.experimental_controls.get("confidence_level", 0.95),
            min_effect_size=self.experimental_controls.get("min_effect_size", 0.1),
            randomization_seed=self.experimental_controls.get("randomization_seed"),
            max_simulation_ticks=self.experimental_controls.get("max_simulation_ticks", 1000)
        )


class ExperimentProtocols:
    """Collection of standardized experiment protocols for memory research."""
    
    @staticmethod
    def get_consolidation_vs_decay_protocol() -> ExperimentProtocol:
        """
        Protocol for testing memory consolidation vs. simple time-based decay.
        
        Research Question: Does intelligent memory consolidation outperform 
        simple time-based memory decay?
        """
        
        # Reflection-enabled configuration
        reflection_config = {
            "budget_constraints": {
                "max_tokens_per_action": 16000,
                "max_total_tokens": 500000,
                "token_cost_per_1k": 0.06,
                "violation_penalty_weight": 2.0,
                "grace_period_percentage": 5.0
            },
            "enforcement": {
                "hard_fail_on_violation": True,
                "inject_budget_status": True,
                "track_token_efficiency": True
            },
            "memory_configuration": {
                "memory_mode": "reflection_enabled",
                "short_term_capacity": 100,
                "short_term_retention_days": 2,
                "long_term_capacity": 500,
                "reflection_enabled": True,
                "consolidation_algorithm": "importance_score",
                "consolidation_percentage": 0.3,
                "enable_memory_injection": True,
                "memory_budget_tokens": 2000
            }
        }
        
        # Consolidation-disabled configuration (time-based only)
        time_based_config = {
            "budget_constraints": {
                "max_tokens_per_action": 16000,
                "max_total_tokens": 500000,
                "token_cost_per_1k": 0.06,
                "violation_penalty_weight": 2.0,
                "grace_period_percentage": 5.0
            },
            "enforcement": {
                "hard_fail_on_violation": True,
                "inject_budget_status": True,
                "track_token_efficiency": True
            },
            "memory_configuration": {
                "memory_mode": "consolidation_disabled",
                "short_term_capacity": 100,
                "short_term_retention_days": 2,
                "long_term_capacity": 500,
                "reflection_enabled": False,
                "consolidation_algorithm": "random_selection",
                "consolidation_percentage": 0.3,
                "enable_memory_injection": True,
                "memory_budget_tokens": 2000
            }
        }
        
        # Baseline: Memory-free
        baseline_config = {
            "budget_constraints": {
                "max_tokens_per_action": 16000,
                "max_total_tokens": 500000,
                "token_cost_per_1k": 0.06,
                "violation_penalty_weight": 2.0,
                "grace_period_percentage": 5.0
            },
            "enforcement": {
                "hard_fail_on_violation": True,
                "inject_budget_status": True,
                "track_token_efficiency": True
            },
            "memory_configuration": {
                "memory_mode": "memory_free",
                "short_term_capacity": 0,
                "short_term_retention_days": 0,
                "long_term_capacity": 0,
                "reflection_enabled": False,
                "consolidation_algorithm": "importance_score",
                "consolidation_percentage": 0.0,
                "enable_memory_injection": False,
                "memory_budget_tokens": 0
            }
        }
        
        return ExperimentProtocol(
            protocol_id="consolidation_vs_decay",
            name="Memory Consolidation vs. Time-Based Decay",
            description="Compares intelligent memory consolidation against simple time-based promotion",
            research_questions=[
                "Does intelligent consolidation outperform time-based memory decay?",
                "What is the performance difference between reflection and random promotion?",
                "How much do memory systems contribute to overall performance?"
            ],
            hypotheses=[
                "H1: Reflection-based consolidation will outperform time-based decay",
                "H2: Both memory systems will outperform memory-free baseline",
                "H3: Consolidation quality will correlate with performance improvement"
            ],
            memory_configs=[reflection_config, time_based_config],
            baseline_config=baseline_config,
            experimental_controls={
                "sample_size": 40,
                "confidence_level": 0.95,
                "min_effect_size": 0.2,
                "max_simulation_ticks": 1200,
                "randomization_seed": 12345
            },
            expected_outcomes=[
                "Reflection-enabled mode shows highest performance",
                "Time-based mode shows moderate improvement over baseline",
                "Statistical significance in consolidation quality metrics"
            ]
        )
    
    @staticmethod
    def get_memory_window_optimization_protocol() -> ExperimentProtocol:
        """
        Protocol for finding optimal memory retention windows.
        
        Research Question: What is the optimal memory retention period 
        for FBA-Bench scenarios?
        """
        
        # Different retention windows to test
        retention_configs = []
        for days in [3, 7, 14, 30]:
            config = {
                "budget_constraints": {
                    "max_tokens_per_action": 16000,
                    "max_total_tokens": 500000,
                    "token_cost_per_1k": 0.06,
                    "violation_penalty_weight": 2.0,
                    "grace_period_percentage": 5.0
                },
                "enforcement": {
                    "hard_fail_on_violation": True,
                    "inject_budget_status": True,
                    "track_token_efficiency": True
                },
                "memory_configuration": {
                    "memory_mode": "short_term_only",
                    "short_term_capacity": 200,
                    "short_term_retention_days": days,
                    "long_term_capacity": 0,
                    "reflection_enabled": False,
                    "consolidation_algorithm": "importance_score",
                    "consolidation_percentage": 0.0,
                    "enable_memory_injection": True,
                    "memory_budget_tokens": 2000
                }
            }
            retention_configs.append(config)
        
        baseline_config = {
            "budget_constraints": {
                "max_tokens_per_action": 16000,
                "max_total_tokens": 500000,
                "token_cost_per_1k": 0.06,
                "violation_penalty_weight": 2.0,
                "grace_period_percentage": 5.0
            },
            "enforcement": {
                "hard_fail_on_violation": True,
                "inject_budget_status": True,
                "track_token_efficiency": True
            },
            "memory_configuration": {
                "memory_mode": "memory_free",
                "short_term_capacity": 0,
                "short_term_retention_days": 0,
                "long_term_capacity": 0,
                "reflection_enabled": False,
                "consolidation_algorithm": "importance_score",
                "consolidation_percentage": 0.0,
                "enable_memory_injection": False,
                "memory_budget_tokens": 0
            }
        }
        
        return ExperimentProtocol(
            protocol_id="memory_window_optimization",
            name="Optimal Memory Retention Window",
            description="Tests different memory retention periods to find optimal window",
            research_questions=[
                "What is the optimal memory retention period for FBA scenarios?",
                "Does performance plateau after a certain retention period?",
                "Is there a point of diminishing returns for longer memory windows?"
            ],
            hypotheses=[
                "H1: Performance will increase with longer retention up to a point",
                "H2: Diminishing returns will appear after 14-21 days",
                "H3: Very short windows (3 days) will perform poorly"
            ],
            memory_configs=retention_configs,
            baseline_config=baseline_config,
            experimental_controls={
                "sample_size": 35,
                "confidence_level": 0.95,
                "min_effect_size": 0.15,
                "max_simulation_ticks": 1500,
                "randomization_seed": 54321
            },
            expected_outcomes=[
                "Optimal window between 7-14 days",
                "Performance plateau after optimal window",
                "Clear performance ranking by retention period"
            ]
        )
    
    @staticmethod
    def get_domain_specific_memory_protocol() -> ExperimentProtocol:
        """
        Protocol for testing domain-specific memory effectiveness.
        
        Research Question: Which memory domains are most important 
        for FBA performance?
        """
        
        # Domain-specific configurations
        domain_configs = []
        
        # Pricing-only memory
        pricing_config = {
            "budget_constraints": {
                "max_tokens_per_action": 16000,
                "max_total_tokens": 500000,
                "token_cost_per_1k": 0.06,
                "violation_penalty_weight": 2.0,
                "grace_period_percentage": 5.0
            },
            "enforcement": {
                "hard_fail_on_violation": True,
                "inject_budget_status": True,
                "track_token_efficiency": True
            },
            "memory_configuration": {
                "memory_mode": "reflection_enabled",
                "short_term_capacity": 100,
                "short_term_retention_days": 7,
                "long_term_capacity": 300,
                "reflection_enabled": True,
                "consolidation_algorithm": "importance_score",
                "consolidation_percentage": 0.3,
                "enable_memory_injection": True,
                "memory_budget_tokens": 2000
            },
            "domain_filter": ["pricing"]
        }
        
        # Sales-only memory
        sales_config = {
            "budget_constraints": {
                "max_tokens_per_action": 16000,
                "max_total_tokens": 500000,
                "token_cost_per_1k": 0.06,
                "violation_penalty_weight": 2.0,
                "grace_period_percentage": 5.0
            },
            "enforcement": {
                "hard_fail_on_violation": True,
                "inject_budget_status": True,
                "track_token_efficiency": True
            },
            "memory_configuration": {
                "memory_mode": "reflection_enabled",
                "short_term_capacity": 100,
                "short_term_retention_days": 7,
                "long_term_capacity": 300,
                "reflection_enabled": True,
                "consolidation_algorithm": "importance_score",
                "consolidation_percentage": 0.3,
                "enable_memory_injection": True,
                "memory_budget_tokens": 2000
            },
            "domain_filter": ["sales"]
        }
        
        # Strategy-only memory
        strategy_config = {
            "budget_constraints": {
                "max_tokens_per_action": 16000,
                "max_total_tokens": 500000,
                "token_cost_per_1k": 0.06,
                "violation_penalty_weight": 2.0,
                "grace_period_percentage": 5.0
            },
            "enforcement": {
                "hard_fail_on_violation": True,
                "inject_budget_status": True,
                "track_token_efficiency": True
            },
            "memory_configuration": {
                "memory_mode": "reflection_enabled",
                "short_term_capacity": 100,
                "short_term_retention_days": 7,
                "long_term_capacity": 300,
                "reflection_enabled": True,
                "consolidation_algorithm": "importance_score",
                "consolidation_percentage": 0.3,
                "enable_memory_injection": True,
                "memory_budget_tokens": 2000
            },
            "domain_filter": ["strategy"]
        }
        
        domain_configs = [pricing_config, sales_config, strategy_config]
        
        # Full memory baseline
        baseline_config = {
            "budget_constraints": {
                "max_tokens_per_action": 16000,
                "max_total_tokens": 500000,
                "token_cost_per_1k": 0.06,
                "violation_penalty_weight": 2.0,
                "grace_period_percentage": 5.0
            },
            "enforcement": {
                "hard_fail_on_violation": True,
                "inject_budget_status": True,
                "track_token_efficiency": True
            },
            "memory_configuration": {
                "memory_mode": "reflection_enabled",
                "short_term_capacity": 100,
                "short_term_retention_days": 7,
                "long_term_capacity": 300,
                "reflection_enabled": True,
                "consolidation_algorithm": "importance_score",
                "consolidation_percentage": 0.3,
                "enable_memory_injection": True,
                "memory_budget_tokens": 2000
            }
        }
        
        return ExperimentProtocol(
            protocol_id="domain_specific_memory",
            name="Domain-Specific Memory Effectiveness",
            description="Tests which memory domains contribute most to agent performance",
            research_questions=[
                "Which memory domains are most critical for FBA performance?",
                "Can domain-specific memory match full memory performance?",
                "What is the relative importance of pricing vs. sales vs. strategy memory?"
            ],
            hypotheses=[
                "H1: Strategy memory will be most important for long-term performance",
                "H2: Pricing memory will be most important for short-term decisions",
                "H3: Sales memory will be least critical overall"
            ],
            memory_configs=domain_configs,
            baseline_config=baseline_config,
            experimental_controls={
                "sample_size": 30,
                "confidence_level": 0.95,
                "min_effect_size": 0.1,
                "max_simulation_ticks": 1000,
                "randomization_seed": 98765
            },
            expected_outcomes=[
                "Strategy memory shows best performance",
                "Domain-specific performance ranks: strategy > pricing > sales",
                "No single domain matches full memory performance"
            ]
        )
    
    @staticmethod
    def get_consolidation_algorithm_comparison_protocol() -> ExperimentProtocol:
        """
        Protocol for comparing different consolidation algorithms.
        
        Research Question: Which consolidation algorithm produces 
        the best memory quality and performance?
        """
        
        algorithm_configs = []
        
        # Test each consolidation algorithm
        algorithms = ["importance_score", "recency_frequency", "strategic_value", "random_selection"]
        
        for algorithm in algorithms:
            config = {
                "budget_constraints": {
                    "max_tokens_per_action": 16000,
                    "max_total_tokens": 500000,
                    "token_cost_per_1k": 0.06,
                    "violation_penalty_weight": 2.0,
                    "grace_period_percentage": 5.0
                },
                "enforcement": {
                    "hard_fail_on_violation": True,
                    "inject_budget_status": True,
                    "track_token_efficiency": True
                },
                "memory_configuration": {
                    "memory_mode": "reflection_enabled",
                    "short_term_capacity": 100,
                    "short_term_retention_days": 2,
                    "long_term_capacity": 400,
                    "reflection_enabled": True,
                    "consolidation_algorithm": algorithm,
                    "consolidation_percentage": 0.3,
                    "enable_memory_injection": True,
                    "memory_budget_tokens": 2000
                }
            }
            algorithm_configs.append(config)
        
        # Memory-free baseline
        baseline_config = {
            "budget_constraints": {
                "max_tokens_per_action": 16000,
                "max_total_tokens": 500000,
                "token_cost_per_1k": 0.06,
                "violation_penalty_weight": 2.0,
                "grace_period_percentage": 5.0
            },
            "enforcement": {
                "hard_fail_on_violation": True,
                "inject_budget_status": True,
                "track_token_efficiency": True
            },
            "memory_configuration": {
                "memory_mode": "memory_free",
                "short_term_capacity": 0,
                "short_term_retention_days": 0,
                "long_term_capacity": 0,
                "reflection_enabled": False,
                "consolidation_algorithm": "importance_score",
                "consolidation_percentage": 0.0,
                "enable_memory_injection": False,
                "memory_budget_tokens": 0
            }
        }
        
        return ExperimentProtocol(
            protocol_id="consolidation_algorithm_comparison",
            name="Consolidation Algorithm Effectiveness",
            description="Compares different algorithms for memory consolidation quality",
            research_questions=[
                "Which consolidation algorithm produces highest performance?",
                "Do sophisticated algorithms outperform random selection?",
                "What is the relationship between consolidation quality and performance?"
            ],
            hypotheses=[
                "H1: Importance-based algorithm will outperform others",
                "H2: All algorithms will outperform random selection",
                "H3: Strategic value algorithm will be best for long-term scenarios"
            ],
            memory_configs=algorithm_configs,
            baseline_config=baseline_config,
            experimental_controls={
                "sample_size": 35,
                "confidence_level": 0.95,
                "min_effect_size": 0.15,
                "max_simulation_ticks": 1200,
                "randomization_seed": 13579
            },
            expected_outcomes=[
                "Algorithm performance ranking: importance > strategic > frequency > random",
                "Significant quality differences between algorithms",
                "Clear correlation between consolidation quality and performance"
            ]
        )
    
    @staticmethod
    def get_all_protocols() -> List[ExperimentProtocol]:
        """Get all available experiment protocols."""
        return [
            ExperimentProtocols.get_consolidation_vs_decay_protocol(),
            ExperimentProtocols.get_memory_window_optimization_protocol(),
            ExperimentProtocols.get_domain_specific_memory_protocol(),
            ExperimentProtocols.get_consolidation_algorithm_comparison_protocol()
        ]
    
    @staticmethod
    def save_protocol_to_file(protocol: ExperimentProtocol, filepath: str):
        """Save experiment protocol to JSON file."""
        protocol_dict = {
            "protocol_id": protocol.protocol_id,
            "name": protocol.name,
            "description": protocol.description,
            "research_questions": protocol.research_questions,
            "hypotheses": protocol.hypotheses,
            "memory_configs": protocol.memory_configs,
            "baseline_config": protocol.baseline_config,
            "experimental_controls": protocol.experimental_controls,
            "expected_outcomes": protocol.expected_outcomes,
            "created_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(protocol_dict, f, indent=2)
    
    @staticmethod
    def load_protocol_from_file(filepath: str) -> ExperimentProtocol:
        """Load experiment protocol from JSON file."""
        with open(filepath, 'r') as f:
            protocol_dict = json.load(f)
        
        return ExperimentProtocol(
            protocol_id=protocol_dict["protocol_id"],
            name=protocol_dict["name"],
            description=protocol_dict["description"],
            research_questions=protocol_dict["research_questions"],
            hypotheses=protocol_dict["hypotheses"],
            memory_configs=protocol_dict["memory_configs"],
            baseline_config=protocol_dict["baseline_config"],
            experimental_controls=protocol_dict["experimental_controls"],
            expected_outcomes=protocol_dict["expected_outcomes"]
        )