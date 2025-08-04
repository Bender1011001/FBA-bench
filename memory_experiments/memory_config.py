"""
Memory Configuration System

Extends FBA-Bench's existing ConstraintConfig to include memory-specific settings
for dual-memory architecture and reflection parameters.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

from constraints.constraint_config import ConstraintConfig


class MemoryMode(Enum):
    """Memory experiment modes for systematic testing."""
    MEMORY_FREE = "memory_free"
    SHORT_TERM_ONLY = "short_term_only" 
    LONG_TERM_ONLY = "long_term_only"
    REFLECTION_ENABLED = "reflection_enabled"
    CONSOLIDATION_DISABLED = "consolidation_disabled"
    HYBRID_REFLECTION = "hybrid_reflection"


class ConsolidationAlgorithm(Enum):
    """Algorithms for determining memory consolidation from short-term to long-term."""
    IMPORTANCE_SCORE = "importance_score"
    RECENCY_FREQUENCY = "recency_frequency"
    STRATEGIC_VALUE = "strategic_value"
    RANDOM_SELECTION = "random_selection"
    LLM_REFLECTION = "llm_reflection"


class DecayFunction(Enum):
    """Memory decay functions for forgetting experiments."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP_FUNCTION = "step_function"


class MemoryStoreType(Enum):
    """Types of memory store implementations."""
    IN_MEMORY = "in_memory"
    # Add other store types here as they become available
    # For example:
    # SQLITE = "sqlite"
    # REDIS = "redis"
    # VECTOR_DB = "vector_db"


@dataclass
class MemoryConfig:
    """
    Configuration for memory experiments extending FBA-Bench constraint system.
    
    Supports dual-memory architecture with short-term/long-term stores,
    daily reflection, and memory consolidation algorithms.
    """
    
    # Base constraint config
    base_config: ConstraintConfig
    
    # Memory Mode Configuration
    memory_mode: MemoryMode = MemoryMode.REFLECTION_ENABLED
    
    # Short-Term Memory Settings
    short_term_capacity: int = 100  # Max events in short-term memory
    short_term_retention_days: int = 1  # How long memories stay in short-term
    short_term_decay_function: DecayFunction = DecayFunction.LINEAR
    
    # Long-Term Memory Settings  
    long_term_capacity: int = 1000  # Max events in long-term memory
    long_term_retention_days: Optional[int] = None  # None = unlimited
    long_term_decay_function: DecayFunction = DecayFunction.NONE
    
    # Reflection and Consolidation Settings
    reflection_enabled: bool = True
    reflection_frequency_hours: int = 24  # Daily reflection by default
    consolidation_algorithm: ConsolidationAlgorithm = ConsolidationAlgorithm.IMPORTANCE_SCORE
    consolidation_percentage: float = 0.3  # % of short-term memories to promote
    
    # Memory Retrieval Settings
    max_retrieval_events: int = 20  # Max memories retrieved per query
    retrieval_relevance_threshold: float = 0.7  # Minimum similarity score
    
    # Vector Store Configuration
    vector_store_enabled: bool = True
    vector_store_backend: str = "faiss"  # faiss, chroma, pinecone
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dimension: int = 384
    
    # Experiment-Specific Settings
    enable_memory_injection: bool = True  # Inject memory into agent prompts
    memory_budget_tokens: int = 2000  # Token budget for memory content
    track_memory_usage: bool = True
    
    # Advanced Settings
    memory_domains: List[str] = field(default_factory=lambda: [
        "pricing", "sales", "competitors", "strategy", "operations", "all"
    ])
    domain_specific_retention: Dict[str, int] = field(default_factory=dict)
    
    # Memory Store Configuration
    short_term_store_type: MemoryStoreType = MemoryStoreType.IN_MEMORY
    long_term_store_type: MemoryStoreType = MemoryStoreType.IN_MEMORY
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'MemoryConfig':
        """Load memory configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Load base constraint config
        base_config = ConstraintConfig.from_yaml(filepath)
        
        # Extract memory-specific settings
        memory_settings = config_dict.get('memory_configuration', {})
        
        return cls(
            base_config=base_config,
            memory_mode=MemoryMode(memory_settings.get('memory_mode', 'reflection_enabled')),
            short_term_capacity=memory_settings.get('short_term_capacity', 100),
            short_term_retention_days=memory_settings.get('short_term_retention_days', 1),
            short_term_decay_function=DecayFunction(memory_settings.get('short_term_decay_function', 'linear')),
            long_term_capacity=memory_settings.get('long_term_capacity', 1000),
            long_term_retention_days=memory_settings.get('long_term_retention_days'),
            long_term_decay_function=DecayFunction(memory_settings.get('long_term_decay_function', 'none')),
            reflection_enabled=memory_settings.get('reflection_enabled', True),
            reflection_frequency_hours=memory_settings.get('reflection_frequency_hours', 24),
            consolidation_algorithm=ConsolidationAlgorithm(memory_settings.get('consolidation_algorithm', 'importance_score')),
            consolidation_percentage=memory_settings.get('consolidation_percentage', 0.3),
            max_retrieval_events=memory_settings.get('max_retrieval_events', 20),
            retrieval_relevance_threshold=memory_settings.get('retrieval_relevance_threshold', 0.7),
            vector_store_enabled=memory_settings.get('vector_store_enabled', True),
            vector_store_backend=memory_settings.get('vector_store_backend', 'faiss'),
            embedding_model=memory_settings.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            vector_dimension=memory_settings.get('vector_dimension', 384),
            enable_memory_injection=memory_settings.get('enable_memory_injection', True),
            memory_budget_tokens=memory_settings.get('memory_budget_tokens', 2000),
            track_memory_usage=memory_settings.get('track_memory_usage', True),
            memory_domains=memory_settings.get('memory_domains', ['pricing', 'sales', 'competitors', 'strategy', 'operations', 'all']),
            domain_specific_retention=memory_settings.get('domain_specific_retention', {}),
            short_term_store_type=MemoryStoreType(memory_settings.get('short_term_store_type', 'in_memory')),
            long_term_store_type=MemoryStoreType(memory_settings.get('long_term_store_type', 'in_memory'))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'base_config': {
                'max_tokens_per_action': self.base_config.max_tokens_per_action,
                'max_total_tokens': self.base_config.max_total_tokens,
                'token_cost_per_1k': self.base_config.token_cost_per_1k,
                'violation_penalty_weight': self.base_config.violation_penalty_weight,
                'grace_period_percentage': self.base_config.grace_period_percentage,
                'hard_fail_on_violation': self.base_config.hard_fail_on_violation,
                'inject_budget_status': self.base_config.inject_budget_status,
                'track_token_efficiency': self.base_config.track_token_efficiency
            },
            'memory_configuration': {
                'memory_mode': self.memory_mode.value,
                'short_term_capacity': self.short_term_capacity,
                'short_term_retention_days': self.short_term_retention_days,
                'short_term_decay_function': self.short_term_decay_function.value,
                'long_term_capacity': self.long_term_capacity,
                'long_term_retention_days': self.long_term_retention_days,
                'long_term_decay_function': self.long_term_decay_function.value,
                'reflection_enabled': self.reflection_enabled,
                'reflection_frequency_hours': self.reflection_frequency_hours,
                'consolidation_algorithm': self.consolidation_algorithm.value,
                'consolidation_percentage': self.consolidation_percentage,
                'max_retrieval_events': self.max_retrieval_events,
                'retrieval_relevance_threshold': self.retrieval_relevance_threshold,
                'vector_store_enabled': self.vector_store_enabled,
                'vector_store_backend': self.vector_store_backend,
                'embedding_model': self.embedding_model,
                'vector_dimension': self.vector_dimension,
                'enable_memory_injection': self.enable_memory_injection,
                'memory_budget_tokens': self.memory_budget_tokens,
                'track_memory_usage': self.track_memory_usage,
                'memory_domains': self.memory_domains,
                'domain_specific_retention': self.domain_specific_retention,
                'short_term_store_type': self.short_term_store_type.value,
                'long_term_store_type': self.long_term_store_type.value
            }
        }
    
    def get_effective_token_budget(self) -> int:
        """Calculate effective token budget including memory allocation."""
        base_budget = self.base_config.max_tokens_per_action
        if self.enable_memory_injection:
            return max(0, base_budget - self.memory_budget_tokens)
        return base_budget
    
    def is_memory_enabled(self) -> bool:
        """Check if any memory functionality is enabled."""
        return self.memory_mode != MemoryMode.MEMORY_FREE
    
    def should_use_reflection(self) -> bool:
        """Check if reflection should be used based on mode and settings."""
        reflection_modes = {
            MemoryMode.REFLECTION_ENABLED,
            MemoryMode.HYBRID_REFLECTION
        }
        return self.memory_mode in reflection_modes and self.reflection_enabled


# Default configurations for different experiment types
def get_memory_free_config(base_config: ConstraintConfig) -> MemoryConfig:
    """Create a memory-free configuration for baseline experiments."""
    return MemoryConfig(
        base_config=base_config,
        memory_mode=MemoryMode.MEMORY_FREE,
        vector_store_enabled=False,
        enable_memory_injection=False,
        reflection_enabled=False
    )


def get_ablated_memory_config(base_config: ConstraintConfig, retention_days: int = 7) -> MemoryConfig:
    """Create an ablated memory configuration with limited retention."""
    return MemoryConfig(
        base_config=base_config,
        memory_mode=MemoryMode.SHORT_TERM_ONLY,
        short_term_retention_days=retention_days,
        long_term_capacity=0,
        reflection_enabled=False
    )


def get_saturated_memory_config(base_config: ConstraintConfig) -> MemoryConfig:
    """Create a memory-saturated configuration with full context."""
    return MemoryConfig(
        base_config=base_config,
        memory_mode=MemoryMode.REFLECTION_ENABLED,
        short_term_capacity=500,
        long_term_capacity=5000,
        long_term_retention_days=None,  # Unlimited
        memory_budget_tokens=4000,  # High memory budget
        consolidation_percentage=0.5  # Promote more memories
    )