"""
Reflection Module

Handles daily memory consolidation and sorting, determining which memories
get promoted from short-term to long-term storage using various algorithms.
"""

import asyncio
import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .memory_config import MemoryConfig, ConsolidationAlgorithm
from .dual_memory_manager import MemoryEvent, DualMemoryManager


logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Results from a memory consolidation process."""
    memories_considered: int
    memories_promoted: int
    memories_discarded: int
    consolidation_time: datetime
    algorithm_used: str
    quality_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memories_considered": self.memories_considered,
            "memories_promoted": self.memories_promoted,
            "memories_discarded": self.memories_discarded,
            "consolidation_time": self.consolidation_time.isoformat(),
            "algorithm_used": self.algorithm_used,
            "quality_metrics": self.quality_metrics
        }


class ConsolidationAlgorithmBase(ABC):
    """Abstract base class for memory consolidation algorithms."""
    
    @abstractmethod
    async def score_memories(self, memories: List[MemoryEvent], config: MemoryConfig) -> Dict[str, float]:
        """
        Score memories for consolidation priority.
        
        Args:
            memories: List of candidate memories for consolidation
            config: Memory configuration
            
        Returns:
            Dict mapping memory event_id to consolidation score (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of this consolidation algorithm."""
        pass


class ImportanceScoreAlgorithm(ConsolidationAlgorithmBase):
    """Consolidation based on memory importance scores."""
    
    async def score_memories(self, memories: List[MemoryEvent], config: MemoryConfig) -> Dict[str, float]:
        """Score memories based on their importance scores and access patterns."""
        scores = {}
        
        for memory in memories:
            base_score = memory.importance_score
            
            # Boost score based on access frequency
            access_boost = min(0.3, memory.access_count * 0.05)
            
            # Boost score for recent access
            recency_boost = 0.0
            if memory.last_accessed:
                hours_since_access = (datetime.now() - memory.last_accessed).total_seconds() / 3600
                if hours_since_access < 24:
                    recency_boost = 0.2 * (1 - hours_since_access / 24)
            
            # Domain-specific adjustments
            domain_multiplier = {
                "strategy": 1.3,
                "pricing": 1.2,
                "sales": 1.1,
                "competitors": 1.0,
                "operations": 0.9
            }.get(memory.domain, 1.0)
            
            final_score = min(1.0, (base_score + access_boost + recency_boost) * domain_multiplier)
            scores[memory.event_id] = final_score
            
        return scores
    
    def get_algorithm_name(self) -> str:
        return "importance_score"


class RecencyFrequencyAlgorithm(ConsolidationAlgorithmBase):
    """Consolidation based on recency and frequency of access."""
    
    async def score_memories(self, memories: List[MemoryEvent], config: MemoryConfig) -> Dict[str, float]:
        """Score memories using recency-frequency analysis."""
        scores = {}
        current_time = datetime.now()
        
        # Calculate frequency and recency metrics
        max_access_count = max((m.access_count for m in memories), default=1)
        
        for memory in memories:
            # Frequency component (0.0 to 0.5)
            frequency_score = min(0.5, memory.access_count / max_access_count * 0.5)
            
            # Recency component (0.0 to 0.5)
            recency_score = 0.0
            if memory.last_accessed:
                hours_since_access = (current_time - memory.last_accessed).total_seconds() / 3600
                # More recent access = higher score, decay over 7 days
                recency_score = max(0.0, 0.5 * (1 - hours_since_access / (7 * 24)))
            
            scores[memory.event_id] = frequency_score + recency_score
            
        return scores
    
    def get_algorithm_name(self) -> str:
        return "recency_frequency"


class StrategicValueAlgorithm(ConsolidationAlgorithmBase):
    """Consolidation based on strategic business value."""
    
    async def score_memories(self, memories: List[MemoryEvent], config: MemoryConfig) -> Dict[str, float]:
        """Score memories based on strategic business importance."""
        scores = {}
        
        # Strategic event type weights
        strategic_weights = {
            "SaleOccurred": 0.7,
            "CompetitorPriceUpdated": 0.8,
            "ProductPriceUpdated": 0.6,
            "BudgetWarning": 0.9,
            "BudgetExceeded": 1.0,
            "DemandOscillationEvent": 0.8,
            "FeeHikeEvent": 0.7,
            "ReviewBombEvent": 0.9,
            "ListingHijackEvent": 1.0
        }
        
        for memory in memories:
            base_weight = strategic_weights.get(memory.event_type, 0.3)
            
            # Adjust for memory age - newer strategic events are more valuable
            age_days = (datetime.now() - memory.timestamp).days
            age_penalty = max(0.0, 1.0 - age_days * 0.1)  # 10% penalty per day
            
            # Adjust for domain strategic importance
            domain_weight = {
                "strategy": 1.0,
                "pricing": 0.9,
                "competitors": 0.8,
                "sales": 0.7,
                "operations": 0.6
            }.get(memory.domain, 0.5)
            
            scores[memory.event_id] = base_weight * age_penalty * domain_weight
            
        return scores
    
    def get_algorithm_name(self) -> str:
        return "strategic_value"


class RandomSelectionAlgorithm(ConsolidationAlgorithmBase):
    """Random consolidation for baseline experiments."""
    
    async def score_memories(self, memories: List[MemoryEvent], config: MemoryConfig) -> Dict[str, float]:
        """Assign random scores for baseline comparison."""
        scores = {}
        
        for memory in memories:
            scores[memory.event_id] = random.random()
            
        return scores
    
    def get_algorithm_name(self) -> str:
        return "random_selection"


class LLMReflectionAlgorithm(ConsolidationAlgorithmBase):
    """LLM-based reflection for intelligent memory consolidation."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client  # Optional LLM client for reflection
    
    async def score_memories(self, memories: List[MemoryEvent], config: MemoryConfig) -> Dict[str, float]:
        """Score memories using LLM-based reflection."""
        # For now, fallback to importance scoring if no LLM client
        if not self.llm_client:
            logger.warning("No LLM client provided for reflection, falling back to importance scoring")
            fallback = ImportanceScoreAlgorithm()
            return await fallback.score_memories(memories, config)
        
        # TODO: Implement LLM-based reflection
        # This would involve sending memory summaries to an LLM and asking it to
        # rate the strategic importance and relevance for long-term retention
        
        scores = {}
        for memory in memories:
            # Placeholder implementation
            scores[memory.event_id] = memory.importance_score
            
        return scores
    
    def get_algorithm_name(self) -> str:
        return "llm_reflection"


class ReflectionModule:
    """
    Daily reflection and memory consolidation system.
    
    Analyzes short-term memories and determines which should be promoted
    to long-term storage using configurable consolidation algorithms.
    """
    
    def __init__(self, memory_manager: DualMemoryManager, config: MemoryConfig):
        self.memory_manager = memory_manager
        self.config = config
        self.agent_id = memory_manager.agent_id
        
        # Initialize consolidation algorithms
        self.algorithms: Dict[ConsolidationAlgorithm, ConsolidationAlgorithmBase] = {
            ConsolidationAlgorithm.IMPORTANCE_SCORE: ImportanceScoreAlgorithm(),
            ConsolidationAlgorithm.RECENCY_FREQUENCY: RecencyFrequencyAlgorithm(),
            ConsolidationAlgorithm.STRATEGIC_VALUE: StrategicValueAlgorithm(),
            ConsolidationAlgorithm.RANDOM_SELECTION: RandomSelectionAlgorithm(),
            ConsolidationAlgorithm.LLM_REFLECTION: LLMReflectionAlgorithm()
        }
        
        # Reflection statistics
        self.reflection_history: List[ConsolidationResult] = []
        self.total_reflections = 0
        
        logger.info(f"ReflectionModule initialized for agent {self.agent_id}")
    
    async def perform_reflection(self, current_time: Optional[datetime] = None) -> ConsolidationResult:
        """
        Perform daily reflection and memory consolidation.
        
        Args:
            current_time: Current simulation time (defaults to now)
            
        Returns:
            ConsolidationResult with details about the consolidation process
        """
        current_time = current_time or datetime.now()
        
        logger.info(f"Starting reflection for agent {self.agent_id} at {current_time}")
        
        # Get candidate memories for consolidation
        candidate_memories = await self.memory_manager.get_memories_for_promotion()
        
        if not candidate_memories:
            logger.info("No memories available for consolidation")
            return ConsolidationResult(
                memories_considered=0,
                memories_promoted=0,
                memories_discarded=0,
                consolidation_time=current_time,
                algorithm_used=self.config.consolidation_algorithm.value,
                quality_metrics={}
            )
        
        # Get consolidation algorithm
        algorithm = self.algorithms[self.config.consolidation_algorithm]
        
        # Score memories for consolidation
        memory_scores = await algorithm.score_memories(candidate_memories, self.config)
        
        # Select memories for promotion
        memories_to_promote = await self._select_memories_for_promotion(
            candidate_memories, memory_scores
        )
        
        # Promote selected memories
        await self.memory_manager.promote_memories(memories_to_promote)
        
        # Clean up old short-term memories that weren't promoted
        memories_to_discard = [
            m for m in candidate_memories 
            if m not in memories_to_promote and self._should_discard_memory(m, current_time)
        ]
        
        if memories_to_discard:
            memory_ids_to_discard = [m.event_id for m in memories_to_discard]
            await self.memory_manager.short_term_store.remove(memory_ids_to_discard)
        
        # Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(
            candidate_memories, memories_to_promote, memory_scores
        )
        
        # Create consolidation result
        result = ConsolidationResult(
            memories_considered=len(candidate_memories),
            memories_promoted=len(memories_to_promote),
            memories_discarded=len(memories_to_discard),
            consolidation_time=current_time,
            algorithm_used=algorithm.get_algorithm_name(),
            quality_metrics=quality_metrics
        )
        
        # Update reflection statistics
        self.reflection_history.append(result)
        self.total_reflections += 1
        self.memory_manager.last_reflection_time = current_time
        
        logger.info(f"Reflection completed: {result.memories_promoted} promoted, {result.memories_discarded} discarded")
        
        return result
    
    async def _select_memories_for_promotion(self, 
                                           candidate_memories: List[MemoryEvent],
                                           memory_scores: Dict[str, float]) -> List[MemoryEvent]:
        """Select memories for promotion based on scores and capacity constraints."""
        
        # Sort memories by consolidation score
        scored_memories = [
            (memory, memory_scores.get(memory.event_id, 0.0))
            for memory in candidate_memories
        ]
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate how many memories to promote
        total_candidates = len(candidate_memories)
        max_promotions = int(total_candidates * self.config.consolidation_percentage)
        
        # Check long-term memory capacity
        current_long_term_size = await self.memory_manager.long_term_store.size()
        available_capacity = max(0, self.config.long_term_capacity - current_long_term_size)
        
        # Limit promotions by available capacity
        max_promotions = min(max_promotions, available_capacity)
        
        # Select top-scored memories for promotion
        memories_to_promote = [
            memory for memory, score in scored_memories[:max_promotions]
            if score > 0.1  # Minimum threshold for promotion
        ]
        
        return memories_to_promote
    
    def _should_discard_memory(self, memory: MemoryEvent, current_time: datetime) -> bool:
        """Determine if a memory should be discarded from short-term storage."""
        
        # Check if memory has exceeded retention period
        age = current_time - memory.timestamp
        max_age = timedelta(days=self.config.short_term_retention_days)
        
        if age > max_age:
            return True
        
        # Check if memory has very low importance and hasn't been accessed
        if memory.importance_score < 0.1 and memory.access_count == 0:
            return True
        
        return False
    
    async def _calculate_quality_metrics(self, 
                                       candidate_memories: List[MemoryEvent],
                                       promoted_memories: List[MemoryEvent],
                                       memory_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate quality metrics for the consolidation process."""
        
        if not candidate_memories:
            return {}
        
        # Calculate promotion rate
        promotion_rate = len(promoted_memories) / len(candidate_memories)
        
        # Calculate average score of promoted memories
        promoted_scores = [
            memory_scores.get(memory.event_id, 0.0) for memory in promoted_memories
        ]
        avg_promoted_score = sum(promoted_scores) / len(promoted_scores) if promoted_scores else 0.0
        
        # Calculate average score of all candidates
        all_scores = [memory_scores.get(memory.event_id, 0.0) for memory in candidate_memories]
        avg_candidate_score = sum(all_scores) / len(all_scores)
        
        # Calculate score selectivity (how much better promoted memories scored)
        score_selectivity = avg_promoted_score - avg_candidate_score if avg_candidate_score > 0 else 0.0
        
        # Domain diversity of promoted memories
        promoted_domains = set(memory.domain for memory in promoted_memories)
        domain_diversity = len(promoted_domains) / len(self.config.memory_domains)
        
        # Event type diversity
        promoted_event_types = set(memory.event_type for memory in promoted_memories)
        event_type_diversity = len(promoted_event_types)
        
        return {
            "promotion_rate": promotion_rate,
            "avg_promoted_score": avg_promoted_score,
            "avg_candidate_score": avg_candidate_score,
            "score_selectivity": score_selectivity,
            "domain_diversity": domain_diversity,
            "event_type_diversity": event_type_diversity
        }
    
    def get_reflection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about reflection history."""
        
        if not self.reflection_history:
            return {
                "total_reflections": 0,
                "avg_promotion_rate": 0.0,
                "avg_quality_score": 0.0,
                "reflection_history": []
            }
        
        # Calculate averages
        total_considered = sum(r.memories_considered for r in self.reflection_history)
        total_promoted = sum(r.memories_promoted for r in self.reflection_history)
        avg_promotion_rate = total_promoted / total_considered if total_considered > 0 else 0.0
        
        # Calculate average quality score (composite metric)
        quality_scores = []
        for result in self.reflection_history:
            if result.quality_metrics:
                # Composite quality score from multiple metrics
                quality_score = (
                    result.quality_metrics.get("score_selectivity", 0.0) * 0.4 +
                    result.quality_metrics.get("domain_diversity", 0.0) * 0.3 +
                    result.quality_metrics.get("avg_promoted_score", 0.0) * 0.3
                )
                quality_scores.append(quality_score)
        
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {
            "total_reflections": self.total_reflections,
            "avg_promotion_rate": avg_promotion_rate,
            "avg_quality_score": avg_quality_score,
            "total_memories_considered": total_considered,
            "total_memories_promoted": total_promoted,
            "reflection_history": [result.to_dict() for result in self.reflection_history[-10:]]  # Last 10
        }
    
    async def set_consolidation_algorithm(self, algorithm: ConsolidationAlgorithm):
        """Change the consolidation algorithm for future reflections."""
        if algorithm in self.algorithms:
            self.config.consolidation_algorithm = algorithm
            logger.info(f"Consolidation algorithm changed to {algorithm.value}")
        else:
            raise ValueError(f"Unknown consolidation algorithm: {algorithm}")
    
    def clear_reflection_history(self):
        """Clear reflection history for fresh experiments."""
        self.reflection_history.clear()
        self.total_reflections = 0
        logger.info("Reflection history cleared")