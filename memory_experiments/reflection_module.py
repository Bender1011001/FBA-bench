"""
Reflection Module

Handles daily memory consolidation and sorting, determining which memories
get promoted from short-term to long-term storage using various algorithms.
"""

import asyncio
import logging
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .memory_config import MemoryConfig, ConsolidationAlgorithm
from .dual_memory_manager import MemoryEvent, DualMemoryManager
from event_bus import EventBus, get_event_bus
from events import BaseEvent


logger = logging.getLogger(__name__)


class ReflectionTrigger(Enum):
    """Types of reflection triggers for structured reflection loops."""
    PERIODIC = "periodic"
    EVENT_DRIVEN = "event_driven"
    PERFORMANCE_THRESHOLD = "performance_threshold"


@dataclass
class ReflectionInsight:
    """A structured insight generated from reflection analysis."""
    insight_id: str
    category: str  # e.g., "strategy", "performance", "behavior", "environment"
    title: str
    description: str
    evidence: List[str]  # Supporting evidence from analysis
    confidence: float  # 0.0 to 1.0
    actionability: float  # How actionable this insight is
    priority: str  # "low", "medium", "high", "critical"
    suggested_actions: List[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "actionability": self.actionability,
            "priority": self.priority,
            "suggested_actions": self.suggested_actions,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class PolicyAdjustment:
    """A policy adjustment recommendation from reflection."""
    adjustment_id: str
    policy_area: str  # e.g., "pricing", "inventory", "marketing", "risk"
    current_parameters: Dict[str, Any]
    recommended_parameters: Dict[str, Any]
    rationale: str
    expected_impact: Dict[str, float]
    confidence: float
    implementation_urgency: str  # "immediate", "within_day", "within_week", "next_cycle"
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adjustment_id": self.adjustment_id,
            "policy_area": self.policy_area,
            "current_parameters": self.current_parameters,
            "recommended_parameters": self.recommended_parameters,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "implementation_urgency": self.implementation_urgency,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class StructuredReflectionResult:
    """Comprehensive result from structured reflection process."""
    reflection_id: str
    agent_id: str
    trigger_type: ReflectionTrigger
    reflection_timestamp: datetime
    analysis_period_start: datetime
    analysis_period_end: datetime
    
    # Analysis results
    decisions_analyzed: int
    events_processed: int
    performance_metrics: Dict[str, float]
    
    # Generated insights
    insights: List[ReflectionInsight]
    critical_insights_count: int
    
    # Policy recommendations
    policy_adjustments: List[PolicyAdjustment]
    high_priority_adjustments: int
    
    # Reflection quality metrics
    analysis_depth_score: float
    insight_novelty_score: float
    actionability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflection_id": self.reflection_id,
            "agent_id": self.agent_id,
            "trigger_type": self.trigger_type.value,
            "reflection_timestamp": self.reflection_timestamp.isoformat(),
            "analysis_period_start": self.analysis_period_start.isoformat(),
            "analysis_period_end": self.analysis_period_end.isoformat(),
            "decisions_analyzed": self.decisions_analyzed,
            "events_processed": self.events_processed,
            "performance_metrics": self.performance_metrics,
            "insights": [insight.to_dict() for insight in self.insights],
            "critical_insights_count": self.critical_insights_count,
            "policy_adjustments": [adj.to_dict() for adj in self.policy_adjustments],
            "high_priority_adjustments": self.high_priority_adjustments,
            "analysis_depth_score": self.analysis_depth_score,
            "insight_novelty_score": self.insight_novelty_score,
            "actionability_score": self.actionability_score
        }


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
        
        prompt_template = """
        You are an advanced cognitive reflection module for an FBA agent. Your task is to review a list of recent memories (events) and assess their strategic importance for long-term retention.
        
        For each memory, provide a strategic importance score between 0.0 (not important) and 1.0 (critically important).
        
        Consider the following criteria for importance:
        - **Impact on long-term business goals:** Does this memory influence strategic decisions, market positioning, or sustained profitability?
        - **Novelty/Surprise:** Was this an unexpected event or observation that changes the agent's understanding of the environment?
        - **Actionability:** Does this memory directly inform potential future actions or prevent past mistakes?
        - **Recurrence:** Is this a pattern or trend that the agent should learn from?
        - **Severity/Magnitude:** For negative events, how significant was the impact?
        
        Your output should be a JSON array of objects, each with the `event_id` of the memory and its `strategic_score`.
        
        ---
        
        MEMORIES FOR REFLECTION:
        {memories_json}
        
        ---
        
        Provide your response as a JSON array ONLY:
        [
            {{
                "event_id": "...",
                "strategic_score": 0.X
            }},
            ...
        ]
        """
        
        if not self.llm_client:
            logger.warning("No LLM client provided for reflection, falling back to ImportanceScoreAlgorithm.")
            fallback = ImportanceScoreAlgorithm()
            return await fallback.score_memories(memories, config)

        memory_summaries = []
        for memory in memories:
            memory_summaries.append({
                "event_id": memory.event_id,
                "timestamp": memory.timestamp.isoformat(),
                "content": memory.content,
                "domain": memory.domain,
                "event_type": memory.event_type
            })

        memories_json = json.dumps(memory_summaries, indent=2)
        full_prompt = prompt_template.format(memories_json=memories_json)

        try:
            # For reflection, we consider the current action type as 'reflection'
            llm_response = await self.llm_client.generate_response(
                prompt=full_prompt,
                model=config.embedding_model, # Using embedding model for reflection for consistency, or a dedicated reflection model
                temperature=0.1, # Keep reflection deterministic for consistent scoring
                max_tokens=config.memory_budget_tokens # Use memory budget for reflection
            )
            
            response_content = llm_response.get('choices', [{}])[0].get('message', {}).get('content')
            
            if not response_content:
                raise ValueError("LLM reflection response content is empty.")
            
            scores_list = json.loads(response_content)
            
            scores = {item['event_id']: item['strategic_score'] for item in scores_list}
            
            # Validate scores
            for event_id, score in scores.items():
                if not (0.0 <= score <= 1.0):
                    logger.warning(f"LLM returned out-of-range score {score} for {event_id}. Clamping to 0-1.")
                    scores[event_id] = max(0.0, min(1.0, score))

            if not any(scores.values()):
                logger.warning("LLM reflection resulted in all zero scores. Falling back to ImportanceScoreAlgorithm.")
                fallback = ImportanceScoreAlgorithm()
                return await fallback.score_memories(memories, config)

            return scores
        
        except json.JSONDecodeError as e:
            logger.error(f"LLM reflection response was not valid JSON: {response_content}. Error: {e}")
            logger.warning("Falling back to ImportanceScoreAlgorithm due to invalid LLM response.")
            fallback = ImportanceScoreAlgorithm()
            return await fallback.score_memories(memories, config)
        except Exception as e:
            logger.error(f"Error during LLM reflection: {e}")
            logger.warning("Falling back to ImportanceScoreAlgorithm due to LLM reflection error.")
            fallback = ImportanceScoreAlgorithm()
            return await fallback.score_memories(memories, config)
    
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


class StructuredReflectionLoop:
    """
    Advanced structured reflection system that provides deep cognitive analysis.
    
    Implements systematic reflection cycles triggered by various conditions,
    analyzes agent performance and decisions, generates actionable insights,
    and recommends policy adjustments for improved decision-making.
    """
    
    def __init__(self, agent_id: str, memory_manager: DualMemoryManager,
                 config: MemoryConfig, event_bus: Optional[EventBus] = None):
        """
        Initialize the Structured Reflection Loop.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_manager: Memory management system
            config: Memory configuration
            event_bus: Event bus for publishing reflection events
        """
        self.agent_id = agent_id
        self.memory_manager = memory_manager
        self.config = config
        self.event_bus = event_bus or get_event_bus()
        
        # Reflection state
        self.reflection_history: List[StructuredReflectionResult] = []
        self.last_reflection_time: Optional[datetime] = None
        self.reflection_triggers_enabled: Dict[ReflectionTrigger, bool] = {
            ReflectionTrigger.PERIODIC: True,
            ReflectionTrigger.EVENT_DRIVEN: True,
            ReflectionTrigger.PERFORMANCE_THRESHOLD: True
        }
        
        # Performance tracking for threshold triggers
        self.performance_history: List[Dict[str, float]] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.major_events: List[Dict[str, Any]] = []
        
        # Reflection parameters
        self.periodic_interval_hours = 24  # Daily reflection by default
        self.performance_threshold_degradation = 0.2  # 20% performance drop triggers reflection
        self.min_decisions_for_reflection = 5
        
        # Insight tracking
        self.insight_categories = ["strategy", "performance", "behavior", "environment", "risk"]
        self.policy_areas = ["pricing", "inventory", "marketing", "risk_management", "operations"]
        
        logger.info(f"StructuredReflectionLoop initialized for agent {agent_id}")
    
    async def trigger_reflection(self, tick_interval: Optional[int] = None,
                                major_events: Optional[List[Dict[str, Any]]] = None) -> Optional[StructuredReflectionResult]:
        """
        Initiate a reflection cycle based on various triggers.
        
        Args:
            tick_interval: Current tick interval for periodic checks
            major_events: List of major events that might trigger reflection
            
        Returns:
            StructuredReflectionResult if reflection was triggered, None otherwise
        """
        current_time = datetime.now()
        
        # Determine trigger type
        trigger_type = await self._determine_reflection_trigger(tick_interval, major_events, current_time)
        
        if trigger_type is None:
            return None
        
        logger.info(f"Triggering structured reflection for agent {self.agent_id} - trigger: {trigger_type.value}")
        
        # Perform structured reflection
        reflection_result = await self._perform_structured_reflection(trigger_type, current_time)
        
        # Store reflection result
        self.reflection_history.append(reflection_result)
        self.last_reflection_time = current_time
        
        # Publish reflection completed event
        await self._publish_reflection_completed_event(reflection_result)
        
        logger.info(f"Structured reflection completed - generated {len(reflection_result.insights)} insights "
                   f"and {len(reflection_result.policy_adjustments)} policy adjustments")
        
        return reflection_result
    
    async def analyze_recent_decisions(self, event_history: List[Dict[str, Any]],
                                      outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze recent decisions and their outcomes for learning.
        
        Args:
            event_history: Recent events and decisions
            outcomes: Outcomes and results from recent decisions
            
        Returns:
            Analysis results with patterns, successes, and failures
        """
        logger.info(f"Analyzing {len(event_history)} recent decisions for agent {self.agent_id}")
        
        analysis = {
            "analysis_timestamp": datetime.now().isoformat(),
            "decisions_analyzed": len(event_history),
            "performance_patterns": {},
            "decision_effectiveness": {},
            "failure_analysis": {},
            "success_factors": {},
            "recommendations": []
        }
        
        # Categorize decisions by type
        decision_categories = {}
        for event in event_history:
            decision_type = event.get("decision_type", "unknown")
            if decision_type not in decision_categories:
                decision_categories[decision_type] = []
            decision_categories[decision_type].append(event)
        
        # Analyze each decision category
        for decision_type, decisions in decision_categories.items():
            category_analysis = await self._analyze_decision_category(
                decision_type, decisions, outcomes
            )
            analysis["decision_effectiveness"][decision_type] = category_analysis
        
        # Identify performance patterns
        analysis["performance_patterns"] = await self._identify_performance_patterns(
            event_history, outcomes
        )
        
        # Analyze failures and successes
        analysis["failure_analysis"] = await self._analyze_decision_failures(
            event_history, outcomes
        )
        analysis["success_factors"] = await self._analyze_decision_successes(
            event_history, outcomes
        )
        
        # Generate recommendations
        analysis["recommendations"] = await self._generate_decision_recommendations(
            analysis
        )
        
        return analysis
    
    async def generate_insights(self, analysis_results: Dict[str, Any]) -> List[ReflectionInsight]:
        """
        Generate actionable insights from analysis results.
        
        Args:
            analysis_results: Results from decision and performance analysis
            
        Returns:
            List of structured insights with priorities and suggested actions
        """
        logger.info(f"Generating insights from analysis results for agent {self.agent_id}")
        
        insights = []
        current_time = datetime.now()
        
        # Generate insights from performance patterns
        pattern_insights = await self._generate_pattern_insights(analysis_results, current_time)
        insights.extend(pattern_insights)
        
        # Generate insights from decision effectiveness
        decision_insights = await self._generate_decision_insights(analysis_results, current_time)
        insights.extend(decision_insights)
        
        # Generate insights from failure analysis
        failure_insights = await self._generate_failure_insights(analysis_results, current_time)
        insights.extend(failure_insights)
        
        # Generate strategic insights
        strategic_insights = await self._generate_strategic_insights(analysis_results, current_time)
        insights.extend(strategic_insights)
        
        # Rank insights by priority and actionability
        insights = await self._rank_insights_by_priority(insights)
        
        logger.info(f"Generated {len(insights)} insights - "
                   f"{len([i for i in insights if i.priority == 'critical'])} critical")
        
        return insights
    
    async def update_agent_policy(self, insights: List[ReflectionInsight]) -> List[PolicyAdjustment]:
        """
        Generate policy adjustments based on insights.
        
        Args:
            insights: List of reflection insights
            
        Returns:
            List of recommended policy adjustments
        """
        logger.info(f"Updating agent policy based on {len(insights)} insights")
        
        policy_adjustments = []
        current_time = datetime.now()
        
        # Group insights by policy area
        insights_by_area = {}
        for insight in insights:
            policy_area = self._map_insight_to_policy_area(insight)
            if policy_area not in insights_by_area:
                insights_by_area[policy_area] = []
            insights_by_area[policy_area].append(insight)
        
        # Generate policy adjustments for each area
        for policy_area, area_insights in insights_by_area.items():
            adjustments = await self._generate_policy_adjustments_for_area(
                policy_area, area_insights, current_time
            )
            policy_adjustments.extend(adjustments)
        
        # Validate and prioritize adjustments
        validated_adjustments = await self._validate_policy_adjustments(policy_adjustments)
        
        logger.info(f"Generated {len(validated_adjustments)} policy adjustments - "
                   f"{len([a for a in validated_adjustments if a.implementation_urgency == 'immediate'])} immediate")
        
        return validated_adjustments
    
    def get_reflection_status(self) -> Dict[str, Any]:
        """Get comprehensive status of reflection system."""
        current_time = datetime.now()
        
        status = {
            "agent_id": self.agent_id,
            "last_reflection": self.last_reflection_time.isoformat() if self.last_reflection_time else None,
            "total_reflections": len(self.reflection_history),
            "enabled_triggers": [t.value for t, enabled in self.reflection_triggers_enabled.items() if enabled],
            "periodic_interval_hours": self.periodic_interval_hours,
            "performance_threshold": self.performance_threshold_degradation,
            "decisions_tracked": len(self.decision_history),
            "major_events_tracked": len(self.major_events)
        }
        
        if self.reflection_history:
            # Calculate average reflection quality
            quality_scores = [r.analysis_depth_score for r in self.reflection_history]
            status["average_reflection_quality"] = sum(quality_scores) / len(quality_scores)
            
            # Count insights and adjustments
            total_insights = sum(len(r.insights) for r in self.reflection_history)
            total_adjustments = sum(len(r.policy_adjustments) for r in self.reflection_history)
            status["total_insights_generated"] = total_insights
            status["total_policy_adjustments"] = total_adjustments
            
            # Recent reflection summary
            latest_reflection = self.reflection_history[-1]
            status["latest_reflection"] = {
                "timestamp": latest_reflection.reflection_timestamp.isoformat(),
                "trigger": latest_reflection.trigger_type.value,
                "insights_count": len(latest_reflection.insights),
                "critical_insights": latest_reflection.critical_insights_count,
                "policy_adjustments": len(latest_reflection.policy_adjustments)
            }
        
        return status
    
    # Private helper methods
    
    async def _determine_reflection_trigger(self, tick_interval: Optional[int],
                                           major_events: Optional[List[Dict[str, Any]]],
                                           current_time: datetime) -> Optional[ReflectionTrigger]:
        """Determine if reflection should be triggered and which type."""
        
        # Check periodic trigger
        if self.reflection_triggers_enabled[ReflectionTrigger.PERIODIC]:
            if await self._should_trigger_periodic_reflection(current_time):
                return ReflectionTrigger.PERIODIC
        
        # Check event-driven trigger
        if self.reflection_triggers_enabled[ReflectionTrigger.EVENT_DRIVEN]:
            if await self._should_trigger_event_driven_reflection(major_events):
                return ReflectionTrigger.EVENT_DRIVEN
        
        # Check performance threshold trigger
        if self.reflection_triggers_enabled[ReflectionTrigger.PERFORMANCE_THRESHOLD]:
            if await self._should_trigger_performance_threshold_reflection():
                return ReflectionTrigger.PERFORMANCE_THRESHOLD
        
        return None
    
    async def _should_trigger_periodic_reflection(self, current_time: datetime) -> bool:
        """Check if periodic reflection should be triggered."""
        if not self.last_reflection_time:
            return True
        
        hours_since_reflection = (current_time - self.last_reflection_time).total_seconds() / 3600
        return hours_since_reflection >= self.periodic_interval_hours
    
    async def _should_trigger_event_driven_reflection(self, major_events: Optional[List[Dict[str, Any]]]) -> bool:
        """Check if event-driven reflection should be triggered."""
        if not major_events:
            return False
        
        # Check for high-impact events
        high_impact_events = [
            event for event in major_events
            if event.get("severity", 0) > 0.7 or event.get("impact_level") == "high"
        ]
        
        return len(high_impact_events) > 0
    
    async def _should_trigger_performance_threshold_reflection(self) -> bool:
        """Check if performance degradation should trigger reflection."""
        if len(self.performance_history) < 2:
            return False
        
        # Compare recent performance to baseline
        recent_performance = self.performance_history[-1]
        baseline_performance = sum(
            perf.get("overall_score", 0.5) for perf in self.performance_history[-5:]
        ) / min(5, len(self.performance_history))
        
        current_performance = recent_performance.get("overall_score", 0.5)
        performance_drop = baseline_performance - current_performance
        
        return performance_drop > self.performance_threshold_degradation
    
    async def _perform_structured_reflection(self, trigger_type: ReflectionTrigger,
                                           current_time: datetime) -> StructuredReflectionResult:
        """Perform comprehensive structured reflection."""
        
        # Define analysis period
        analysis_period_start = current_time - timedelta(hours=self.periodic_interval_hours)
        if self.last_reflection_time:
            analysis_period_start = self.last_reflection_time
        
        # Gather data for analysis
        event_history = await self._gather_analysis_events(analysis_period_start, current_time)
        decision_history = await self._gather_decision_history(analysis_period_start, current_time)
        performance_metrics = await self._calculate_performance_metrics(analysis_period_start, current_time)
        
        # Perform decision analysis
        decision_analysis = await self.analyze_recent_decisions(decision_history, performance_metrics)
        
        # Generate insights
        insights = await self.generate_insights(decision_analysis)
        
        # Generate policy adjustments
        policy_adjustments = await self.update_agent_policy(insights)
        
        # Calculate reflection quality metrics
        analysis_depth_score = await self._calculate_analysis_depth_score(decision_analysis)
        insight_novelty_score = await self._calculate_insight_novelty_score(insights)
        actionability_score = await self._calculate_actionability_score(insights, policy_adjustments)
        
        # Create reflection result
        reflection_result = StructuredReflectionResult(
            reflection_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            trigger_type=trigger_type,
            reflection_timestamp=current_time,
            analysis_period_start=analysis_period_start,
            analysis_period_end=current_time,
            decisions_analyzed=len(decision_history),
            events_processed=len(event_history),
            performance_metrics=performance_metrics,
            insights=insights,
            critical_insights_count=len([i for i in insights if i.priority == "critical"]),
            policy_adjustments=policy_adjustments,
            high_priority_adjustments=len([a for a in policy_adjustments if a.implementation_urgency == "immediate"]),
            analysis_depth_score=analysis_depth_score,
            insight_novelty_score=insight_novelty_score,
            actionability_score=actionability_score
        )
        
        return reflection_result
    
    async def _gather_analysis_events(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Gather events for analysis from the specified time period."""
        # In a real implementation, this would query the memory system
        # For now, return recent events from major_events
        return [
            event for event in self.major_events[-20:]  # Last 20 events
            if start_time <= datetime.fromisoformat(event.get("timestamp", start_time.isoformat())) <= end_time
        ]
    
    async def _gather_decision_history(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Gather decision history for analysis."""
        return [
            decision for decision in self.decision_history[-50:]  # Last 50 decisions
            if start_time <= datetime.fromisoformat(decision.get("timestamp", start_time.isoformat())) <= end_time
        ]
    
    async def _calculate_performance_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Calculate performance metrics for the analysis period."""
        # Calculate basic performance metrics
        metrics = {
            "revenue_growth": 0.05,  # Placeholder
            "profit_margin": 0.15,
            "decision_success_rate": 0.75,
            "response_time": 2.5,
            "customer_satisfaction": 0.8,
            "overall_score": 0.7
        }
        
        # In a real implementation, this would calculate actual metrics
        return metrics
    
    async def _analyze_decision_category(self, decision_type: str, decisions: List[Dict[str, Any]],
                                        outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze effectiveness of decisions in a specific category."""
        if not decisions:
            return {"effectiveness": 0.0, "pattern": "insufficient_data"}
        
        # Calculate success rate for this decision type
        successful_decisions = len([d for d in decisions if d.get("outcome", "unknown") == "success"])
        success_rate = successful_decisions / len(decisions)
        
        analysis = {
            "total_decisions": len(decisions),
            "success_rate": success_rate,
            "effectiveness": success_rate,
            "average_impact": sum(d.get("impact_score", 0.5) for d in decisions) / len(decisions),
            "pattern": "improving" if success_rate > 0.7 else "needs_attention" if success_rate < 0.4 else "stable"
        }
        
        return analysis
    
    async def _identify_performance_patterns(self, event_history: List[Dict[str, Any]],
                                           outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in performance data."""
        patterns = {
            "trend": "stable",
            "cyclical_patterns": [],
            "performance_drivers": [],
            "risk_factors": []
        }
        
        # Analyze trends (simplified)
        if outcomes.get("overall_score", 0.5) > 0.7:
            patterns["trend"] = "improving"
        elif outcomes.get("overall_score", 0.5) < 0.4:
            patterns["trend"] = "declining"
        
        # Identify key performance drivers
        if outcomes.get("decision_success_rate", 0.5) > 0.8:
            patterns["performance_drivers"].append("strong_decision_making")
        
        if outcomes.get("response_time", 5.0) < 2.0:
            patterns["performance_drivers"].append("quick_response")
        
        return patterns
    
    async def _analyze_decision_failures(self, event_history: List[Dict[str, Any]],
                                        outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze failed decisions to identify improvement areas."""
        failed_decisions = [
            event for event in event_history
            if event.get("outcome") == "failure" or event.get("success", True) == False
        ]
        
        failure_analysis = {
            "total_failures": len(failed_decisions),
            "failure_rate": len(failed_decisions) / max(1, len(event_history)),
            "common_failure_types": {},
            "failure_patterns": [],
            "mitigation_strategies": []
        }
        
        # Categorize failure types
        for failure in failed_decisions:
            failure_type = failure.get("failure_type", "unknown")
            failure_analysis["common_failure_types"][failure_type] = \
                failure_analysis["common_failure_types"].get(failure_type, 0) + 1
        
        # Suggest mitigation strategies
        if "timing" in failure_analysis["common_failure_types"]:
            failure_analysis["mitigation_strategies"].append("improve_timing_analysis")
        
        if "market_conditions" in failure_analysis["common_failure_types"]:
            failure_analysis["mitigation_strategies"].append("enhance_market_monitoring")
        
        return failure_analysis
    
    async def _analyze_decision_successes(self, event_history: List[Dict[str, Any]],
                                         outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze successful decisions to identify success factors."""
        successful_decisions = [
            event for event in event_history
            if event.get("outcome") == "success" or event.get("success", False) == True
        ]
        
        success_analysis = {
            "total_successes": len(successful_decisions),
            "success_rate": len(successful_decisions) / max(1, len(event_history)),
            "success_factors": {},
            "high_impact_decisions": [],
            "replicable_strategies": []
        }
        
        # Identify success factors
        for success in successful_decisions:
            factors = success.get("success_factors", [])
            for factor in factors:
                success_analysis["success_factors"][factor] = \
                    success_analysis["success_factors"].get(factor, 0) + 1
        
        # Identify high-impact successful decisions
        high_impact = [d for d in successful_decisions if d.get("impact_score", 0) > 0.8]
        success_analysis["high_impact_decisions"] = [
            {"decision_id": d.get("decision_id"), "impact": d.get("impact_score")}
            for d in high_impact
        ]
        
        return success_analysis
    
    async def _generate_decision_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on decision analysis."""
        recommendations = []
        
        # Recommendations based on failure analysis
        failure_rate = analysis.get("failure_analysis", {}).get("failure_rate", 0)
        if failure_rate > 0.3:
            recommendations.append("improve_decision_validation_process")
            recommendations.append("implement_risk_assessment_before_decisions")
        
        # Recommendations based on success factors
        success_factors = analysis.get("success_factors", {})
        if "data_analysis" in success_factors:
            recommendations.append("increase_data_driven_decision_making")
        
        # Recommendations based on performance patterns
        trend = analysis.get("performance_patterns", {}).get("trend", "stable")
        if trend == "declining":
            recommendations.append("conduct_comprehensive_strategy_review")
            recommendations.append("implement_performance_monitoring_alerts")
        
        return recommendations
    
    async def _generate_pattern_insights(self, analysis_results: Dict[str, Any],
                                        current_time: datetime) -> List[ReflectionInsight]:
        """Generate insights from performance patterns."""
        insights = []
        patterns = analysis_results.get("performance_patterns", {})
        
        trend = patterns.get("trend", "stable")
        if trend == "declining":
            insights.append(ReflectionInsight(
                insight_id=str(uuid.uuid4()),
                category="performance",
                title="Performance Decline Detected",
                description="Analysis indicates declining performance trend requiring attention",
                evidence=[f"Performance trend: {trend}", "Multiple metrics showing downward movement"],
                confidence=0.8,
                actionability=0.9,
                priority="high",
                suggested_actions=["review_strategy", "analyze_root_causes", "implement_corrective_measures"],
                created_at=current_time
            ))
        
        return insights
    
    async def _generate_decision_insights(self, analysis_results: Dict[str, Any],
                                         current_time: datetime) -> List[ReflectionInsight]:
        """Generate insights from decision effectiveness analysis."""
        insights = []
        decision_effectiveness = analysis_results.get("decision_effectiveness", {})
        
        for decision_type, effectiveness in decision_effectiveness.items():
            success_rate = effectiveness.get("success_rate", 0.5)
            
            if success_rate < 0.4:
                insights.append(ReflectionInsight(
                    insight_id=str(uuid.uuid4()),
                    category="behavior",
                    title=f"Low Success Rate in {decision_type.title()} Decisions",
                    description=f"Success rate of {success_rate:.1%} for {decision_type} decisions is below acceptable threshold",
                    evidence=[f"Success rate: {success_rate:.1%}", f"Total decisions analyzed: {effectiveness.get('total_decisions', 0)}"],
                    confidence=0.85,
                    actionability=0.8,
                    priority="medium" if success_rate > 0.2 else "high",
                    suggested_actions=[f"review_{decision_type}_strategy", "improve_analysis_before_decisions", "seek_additional_data"],
                    created_at=current_time
                ))
        
        return insights
    
    async def _generate_failure_insights(self, analysis_results: Dict[str, Any],
                                        current_time: datetime) -> List[ReflectionInsight]:
        """Generate insights from failure analysis."""
        insights = []
        failure_analysis = analysis_results.get("failure_analysis", {})
        
        failure_rate = failure_analysis.get("failure_rate", 0)
        if failure_rate > 0.3:
            insights.append(ReflectionInsight(
                insight_id=str(uuid.uuid4()),
                category="risk",
                title="High Decision Failure Rate",
                description=f"Failure rate of {failure_rate:.1%} indicates systematic issues in decision-making process",
                evidence=[f"Failure rate: {failure_rate:.1%}", "Multiple failure types identified"],
                confidence=0.9,
                actionability=0.85,
                priority="critical",
                suggested_actions=["implement_decision_validation", "improve_risk_assessment", "enhance_data_analysis"],
                created_at=current_time
            ))
        
        return insights
    
    async def _generate_strategic_insights(self, analysis_results: Dict[str, Any],
                                          current_time: datetime) -> List[ReflectionInsight]:
        """Generate strategic-level insights."""
        insights = []
        
        # Analyze overall strategic direction
        recommendations = analysis_results.get("recommendations", [])
        if "conduct_comprehensive_strategy_review" in recommendations:
            insights.append(ReflectionInsight(
                insight_id=str(uuid.uuid4()),
                category="strategy",
                title="Strategic Review Required",
                description="Current performance patterns suggest need for comprehensive strategy review",
                evidence=["Declining performance trend", "Multiple decision categories underperforming"],
                confidence=0.75,
                actionability=0.7,
                priority="high",
                suggested_actions=["strategic_planning_session", "market_analysis", "competitive_assessment"],
                created_at=current_time
            ))
        
        return insights
    
    async def _rank_insights_by_priority(self, insights: List[ReflectionInsight]) -> List[ReflectionInsight]:
        """Rank insights by priority and actionability."""
        priority_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        
        def insight_score(insight):
            priority_score = priority_weights.get(insight.priority, 1)
            return priority_score * insight.confidence * insight.actionability
        
        return sorted(insights, key=insight_score, reverse=True)
    
    def _map_insight_to_policy_area(self, insight: ReflectionInsight) -> str:
        """Map an insight to appropriate policy area."""
        category_mapping = {
            "performance": "operations",
            "behavior": "operations",
            "strategy": "operations",
            "risk": "risk_management",
            "environment": "marketing"
        }
        
        return category_mapping.get(insight.category, "operations")
    
    async def _generate_policy_adjustments_for_area(self, policy_area: str,
                                                   insights: List[ReflectionInsight],
                                                   current_time: datetime) -> List[PolicyAdjustment]:
        """Generate policy adjustments for a specific policy area."""
        adjustments = []
        
        if policy_area == "operations" and any("success_rate" in insight.title.lower() for insight in insights):
            adjustments.append(PolicyAdjustment(
                adjustment_id=str(uuid.uuid4()),
                policy_area="operations",
                current_parameters={"decision_confidence_threshold": 0.5},
                recommended_parameters={"decision_confidence_threshold": 0.7},
                rationale="Increase decision confidence threshold to improve success rate",
                expected_impact={"decision_success_rate": 0.15},
                confidence=0.8,
                implementation_urgency="within_day",
                created_at=current_time
            ))
        
        if policy_area == "risk_management" and any("failure" in insight.title.lower() for insight in insights):
            adjustments.append(PolicyAdjustment(
                adjustment_id=str(uuid.uuid4()),
                policy_area="risk_management",
                current_parameters={"risk_assessment_enabled": False},
                recommended_parameters={"risk_assessment_enabled": True, "risk_threshold": 0.3},
                rationale="Enable risk assessment to reduce decision failures",
                expected_impact={"failure_rate": -0.2},
                confidence=0.85,
                implementation_urgency="immediate",
                created_at=current_time
            ))
        
        return adjustments
    
    async def _validate_policy_adjustments(self, adjustments: List[PolicyAdjustment]) -> List[PolicyAdjustment]:
        """Validate and filter policy adjustments."""
        validated = []
        
        for adjustment in adjustments:
            # Validate confidence threshold
            if adjustment.confidence >= 0.6:
                validated.append(adjustment)
            else:
                logger.warning(f"Policy adjustment {adjustment.adjustment_id} excluded due to low confidence: {adjustment.confidence}")
        
        return validated
    
    async def _calculate_analysis_depth_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate how deep and comprehensive the analysis was."""
        score = 0.0
        
        # Score based on different analysis components
        if "decision_effectiveness" in analysis:
            score += 0.3
        if "performance_patterns" in analysis:
            score += 0.3
        if "failure_analysis" in analysis:
            score += 0.2
        if "success_factors" in analysis:
            score += 0.2
        
        return min(1.0, score)
    
    async def _calculate_insight_novelty_score(self, insights: List[ReflectionInsight]) -> float:
        """Calculate how novel the generated insights are."""
        if not insights:
            return 0.0
        
        # Compare with historical insights to determine novelty
        # For now, use a simple heuristic based on insight diversity
        categories = set(insight.category for insight in insights)
        novelty_score = len(categories) / len(self.insight_categories)
        
        return min(1.0, novelty_score)
    
    async def _calculate_actionability_score(self, insights: List[ReflectionInsight],
                                           adjustments: List[PolicyAdjustment]) -> float:
        """Calculate how actionable the reflection results are."""
        if not insights:
            return 0.0
        
        # Average actionability of insights
        avg_actionability = sum(insight.actionability for insight in insights) / len(insights)
        
        # Bonus for having concrete policy adjustments
        adjustment_bonus = min(0.3, len(adjustments) * 0.1)
        
        return min(1.0, avg_actionability + adjustment_bonus)
    
    async def _publish_reflection_completed_event(self, reflection_result: StructuredReflectionResult):
        """Publish event when structured reflection is completed."""
        event_data = {
            "agent_id": self.agent_id,
            "event_type": "StructuredReflectionCompleted",
            "timestamp": reflection_result.reflection_timestamp.isoformat(),
            "reflection_id": reflection_result.reflection_id,
            "trigger_type": reflection_result.trigger_type.value,
            "insights_generated": len(reflection_result.insights),
            "critical_insights": reflection_result.critical_insights_count,
            "policy_adjustments": len(reflection_result.policy_adjustments),
            "reflection_quality": {
                "analysis_depth": reflection_result.analysis_depth_score,
                "insight_novelty": reflection_result.insight_novelty_score,
                "actionability": reflection_result.actionability_score
            }
        }
        
        try:
            await self.event_bus.publish(BaseEvent(
                event_id=str(uuid.uuid4()),
                timestamp=reflection_result.reflection_timestamp,
                **event_data
            ))
        except Exception as e:
            logger.error(f"Failed to publish structured reflection completed event: {e}")