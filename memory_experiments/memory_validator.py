"""
Memory Integration Validator

Provides comprehensive validation and consistency checking for agent memory systems.
Ensures memory integrity, detects contradictions, and validates action-memory alignment.
"""

import asyncio
import logging
import uuid
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .memory_config import MemoryConfig
from .dual_memory_manager import MemoryEvent, DualMemoryManager
from event_bus import EventBus, get_event_bus
from events import BaseEvent

logger = logging.getLogger(__name__)


class InconsistencyType(Enum):
    """Types of memory inconsistencies that can be detected."""
    CONTRADICTION = "contradiction"
    TEMPORAL_CONFLICT = "temporal_conflict"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    STALE_INFORMATION = "stale_information"
    MISSING_CONTEXT = "missing_context"
    CONFIDENCE_MISMATCH = "confidence_mismatch"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryInconsistency:
    """Represents a detected memory inconsistency."""
    inconsistency_id: str
    inconsistency_type: InconsistencyType
    severity: ValidationSeverity
    description: str
    conflicting_memories: List[str]  # Memory event IDs
    evidence: List[str]
    confidence: float  # Confidence in the inconsistency detection
    suggested_resolution: str
    detected_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "inconsistency_id": self.inconsistency_id,
            "inconsistency_type": self.inconsistency_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "conflicting_memories": self.conflicting_memories,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "suggested_resolution": self.suggested_resolution,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class ValidationResult:
    """Result of memory validation process."""
    validation_id: str
    agent_id: str
    validation_timestamp: datetime
    action_validated: Optional[Dict[str, Any]]
    memories_checked: int
    inconsistencies_found: List[MemoryInconsistency]
    validation_passed: bool
    confidence_score: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "agent_id": self.agent_id,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "action_validated": self.action_validated,
            "memories_checked": self.memories_checked,
            "inconsistencies_found": [inc.to_dict() for inc in self.inconsistencies_found],
            "validation_passed": self.validation_passed,
            "confidence_score": self.confidence_score,
            "recommendations": self.recommendations
        }


class MemoryValidator:
    """
    Backwards-compat wrapper expected by tests:
    from memory_experiments.memory_validator import MemoryValidator
    """
    def __init__(self, memory_manager: DualMemoryManager, config: Optional[MemoryConfig] = None, agent_id: str = "agent"):
        self._checker = MemoryConsistencyChecker(agent_id=agent_id, config=config or MemoryConfig())
        self._memory_manager = memory_manager

    async def validate_all_memory(self) -> bool:
        """Validate current memory contents and return pass/fail."""
        facts = await self._memory_manager.retrieve_recent_memories(limit=100) if hasattr(self._memory_manager, "retrieve_recent_memories") else []
        result = await self._checker.validate_memory_retrieval(facts, proposed_action={"type": "noop"})
        return bool(result.validation_passed)

    def get_statistics(self) -> Dict[str, Any]:
        """Expose validation statistics."""
        return self._checker.get_validation_statistics()


class MemoryConsistencyChecker:
    """
    Validates memory retrieval consistency and detects contradictions.
    
    Provides comprehensive checking of memory consistency, identifies conflicting
    information, and flags potential issues in agent memory systems.
    """
    
    def __init__(self, agent_id: str, config: MemoryConfig):
        """
        Initialize the Memory Consistency Checker.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Memory configuration
        """
        self.agent_id = agent_id
        self.config = config
        
        # Consistency checking parameters
        self.contradiction_threshold = 0.8  # Threshold for detecting contradictions
        self.temporal_window_hours = 24  # Time window for temporal consistency
        self.confidence_threshold = 0.6  # Minimum confidence for reliable memories
        
        # Tracking and statistics
        self.validation_history: List[ValidationResult] = []
        self.detected_inconsistencies: List[MemoryInconsistency] = []
        self.total_validations = 0
        self.total_inconsistencies_found = 0
        
        logger.info(f"MemoryConsistencyChecker initialized for agent {agent_id}")
    
    async def validate_memory_retrieval(self, retrieved_facts: List[MemoryEvent], 
                                       proposed_action: Dict[str, Any]) -> ValidationResult:
        """
        Check consistency between retrieved facts and proposed action.
        
        Args:
            retrieved_facts: List of retrieved memory events
            proposed_action: Action that the agent wants to take
            
        Returns:
            ValidationResult with consistency analysis
        """
        logger.info(f"Validating memory retrieval for agent {self.agent_id} - "
                   f"{len(retrieved_facts)} facts, action: {proposed_action.get('type', 'unknown')}")
        
        validation_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        inconsistencies = []
        
        # Check for contradictions within retrieved facts
        fact_contradictions = await self.detect_contradictions(retrieved_facts)
        inconsistencies.extend(fact_contradictions)
        
        # Check action-memory consistency
        action_inconsistencies = await self.flag_inconsistencies(proposed_action, retrieved_facts)
        inconsistencies.extend(action_inconsistencies)
        
        # Check temporal consistency
        temporal_inconsistencies = await self._check_temporal_consistency(retrieved_facts)
        inconsistencies.extend(temporal_inconsistencies)
        
        # Check confidence consistency
        confidence_inconsistencies = await self._check_confidence_consistency(retrieved_facts)
        inconsistencies.extend(confidence_inconsistencies)
        
        # Determine overall validation result
        critical_inconsistencies = [inc for inc in inconsistencies 
                                  if inc.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]]
        validation_passed = len(critical_inconsistencies) == 0
        
        # Calculate confidence score
        confidence_score = await self._calculate_validation_confidence(
            retrieved_facts, inconsistencies, proposed_action
        )
        
        # Generate recommendations
        recommendations = await self._generate_validation_recommendations(
            inconsistencies, proposed_action
        )
        
        # Create validation result
        validation_result = ValidationResult(
            validation_id=validation_id,
            agent_id=self.agent_id,
            validation_timestamp=current_time,
            action_validated=proposed_action,
            memories_checked=len(retrieved_facts),
            inconsistencies_found=inconsistencies,
            validation_passed=validation_passed,
            confidence_score=confidence_score,
            recommendations=recommendations
        )
        
        # Update statistics
        self.validation_history.append(validation_result)
        self.detected_inconsistencies.extend(inconsistencies)
        self.total_validations += 1
        self.total_inconsistencies_found += len(inconsistencies)
        
        logger.info(f"Memory validation completed - passed: {validation_passed}, "
                   f"inconsistencies: {len(inconsistencies)}, confidence: {confidence_score:.2f}")
        
        return validation_result
    
    async def detect_contradictions(self, fact_set: List[MemoryEvent]) -> List[MemoryInconsistency]:
        """
        Identify conflicting information within a set of facts.
        
        Args:
            fact_set: List of memory events to check for contradictions
            
        Returns:
            List of detected inconsistencies
        """
        logger.debug(f"Detecting contradictions in {len(fact_set)} facts")
        
        inconsistencies = []
        current_time = datetime.now()
        
        # Group facts by domain for targeted contradiction checking
        facts_by_domain = {}
        for fact in fact_set:
            domain = fact.domain
            if domain not in facts_by_domain:
                facts_by_domain[domain] = []
            facts_by_domain[domain].append(fact)
        
        # Check for contradictions within each domain
        for domain, domain_facts in facts_by_domain.items():
            domain_contradictions = await self._detect_domain_contradictions(domain, domain_facts)
            inconsistencies.extend(domain_contradictions)
        
        # Check for cross-domain contradictions
        cross_domain_contradictions = await self._detect_cross_domain_contradictions(facts_by_domain)
        inconsistencies.extend(cross_domain_contradictions)
        
        # Check for temporal contradictions
        temporal_contradictions = await self._detect_temporal_contradictions(fact_set)
        inconsistencies.extend(temporal_contradictions)
        
        logger.debug(f"Found {len(inconsistencies)} contradictions")
        return inconsistencies
    
    async def flag_inconsistencies(self, agent_action: Dict[str, Any], 
                                  known_facts: List[MemoryEvent]) -> List[MemoryInconsistency]:
        """
        Alert on discrepancies between agent action and known facts.
        
        Args:
            agent_action: Proposed action by the agent
            known_facts: Known facts from memory
            
        Returns:
            List of detected inconsistencies
        """
        logger.debug(f"Flagging inconsistencies for action: {agent_action.get('type', 'unknown')}")
        
        inconsistencies = []
        current_time = datetime.now()
        
        action_type = agent_action.get("type", "unknown")
        action_parameters = agent_action.get("parameters", {})
        
        # Check pricing action consistency
        if action_type == "set_price":
            price_inconsistencies = await self._check_pricing_action_consistency(
                action_parameters, known_facts
            )
            inconsistencies.extend(price_inconsistencies)
        
        # Check inventory action consistency
        elif action_type == "place_order":
            inventory_inconsistencies = await self._check_inventory_action_consistency(
                action_parameters, known_facts
            )
            inconsistencies.extend(inventory_inconsistencies)
        
        # Check marketing action consistency
        elif action_type == "run_marketing_campaign":
            marketing_inconsistencies = await self._check_marketing_action_consistency(
                action_parameters, known_facts
            )
            inconsistencies.extend(marketing_inconsistencies)
        
        # Check general action-memory alignment
        general_inconsistencies = await self._check_general_action_consistency(
            agent_action, known_facts
        )
        inconsistencies.extend(general_inconsistencies)
        
        logger.debug(f"Found {len(inconsistencies)} action inconsistencies")
        return inconsistencies
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        current_time = datetime.now()
        
        stats = {
            "agent_id": self.agent_id,
            "total_validations": self.total_validations,
            "total_inconsistencies": self.total_inconsistencies_found,
            "validation_history_size": len(self.validation_history),
            "configuration": {
                "contradiction_threshold": self.contradiction_threshold,
                "temporal_window_hours": self.temporal_window_hours,
                "confidence_threshold": self.confidence_threshold
            }
        }
        
        if self.validation_history:
            # Calculate success rates
            passed_validations = len([v for v in self.validation_history if v.validation_passed])
            stats["validation_success_rate"] = passed_validations / len(self.validation_history)
            
            # Calculate average confidence
            avg_confidence = sum(v.confidence_score for v in self.validation_history) / len(self.validation_history)
            stats["average_confidence_score"] = avg_confidence
            
            # Recent validation summary
            recent_validation = self.validation_history[-1]
            stats["latest_validation"] = {
                "timestamp": recent_validation.validation_timestamp.isoformat(),
                "passed": recent_validation.validation_passed,
                "inconsistencies": len(recent_validation.inconsistencies_found),
                "confidence": recent_validation.confidence_score
            }
        
        if self.detected_inconsistencies:
            # Inconsistency type distribution
            type_counts = {}
            severity_counts = {}
            
            for inconsistency in self.detected_inconsistencies:
                inc_type = inconsistency.inconsistency_type.value
                severity = inconsistency.severity.value
                
                type_counts[inc_type] = type_counts.get(inc_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            stats["inconsistency_types"] = type_counts
            stats["inconsistency_severities"] = severity_counts
        
        return stats
    
    # Private helper methods
    
    async def _detect_domain_contradictions(self, domain: str, 
                                           domain_facts: List[MemoryEvent]) -> List[MemoryInconsistency]:
        """Detect contradictions within a specific domain."""
        inconsistencies = []
        
        if domain == "pricing":
            # Check for conflicting price information
            price_facts = [f for f in domain_facts if "price" in f.content.lower()]
            if len(price_facts) >= 2:
                # Simple contradiction detection for prices
                for i, fact1 in enumerate(price_facts):
                    for fact2 in price_facts[i+1:]:
                        if await self._are_pricing_facts_contradictory(fact1, fact2):
                            inconsistencies.append(MemoryInconsistency(
                                inconsistency_id=str(uuid.uuid4()),
                                inconsistency_type=InconsistencyType.CONTRADICTION,
                                severity=ValidationSeverity.MEDIUM,
                                description=f"Contradictory pricing information detected in {domain} domain",
                                conflicting_memories=[fact1.event_id, fact2.event_id],
                                evidence=[f"Fact 1: {fact1.content[:100]}", f"Fact 2: {fact2.content[:100]}"],
                                confidence=0.8,
                                suggested_resolution="review_and_reconcile_pricing_facts",
                                detected_at=datetime.now()
                            ))
        
        elif domain == "inventory":
            # Check for conflicting inventory information
            inventory_facts = [f for f in domain_facts if "inventory" in f.content.lower() or "stock" in f.content.lower()]
            if len(inventory_facts) >= 2:
                # Check for contradictory inventory levels
                for i, fact1 in enumerate(inventory_facts):
                    for fact2 in inventory_facts[i+1:]:
                        if await self._are_inventory_facts_contradictory(fact1, fact2):
                            inconsistencies.append(MemoryInconsistency(
                                inconsistency_id=str(uuid.uuid4()),
                                inconsistency_type=InconsistencyType.CONTRADICTION,
                                severity=ValidationSeverity.HIGH,
                                description=f"Contradictory inventory information detected",
                                conflicting_memories=[fact1.event_id, fact2.event_id],
                                evidence=[f"Fact 1: {fact1.content[:100]}", f"Fact 2: {fact2.content[:100]}"],
                                confidence=0.75,
                                suggested_resolution="verify_current_inventory_levels",
                                detected_at=datetime.now()
                            ))
        
        return inconsistencies
    
    async def _detect_cross_domain_contradictions(self, facts_by_domain: Dict[str, List[MemoryEvent]]) -> List[MemoryInconsistency]:
        """Detect contradictions across different domains."""
        inconsistencies = []
        
        # Check for contradictions between pricing and sales domains
        if "pricing" in facts_by_domain and "sales" in facts_by_domain:
            pricing_facts = facts_by_domain["pricing"]
            sales_facts = facts_by_domain["sales"]
            
            # Look for pricing strategies that contradict sales performance
            for price_fact in pricing_facts:
                for sales_fact in sales_facts:
                    if await self._are_pricing_sales_contradictory(price_fact, sales_fact):
                        inconsistencies.append(MemoryInconsistency(
                            inconsistency_id=str(uuid.uuid4()),
                            inconsistency_type=InconsistencyType.LOGICAL_INCONSISTENCY,
                            severity=ValidationSeverity.MEDIUM,
                            description="Pricing strategy contradicts sales performance data",
                            conflicting_memories=[price_fact.event_id, sales_fact.event_id],
                            evidence=[f"Pricing: {price_fact.content[:100]}", f"Sales: {sales_fact.content[:100]}"],
                            confidence=0.6,
                            suggested_resolution="review_pricing_strategy_effectiveness",
                            detected_at=datetime.now()
                        ))
        
        return inconsistencies
    
    async def _detect_temporal_contradictions(self, fact_set: List[MemoryEvent]) -> List[MemoryInconsistency]:
        """Detect temporal contradictions in facts."""
        inconsistencies = []
        
        # Sort facts by timestamp
        sorted_facts = sorted(fact_set, key=lambda f: f.timestamp)
        
        for i, fact in enumerate(sorted_facts[:-1]):
            next_fact = sorted_facts[i+1]
            
            # Check if newer fact contradicts older fact in implausible way
            if await self._are_temporally_contradictory(fact, next_fact):
                inconsistencies.append(MemoryInconsistency(
                    inconsistency_id=str(uuid.uuid4()),
                    inconsistency_type=InconsistencyType.TEMPORAL_CONFLICT,
                    severity=ValidationSeverity.MEDIUM,
                    description="Temporal contradiction detected between sequential facts",
                    conflicting_memories=[fact.event_id, next_fact.event_id],
                    evidence=[f"Earlier: {fact.content[:100]}", f"Later: {next_fact.content[:100]}"],
                    confidence=0.7,
                    suggested_resolution="verify_temporal_consistency",
                    detected_at=datetime.now()
                ))
        
        return inconsistencies
    
    async def _check_temporal_consistency(self, retrieved_facts: List[MemoryEvent]) -> List[MemoryInconsistency]:
        """Check for temporal consistency issues."""
        inconsistencies = []
        current_time = datetime.now()
        
        # Check for stale information
        for fact in retrieved_facts:
            age_hours = (current_time - fact.timestamp).total_seconds() / 3600
            
            if age_hours > self.temporal_window_hours * 2:  # Double the normal window
                # Check if this is time-sensitive information
                if await self._is_time_sensitive_information(fact):
                    inconsistencies.append(MemoryInconsistency(
                        inconsistency_id=str(uuid.uuid4()),
                        inconsistency_type=InconsistencyType.STALE_INFORMATION,
                        severity=ValidationSeverity.MEDIUM,
                        description=f"Potentially stale time-sensitive information (age: {age_hours:.1f} hours)",
                        conflicting_memories=[fact.event_id],
                        evidence=[f"Fact age: {age_hours:.1f} hours", f"Content: {fact.content[:100]}"],
                        confidence=0.6,
                        suggested_resolution="refresh_time_sensitive_information",
                        detected_at=current_time
                    ))
        
        return inconsistencies
    
    async def _check_confidence_consistency(self, retrieved_facts: List[MemoryEvent]) -> List[MemoryInconsistency]:
        """Check for confidence-related inconsistencies."""
        inconsistencies = []
        
        # Check for low confidence facts being used for critical decisions
        low_confidence_facts = [f for f in retrieved_facts if f.importance_score < self.confidence_threshold]
        
        if low_confidence_facts:
            inconsistencies.append(MemoryInconsistency(
                inconsistency_id=str(uuid.uuid4()),
                inconsistency_type=InconsistencyType.CONFIDENCE_MISMATCH,
                severity=ValidationSeverity.LOW,
                description=f"Using {len(low_confidence_facts)} low-confidence facts for decision",
                conflicting_memories=[f.event_id for f in low_confidence_facts],
                evidence=[f"Low confidence facts count: {len(low_confidence_facts)}"],
                confidence=0.8,
                suggested_resolution="seek_higher_confidence_information",
                detected_at=datetime.now()
            ))
        
        return inconsistencies
    
    async def _check_pricing_action_consistency(self, action_parameters: Dict[str, Any], 
                                               known_facts: List[MemoryEvent]) -> List[MemoryInconsistency]:
        """Check consistency of pricing actions with known facts."""
        inconsistencies = []
        
        proposed_price = action_parameters.get("price", 0)
        if proposed_price == 0:
            return inconsistencies
        
        # Check against recent pricing facts
        pricing_facts = [f for f in known_facts if "price" in f.content.lower()]
        
        for fact in pricing_facts:
            # Simple check for dramatic price changes without justification
            if "current price" in fact.content.lower():
                # Extract price from fact content (simplified)
                # In a real implementation, this would use more sophisticated parsing
                if await self._is_dramatic_price_change(proposed_price, fact):
                    inconsistencies.append(MemoryInconsistency(
                        inconsistency_id=str(uuid.uuid4()),
                        inconsistency_type=InconsistencyType.LOGICAL_INCONSISTENCY,
                        severity=ValidationSeverity.MEDIUM,
                        description=f"Proposed price change appears dramatic without clear justification",
                        conflicting_memories=[fact.event_id],
                        evidence=[f"Proposed price: {proposed_price}", f"Historical context: {fact.content[:100]}"],
                        confidence=0.7,
                        suggested_resolution="review_price_change_justification",
                        detected_at=datetime.now()
                    ))
        
        return inconsistencies
    
    async def _check_inventory_action_consistency(self, action_parameters: Dict[str, Any], 
                                                 known_facts: List[MemoryEvent]) -> List[MemoryInconsistency]:
        """Check consistency of inventory actions with known facts."""
        inconsistencies = []
        
        order_quantity = action_parameters.get("quantity", 0)
        
        # Check against inventory facts
        inventory_facts = [f for f in known_facts if "inventory" in f.content.lower()]
        
        for fact in inventory_facts:
            # Check for ordering when inventory is already high
            if "high inventory" in fact.content.lower() and order_quantity > 50:
                inconsistencies.append(MemoryInconsistency(
                    inconsistency_id=str(uuid.uuid4()),
                    inconsistency_type=InconsistencyType.LOGICAL_INCONSISTENCY,
                    severity=ValidationSeverity.MEDIUM,
                    description="Large order proposed despite high inventory levels",
                    conflicting_memories=[fact.event_id],
                    evidence=[f"Order quantity: {order_quantity}", f"Inventory status: {fact.content[:100]}"],
                    confidence=0.75,
                    suggested_resolution="review_inventory_levels_before_ordering",
                    detected_at=datetime.now()
                ))
        
        return inconsistencies
    
    async def _check_marketing_action_consistency(self, action_parameters: Dict[str, Any], 
                                                 known_facts: List[MemoryEvent]) -> List[MemoryInconsistency]:
        """Check consistency of marketing actions with known facts."""
        inconsistencies = []
        
        campaign_budget = action_parameters.get("budget", 0)
        
        # Check against financial facts
        financial_facts = [f for f in known_facts if "budget" in f.content.lower() or "financial" in f.content.lower()]
        
        for fact in financial_facts:
            # Check for high marketing spend when finances are tight
            if "tight budget" in fact.content.lower() and campaign_budget > 500:
                inconsistencies.append(MemoryInconsistency(
                    inconsistency_id=str(uuid.uuid4()),
                    inconsistency_type=InconsistencyType.LOGICAL_INCONSISTENCY,
                    severity=ValidationSeverity.HIGH,
                    description="High marketing budget proposed despite tight financial constraints",
                    conflicting_memories=[fact.event_id],
                    evidence=[f"Campaign budget: {campaign_budget}", f"Financial context: {fact.content[:100]}"],
                    confidence=0.8,
                    suggested_resolution="adjust_budget_considering_constraints",
                    detected_at=datetime.now()
                ))
        
        return inconsistencies
    
    async def _check_general_action_consistency(self, agent_action: Dict[str, Any], 
                                               known_facts: List[MemoryEvent]) -> List[MemoryInconsistency]:
        """Check general consistency between action and memory context."""
        inconsistencies = []
        
        action_type = agent_action.get("type", "unknown")
        expected_impact = agent_action.get("expected_impact", {})
        
        # Check if action aligns with recent context
        contextual_facts = [f for f in known_facts if f.timestamp > datetime.now() - timedelta(hours=6)]
        
        if contextual_facts:
            # Look for context that might contraindicate the action
            contradiction_found = False
            for fact in contextual_facts:
                if await self._does_context_contraindicate_action(fact, agent_action):
                    contradiction_found = True
                    break
            
            if contradiction_found:
                inconsistencies.append(MemoryInconsistency(
                    inconsistency_id=str(uuid.uuid4()),
                    inconsistency_type=InconsistencyType.MISSING_CONTEXT,
                    severity=ValidationSeverity.MEDIUM,
                    description=f"Action {action_type} may not align with recent context",
                    conflicting_memories=[f.event_id for f in contextual_facts[:3]],  # Limit to first 3
                    evidence=[f"Action: {action_type}", "Recent context suggests different approach"],
                    confidence=0.6,
                    suggested_resolution="review_recent_context_before_action",
                    detected_at=datetime.now()
                ))
        
        return inconsistencies
    
    async def _calculate_validation_confidence(self, retrieved_facts: List[MemoryEvent], 
                                              inconsistencies: List[MemoryInconsistency], 
                                              proposed_action: Dict[str, Any]) -> float:
        """Calculate confidence score for the validation."""
        base_confidence = 1.0
        
        # Reduce confidence based on inconsistencies
        for inconsistency in inconsistencies:
            severity_penalty = {
                ValidationSeverity.LOW: 0.05,
                ValidationSeverity.MEDIUM: 0.15,
                ValidationSeverity.HIGH: 0.3,
                ValidationSeverity.CRITICAL: 0.5
            }
            base_confidence -= severity_penalty.get(inconsistency.severity, 0.1)
        
        # Adjust confidence based on fact quality
        if retrieved_facts:
            avg_fact_confidence = sum(f.importance_score for f in retrieved_facts) / len(retrieved_facts)
            base_confidence *= avg_fact_confidence
        
        return max(0.0, min(1.0, base_confidence))
    
    async def _generate_validation_recommendations(self, inconsistencies: List[MemoryInconsistency], 
                                                  proposed_action: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not inconsistencies:
            recommendations.append("validation_passed_proceed_with_confidence")
            return recommendations
        
        # Group recommendations by inconsistency type
        contradiction_count = len([i for i in inconsistencies if i.inconsistency_type == InconsistencyType.CONTRADICTION])
        temporal_count = len([i for i in inconsistencies if i.inconsistency_type == InconsistencyType.TEMPORAL_CONFLICT])
        logical_count = len([i for i in inconsistencies if i.inconsistency_type == InconsistencyType.LOGICAL_INCONSISTENCY])
        
        if contradiction_count > 0:
            recommendations.append("resolve_memory_contradictions_before_proceeding")
        
        if temporal_count > 0:
            recommendations.append("refresh_time_sensitive_information")
        
        if logical_count > 0:
            recommendations.append("review_action_logic_and_context")
        
        # Critical severity recommendations
        critical_inconsistencies = [i for i in inconsistencies if i.severity == ValidationSeverity.CRITICAL]
        if critical_inconsistencies:
            recommendations.append("halt_action_due_to_critical_inconsistencies")
        
        return recommendations
    
    # Simplified helper methods for demonstration
    
    async def _are_pricing_facts_contradictory(self, fact1: MemoryEvent, fact2: MemoryEvent) -> bool:
        """Check if two pricing facts contradict each other."""
        # Simplified logic - in reality would parse and compare actual price values
        content1_lower = fact1.content.lower()
        content2_lower = fact2.content.lower()
        
        # Simple contradiction detection
        if "price increase" in content1_lower and "price decrease" in content2_lower:
            return True
        if "expensive" in content1_lower and "cheap" in content2_lower:
            return True
        
        return False
    
    async def _are_inventory_facts_contradictory(self, fact1: MemoryEvent, fact2: MemoryEvent) -> bool:
        """Check if two inventory facts contradict each other."""
        content1_lower = fact1.content.lower()
        content2_lower = fact2.content.lower()
        
        if "high inventory" in content1_lower and "low inventory" in content2_lower:
            return True
        if "out of stock" in content1_lower and "plenty in stock" in content2_lower:
            return True
        
        return False
    
    async def _are_pricing_sales_contradictory(self, price_fact: MemoryEvent, sales_fact: MemoryEvent) -> bool:
        """Check if pricing and sales facts contradict each other."""
        price_content = price_fact.content.lower()
        sales_content = sales_fact.content.lower()
        
        # Check for obvious contradictions
        if "price increase" in price_content and "sales increased" in sales_content:
            # This might actually be contradictory in some contexts
            return True
        
        return False
    
    async def _are_temporally_contradictory(self, earlier_fact: MemoryEvent, later_fact: MemoryEvent) -> bool:
        """Check if two facts are temporally contradictory."""
        # Check if the later fact contradicts the earlier fact in an implausible way
        earlier_content = earlier_fact.content.lower()
        later_content = later_fact.content.lower()
        
        # Time difference
        time_diff = (later_fact.timestamp - earlier_fact.timestamp).total_seconds() / 3600
        
        # If facts are very close in time but completely contradictory, flag it
        if time_diff < 1 and ("good" in earlier_content and "bad" in later_content):
            return True
        
        return False
    
    async def _is_time_sensitive_information(self, fact: MemoryEvent) -> bool:
        """Check if information is time-sensitive."""
        time_sensitive_keywords = ["price", "inventory", "stock", "demand", "market", "competitor"]
        content_lower = fact.content.lower()
        
        return any(keyword in content_lower for keyword in time_sensitive_keywords)
    
    async def _is_dramatic_price_change(self, proposed_price: float, fact: MemoryEvent) -> bool:
        """Check if proposed price represents a dramatic change."""
        # Simplified logic - would need actual price parsing in reality
        return abs(proposed_price - 100) > 50  # Placeholder logic
    
    async def _does_context_contraindicate_action(self, fact: MemoryEvent, action: Dict[str, Any]) -> bool:
        """Check if context contradicts the proposed action."""
        action_type = action.get("type", "")
        fact_content = fact.content.lower()
        
        # Simple heuristics
        if action_type == "run_marketing_campaign" and "budget constraints" in fact_content:
            return True
        
        if action_type == "place_order" and "high inventory" in fact_content:
            return True
        
        return False


class MemoryIntegrationGateway:
    """
    Gateway that intercepts agent actions for validation.
    
    Provides pre-action validation and post-action learning to ensure
    memory-action consistency and continuous improvement.
    """
    
    def __init__(self, agent_id: str, memory_manager: DualMemoryManager, 
                 consistency_checker: MemoryConsistencyChecker, 
                 config: MemoryConfig, event_bus: Optional[EventBus] = None):
        """
        Initialize the Memory Integration Gateway.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_manager: Memory management system
            consistency_checker: Memory consistency checker
            config: Memory configuration
            event_bus: Event bus for publishing validation events
        """
        self.agent_id = agent_id
        self.memory_manager = memory_manager
        self.consistency_checker = consistency_checker
        self.config = config
        self.event_bus = event_bus or get_event_bus()
        
        # Gateway statistics
        self.validations_performed = 0
        self.validations_passed = 0
        self.validations_failed = 0
        self.actions_blocked = 0
        self.learning_events = 0
        
        # Gateway configuration
        self.validation_enabled = True
        self.blocking_mode = False  # If True, blocks actions on validation failure
        self.learning_enabled = True
        
        logger.info(f"MemoryIntegrationGateway initialized for agent {agent_id}")
    
    async def pre_action_validation(self, action: Dict[str, Any], 
                                   agent_memory: Optional[DualMemoryManager] = None) -> Tuple[bool, ValidationResult]:
        """
        Validate action before execution against agent memory.
        
        Args:
            action: Proposed action to validate
            agent_memory: Optional specific memory manager (uses default if None)
            
        Returns:
            Tuple of (should_proceed, validation_result)
        """
        logger.info(f"Pre-action validation for agent {self.agent_id} - action: {action.get('type', 'unknown')}")
        
        if not self.validation_enabled:
            # Create a simple passed validation result
            dummy_result = ValidationResult(
                validation_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                validation_timestamp=datetime.now(),
                action_validated=action,
                memories_checked=0,
                inconsistencies_found=[],
                validation_passed=True,
                confidence_score=1.0,
                recommendations=["validation_disabled"]
            )
            return True, dummy_result
        
        memory_mgr = agent_memory or self.memory_manager
        
        # Retrieve relevant memories for validation
        relevant_memories = await self._retrieve_relevant_memories(action, memory_mgr)
        
        # Perform validation
        validation_result = await self.consistency_checker.validate_memory_retrieval(
            relevant_memories, action
        )
        
        # Update statistics
        self.validations_performed += 1
        if validation_result.validation_passed:
            self.validations_passed += 1
        else:
            self.validations_failed += 1
        
        # Determine if action should proceed
        should_proceed = validation_result.validation_passed
        
        if not should_proceed and self.blocking_mode:
            self.actions_blocked += 1
            logger.warning(f"Action blocked due to validation failure: {action.get('type', 'unknown')}")
        elif not should_proceed:
            logger.warning(f"Action validation failed but proceeding (non-blocking mode): {action.get('type', 'unknown')}")
            should_proceed = True  # Allow action in non-blocking mode
        
        # Publish validation event
        await self._publish_validation_event(validation_result, should_proceed)
        
        logger.info(f"Pre-action validation completed - proceed: {should_proceed}, "
                   f"confidence: {validation_result.confidence_score:.2f}")
        
        return should_proceed, validation_result
    
    async def post_action_learning(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> bool:
        """
        Update memory system based on action results.
        
        Args:
            action: Action that was executed
            outcome: Results and outcomes from the action
            
        Returns:
            True if learning was successful
        """
        logger.info(f"Post-action learning for agent {self.agent_id} - action: {action.get('type', 'unknown')}")
        
        if not self.learning_enabled:
            return True
        
        try:
            # Create learning event from action and outcome
            learning_event = await self._create_learning_event(action, outcome)
            
            # Store learning event in memory
            domain = self._determine_action_domain(action)
            success = await self.memory_manager.store_event(learning_event, domain)
            
            if success:
                self.learning_events += 1
                
                # Update consistency checker with new information
                await self._update_consistency_patterns(action, outcome)
                
                # Publish learning event
                await self._publish_learning_event(action, outcome)
                
                logger.info("Post-action learning completed successfully")
                return True
            else:
                logger.error("Failed to store learning event in memory")
                return False
        
        except Exception as e:
            logger.error(f"Error during post-action learning: {e}")
            return False
    
    def get_gateway_status(self) -> Dict[str, Any]:
        """Get comprehensive gateway status and statistics."""
        total_validations = self.validations_performed
        success_rate = self.validations_passed / max(1, total_validations)
        
        status = {
            "agent_id": self.agent_id,
            "validation_enabled": self.validation_enabled,
            "blocking_mode": self.blocking_mode,
            "learning_enabled": self.learning_enabled,
            "statistics": {
                "validations_performed": self.validations_performed,
                "validations_passed": self.validations_passed,
                "validations_failed": self.validations_failed,
                "validation_success_rate": success_rate,
                "actions_blocked": self.actions_blocked,
                "learning_events": self.learning_events
            },
            "consistency_checker_stats": self.consistency_checker.get_validation_statistics()
        }
        
        return status
    
    def configure_gateway(self, validation_enabled: bool = None, 
                         blocking_mode: bool = None, 
                         learning_enabled: bool = None):
        """Configure gateway behavior."""
        if validation_enabled is not None:
            self.validation_enabled = validation_enabled
            logger.info(f"Validation {'enabled' if validation_enabled else 'disabled'}")
        
        if blocking_mode is not None:
            self.blocking_mode = blocking_mode
            logger.info(f"Blocking mode {'enabled' if blocking_mode else 'disabled'}")
        
        if learning_enabled is not None:
            self.learning_enabled = learning_enabled
            logger.info(f"Learning {'enabled' if learning_enabled else 'disabled'}")
    
    # Private helper methods
    
    async def _retrieve_relevant_memories(self, action: Dict[str, Any], 
                                         memory_mgr: DualMemoryManager) -> List[MemoryEvent]:
        """Retrieve memories relevant to the proposed action."""
        action_type = action.get("type", "unknown")
        
        # Create query based on action type
        if action_type == "set_price":
            query = "price pricing competitor market"
        elif action_type == "place_order":
            query = "inventory stock supplier order"
        elif action_type == "run_marketing_campaign":
            query = "marketing campaign budget promotion"
        else:
            query = action_type
        
        # Retrieve relevant memories
        relevant_memories = await memory_mgr.retrieve_memories(query, max_memories=10)
        
        return relevant_memories
    
    async def _create_learning_event(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> BaseEvent:
        """Create a learning event from action and outcome."""
        action_type = action.get("type", "unknown")
        success = outcome.get("success", False)
        impact = outcome.get("impact", {})
        
        learning_content = f"Action: {action_type}, Success: {success}, Impact: {impact}"
        
        # Create a mock event for learning
        learning_event = BaseEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now()
        )
        
        # Add learning-specific attributes
        learning_event.content = learning_content
        learning_event.action_type = action_type
        learning_event.success = success
        learning_event.impact = impact
        
        return learning_event
    
    def _determine_action_domain(self, action: Dict[str, Any]) -> str:
        """Determine memory domain for an action."""
        action_type = action.get("type", "unknown")
        
        domain_mapping = {
            "set_price": "pricing",
            "place_order": "inventory", 
            "run_marketing_campaign": "marketing",
            "respond_to_customer": "customer_service"
        }
        
        return domain_mapping.get(action_type, "general")
    
    async def _update_consistency_patterns(self, action: Dict[str, Any], outcome: Dict[str, Any]):
        """Update consistency checker with new patterns learned."""
        # Update consistency checker's understanding based on action outcomes
        # This would involve updating its internal patterns and thresholds
        
        action_type = action.get("type", "unknown")
        success = outcome.get("success", False)
        
        # Simple learning: adjust thresholds based on outcomes
        if not success:
            # If action failed, potentially tighten validation
            self.consistency_checker.contradiction_threshold *= 0.95
        else:
            # If action succeeded, potentially relax slightly
            self.consistency_checker.contradiction_threshold *= 1.01
        
        # Keep thresholds within reasonable bounds
        self.consistency_checker.contradiction_threshold = max(0.5, min(0.95, self.consistency_checker.contradiction_threshold))
    
    async def _publish_validation_event(self, validation_result: ValidationResult, should_proceed: bool):
        """Publish validation event."""
        event_data = {
            "agent_id": self.agent_id,
            "event_type": "MemoryValidationCompleted",
            "timestamp": validation_result.validation_timestamp.isoformat(),
            "validation_id": validation_result.validation_id,
            "validation_passed": validation_result.validation_passed,
            "should_proceed": should_proceed,
            "inconsistencies_count": len(validation_result.inconsistencies_found),
            "confidence_score": validation_result.confidence_score,
            "recommendations": validation_result.recommendations
        }
        
        try:
            await self.event_bus.publish(BaseEvent(
                event_id=str(uuid.uuid4()),
                timestamp=validation_result.validation_timestamp,
                **event_data
            ))
        except Exception as e:
            logger.error(f"Failed to publish validation event: {e}")
    
    async def _publish_learning_event(self, action: Dict[str, Any], outcome: Dict[str, Any]):
        """Publish learning event."""
        event_data = {
            "agent_id": self.agent_id,
            "event_type": "MemoryLearningCompleted",
            "timestamp": datetime.now().isoformat(),
            "action_type": action.get("type", "unknown"),
            "learning_success": outcome.get("success", False),
            "impact_metrics": outcome.get("impact", {})
        }
        
        try:
            await self.event_bus.publish(BaseEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                **event_data
            ))
        except Exception as e:
            logger.error(f"Failed to publish learning event: {e}")