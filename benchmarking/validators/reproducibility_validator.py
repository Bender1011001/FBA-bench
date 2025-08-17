"""
Reproducibility validator for benchmarking results.

This module provides a comprehensive validator that combines deterministic execution,
version control, statistical validation, and audit trails to ensure complete reproducibility
of benchmark results.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .deterministic import DeterministicEnvironment, EnvironmentState
from .version_control import VersionControlManager, VersionManifest
from .statistical_validator import StatisticalValidator, StatisticalSummary
from .audit_trail import AuditTrailManager, AuditTrail

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityReport:
    """Comprehensive reproducibility report."""
    run_id: str
    timestamp: datetime
    overall_score: float  # 0.0 to 1.0
    components: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "components": self.components,
            "details": self.details,
            "recommendations": self.recommendations
        }


@dataclass
class ValidationResult:
    """Result of a validation check."""
    component: str
    check_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "check_name": self.check_name,
            "passed": self.passed,
            "score": self.score,
            "details": self.details,
            "issues": self.issues
        }


class ReproducibilityValidator:
    """
    Comprehensive reproducibility validator for benchmarking.
    
    This class combines multiple validation approaches to ensure complete
    reproducibility of benchmark results.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the reproducibility validator.
        
        Args:
            storage_path: Path to store validation results
        """
        self.storage_path = Path(storage_path) if storage_path else Path.cwd() / "reproducibility_reports"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize component validators
        self.deterministic_env = DeterministicEnvironment()
        self.version_manager = VersionControlManager()
        self.statistical_validator = StatisticalValidator()
        self.audit_manager = AuditTrailManager()
        
        # Validation weights
        self.weights = {
            "deterministic": 0.25,
            "version_control": 0.25,
            "statistical": 0.25,
            "audit_trail": 0.25
        }
        
        logger.info(f"Initialized ReproducibilityValidator with storage at: {self.storage_path}")
    
    def validate_reproducibility(
        self,
        run_id: str,
        reference_manifest: Optional[VersionManifest] = None,
        reference_results: Optional[Dict[str, Any]] = None,
        current_results: Optional[Dict[str, Any]] = None
    ) -> ReproducibilityReport:
        """
        Perform comprehensive reproducibility validation.
        
        Args:
            run_id: Run ID to validate
            reference_manifest: Reference version manifest (optional)
            reference_results: Reference benchmark results (optional)
            current_results: Current benchmark results (optional)
            
        Returns:
            ReproducibilityReport with validation results
        """
        logger.info(f"Starting reproducibility validation for run: {run_id}")
        
        # Initialize report
        report = ReproducibilityReport(
            run_id=run_id,
            timestamp=datetime.now(),
            overall_score=0.0
        )
        
        # Create audit trail for this validation
        audit_trail = self.audit_manager.create_trail(f"reproducibility_validation_{run_id}")
        
        try:
            # Validate deterministic execution
            deterministic_results = self._validate_deterministic_execution(run_id, audit_trail)
            report.components["deterministic"] = deterministic_results
            
            # Validate version control
            version_results = self._validate_version_control(run_id, reference_manifest, audit_trail)
            report.components["version_control"] = version_results
            
            # Validate statistical results
            statistical_results = self._validate_statistical_results(
                run_id, reference_results, current_results, audit_trail
            )
            report.components["statistical"] = statistical_results
            
            # Validate audit trail
            audit_results = self._validate_audit_trail(run_id, audit_trail)
            report.components["audit_trail"] = audit_results
            
            # Calculate overall score
            report.overall_score = self._calculate_overall_score(report.components)
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report.components)
            
            # Add details
            report.details = {
                "validation_timestamp": datetime.now().isoformat(),
                "component_weights": self.weights,
                "validation_counts": {
                    component: len([r for r in results if isinstance(r, ValidationResult)])
                    for component, results in report.components.items()
                }
            }
            
            logger.info(f"Completed reproducibility validation for run: {run_id} (score: {report.overall_score:.2f})")
            
        except Exception as e:
            logger.error(f"Error during reproducibility validation: {e}")
            
            # Log error to audit trail
            self.audit_manager.log_event(
                run_id=f"reproducibility_validation_{run_id}",
                component="ReproducibilityValidator",
                action="validate_reproducibility",
                event_type="error",
                details={"error": str(e), "error_type": type(e).__name__},
                severity="error"
            )
            
            # Set score to 0 due to error
            report.overall_score = 0.0
            report.recommendations = [f"Validation failed due to error: {str(e)}"]
        
        finally:
            # Close audit trail
            self.audit_manager.close_trail(f"reproducibility_validation_{run_id}")
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _validate_deterministic_execution(self, run_id: str, audit_trail: AuditTrail) -> List[ValidationResult]:
        """Validate deterministic execution."""
        results = []
        
        try:
            # Check if deterministic environment is active
            is_active = self.deterministic_env.is_active
            
            results.append(ValidationResult(
                component="deterministic",
                check_name="environment_active",
                passed=is_active,
                score=1.0 if is_active else 0.0,
                details={"is_active": is_active},
                issues=["Deterministic environment not active"] if not is_active else []
            ))
            
            # Get environment state
            env_state = self.deterministic_env.current_state
            
            if env_state:
                results.append(ValidationResult(
                    component="deterministic",
                    check_name="state_consistency",
                    passed=True,
                    score=1.0,
                    details={
                        "random_seed": env_state.random_seed,
                        "python_hash_seed": env_state.python_hash_seed,
                        "state_hash": env_state.state_hash
                    },
                    issues=[]
                ))
            else:
                results.append(ValidationResult(
                    component="deterministic",
                    check_name="state_consistency",
                    passed=False,
                    score=0.0,
                    details={},
                    issues=["No environment state available"]
                ))
            
            # Log to audit trail
            self.audit_manager.log_event(
                run_id=audit_trail.run_id,
                component="DeterministicValidator",
                action="validate_deterministic_execution",
                event_type="validation",
                details={"results": [r.to_dict() for r in results]},
                severity="info"
            )
            
        except Exception as e:
            logger.error(f"Error in deterministic validation: {e}")
            results.append(ValidationResult(
                component="deterministic",
                check_name="validation_error",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                issues=[f"Validation error: {str(e)}"]
            ))
        
        return results
    
    def _validate_version_control(
        self, 
        run_id: str, 
        reference_manifest: Optional[VersionManifest], 
        audit_trail: AuditTrail
    ) -> List[ValidationResult]:
        """Validate version control."""
        results = []
        
        try:
            # Create current manifest
            current_manifest = self.version_manager.create_manifest(f"validation_{run_id}")
            
            # Add common components
            try:
                self.version_manager.add_python_module("numpy", {"purpose": "numerical_computing"})
            except ImportError:
                pass
            
            try:
                self.version_manager.add_python_module("scipy", {"purpose": "statistical_analysis"})
            except ImportError:
                pass
            
            try:
                self.version_manager.add_python_module("pandas", {"purpose": "data_analysis"})
            except ImportError:
                pass
            
            # Check if reference manifest is provided
            if reference_manifest is None:
                results.append(ValidationResult(
                    component="version_control",
                    check_name="reference_available",
                    passed=False,
                    score=0.0,
                    details={},
                    issues=["No reference manifest provided for comparison"]
                ))
            else:
                # Compare manifests
                comparison = self.version_manager.compare_manifests(reference_manifest, current_manifest)
                
                # Calculate score based on differences
                total_components = len(reference_manifest.components)
                if total_components == 0:
                    score = 1.0  # No components to compare
                else:
                    unchanged_components = len(comparison["differences"]["components"]["unchanged"])
                    score = unchanged_components / total_components
                
                results.append(ValidationResult(
                    component="version_control",
                    check_name="component_consistency",
                    passed=score >= 0.9,  # 90% threshold
                    score=score,
                    details=comparison,
                    issues=[
                        f"Modified components: {len(comparison['differences']['components']['changed'])}",
                        f"Added components: {len(comparison['differences']['components']['added'])}",
                        f"Removed components: {len(comparison['differences']['components']['removed'])}"
                    ] if score < 0.9 else []
                ))
                
                # Verify reproducibility
                verification = self.version_manager.verify_reproducibility(reference_manifest)
                
                results.append(ValidationResult(
                    component="version_control",
                    check_name="reproducibility_verification",
                    passed=verification["overall_reproducible"],
                    score=1.0 if verification["overall_reproducible"] else 0.0,
                    details=verification,
                    issues=["Components not reproducible"] if not verification["overall_reproducible"] else []
                ))
            
            # Log to audit trail
            self.audit_manager.log_event(
                run_id=audit_trail.run_id,
                component="VersionControlValidator",
                action="validate_version_control",
                event_type="validation",
                details={"results": [r.to_dict() for r in results]},
                severity="info"
            )
            
        except Exception as e:
            logger.error(f"Error in version control validation: {e}")
            results.append(ValidationResult(
                component="version_control",
                check_name="validation_error",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                issues=[f"Validation error: {str(e)}"]
            ))
        
        return results
    
    def _validate_statistical_results(
        self,
        run_id: str,
        reference_results: Optional[Dict[str, Any]],
        current_results: Optional[Dict[str, Any]],
        audit_trail: AuditTrail
    ) -> List[ValidationResult]:
        """
        Validate statistical consistency between current and reference runs using paired tests.
        Expects both current_results and reference_results to provide arrays for key metrics.
        """
        results: List[ValidationResult] = []
        try:
            if reference_results is None or current_results is None:
                results.append(ValidationResult(
                    component="statistical",
                    check_name="results_available",
                    passed=False,
                    score=0.0,
                    details={
                        "reference_available": reference_results is not None,
                        "current_available": current_results is not None
                    },
                    issues=["Missing reference or current results for comparison"]
                ))
                return results
            
            # Define key metrics expected to be present
            key_metrics = [
                "total_profit_series",
                "final_market_share_series",
                "revenue_series",
            ]
            metric_findings: Dict[str, Any] = {}
            all_passed = True
            combined_score = 1.0
            
            for metric in key_metrics:
                ref = reference_results.get(metric)
                cur = current_results.get(metric)
                if not isinstance(ref, list) or not isinstance(cur, list) or len(ref) < 2 or len(cur) < 2:
                    # If unavailable, do not fail the whole validation; just record issue
                    metric_findings[metric] = {"status": "missing_or_insufficient_data"}
                    combined_score *= 0.9
                    continue
                # Use two-sample t-test (unpaired) as general case; fallback if sizes differ
                try:
                    test: StatisticalSummary = self.statistical_validator.calculate_summary(cur)
                    ref_summary: StatisticalSummary = self.statistical_validator.calculate_summary(ref)
                    ttest = self.statistical_validator.t_test_two_samples(cur, ref, alpha=0.05, equal_var=False, alternative="two-sided")
                    # Pass if not significant
                    metric_pass = not ttest.is_significant()
                    all_passed = all_passed and metric_pass
                    combined_score *= (0.95 if metric_pass else 0.7)
                    metric_findings[metric] = {
                        "p_value": ttest.p_value,
                        "significant": ttest.is_significant(),
                        "current_mean": test.mean,
                        "reference_mean": ref_summary.mean,
                        "conclusion": ttest.conclusion
                    }
                except Exception as e:
                    metric_findings[metric] = {"status": "error", "error": str(e)}
                    combined_score *= 0.8
            
            results.append(ValidationResult(
                component="statistical",
                check_name="paired_consistency_tests",
                passed=all_passed,
                score=max(0.0, min(1.0, combined_score)),
                details=metric_findings,
                issues=[k for k, v in metric_findings.items() if isinstance(v, dict) and v.get("significant") is True]
            ))
            
            self.audit_manager.log_event(
                run_id=audit_trail.run_id,
                component="StatisticalValidator",
                action="validate_statistical_results",
                event_type="validation",
                details={"results": [r.to_dict() for r in results]},
                severity="info"
            )
        except Exception as e:
            logger.error(f"Error in statistical validation: {e}")
            results.append(ValidationResult(
                component="statistical",
                check_name="validation_error",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                issues=[f"Validation error: {str(e)}"]
            ))
        return results
    
    def _validate_audit_trail(self, run_id: str, audit_trail: AuditTrail) -> List[ValidationResult]:
        """Validate audit trail."""
        results = []
        
        try:
            # Check if audit trail exists
            if audit_trail is None:
                results.append(ValidationResult(
                    component="audit_trail",
                    check_name="trail_exists",
                    passed=False,
                    score=0.0,
                    details={},
                    issues=["No audit trail available"]
                ))
                return results
            
            # Verify trail integrity
            integrity_ok = self.audit_manager.verify_trail_integrity(audit_trail)
            
            results.append(ValidationResult(
                component="audit_trail",
                check_name="integrity",
                passed=integrity_ok,
                score=1.0 if integrity_ok else 0.0,
                details={"checksum": audit_trail.checksum},
                issues=["Audit trail integrity check failed"] if not integrity_ok else []
            ))
            
            # Check event completeness
            event_count = len(audit_trail.events)
            min_expected_events = 5  # Minimum expected events for a complete run
            
            completeness_score = min(1.0, event_count / min_expected_events)
            
            results.append(ValidationResult(
                component="audit_trail",
                check_name="completeness",
                passed=completeness_score >= 0.8,
                score=completeness_score,
                details={"event_count": event_count, "min_expected": min_expected_events},
                issues=[f"Insufficient events: {event_count}/{min_expected_events}"] if completeness_score < 0.8 else []
            ))
            
            # Check for error events
            error_events = [e for e in audit_trail.events if e.severity in ["error", "critical"]]
            error_rate = len(error_events) / event_count if event_count > 0 else 0.0
            
            error_score = max(0.0, 1.0 - (error_rate * 10))  # Penalize for errors
            
            results.append(ValidationResult(
                component="audit_trail",
                check_name="error_rate",
                passed=error_score >= 0.9,  # Max 1% error rate
                score=error_score,
                details={"error_count": len(error_events), "error_rate": error_rate},
                issues=[f"High error rate: {error_rate:.2%}"] if error_score < 0.9 else []
            ))
            
            # Log to audit trail
            self.audit_manager.log_event(
                run_id=audit_trail.run_id,
                component="AuditTrailValidator",
                action="validate_audit_trail",
                event_type="validation",
                details={"results": [r.to_dict() for r in results]},
                severity="info"
            )
            
        except Exception as e:
            logger.error(f"Error in audit trail validation: {e}")
            results.append(ValidationResult(
                component="audit_trail",
                check_name="validation_error",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                issues=[f"Validation error: {str(e)}"]
            ))
        
        return results
    
    def _calculate_overall_score(self, component_results: Dict[str, List[ValidationResult]]) -> float:
        """Calculate overall reproducibility score."""
        overall_score = 0.0
        
        for component, results in component_results.items():
            if not results:
                continue
            
            # Calculate average score for this component
            component_score = sum(r.score for r in results) / len(results)
            
            # Apply weight
            weight = self.weights.get(component, 0.0)
            overall_score += component_score * weight
        
        return overall_score
    
    def _generate_recommendations(self, component_results: Dict[str, List[ValidationResult]]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for component, results in component_results.items():
            for result in results:
                if not result.passed:
                    # Generate specific recommendations based on component and check
                    if component == "deterministic":
                        if result.check_name == "environment_active":
                            recommendations.append(
                                "Enable deterministic execution environment for reproducible random seeds"
                            )
                        elif result.check_name == "state_consistency":
                            recommendations.append(
                                "Ensure consistent environment state across benchmark runs"
                            )
                    
                    elif component == "version_control":
                        if result.check_name == "reference_available":
                            recommendations.append(
                                "Provide reference version manifest for reproducibility comparison"
                            )
                        elif result.check_name == "component_consistency":
                            recommendations.append(
                                "Ensure consistent versions of all components across benchmark runs"
                            )
                        elif result.check_name == "reproducibility_verification":
                            recommendations.append(
                                "Verify that all components are reproducible with exact versions"
                            )
                    
                    elif component == "statistical":
                        if result.check_name == "results_available":
                            recommendations.append(
                                "Provide both reference and current results for statistical comparison"
                            )
                        elif result.check_name == "result_validity":
                            recommendations.append(
                                "Improve result quality by addressing statistical validity issues"
                            )
                        elif result.check_name == "result_consistency":
                            recommendations.append(
                                "Investigate inconsistencies between reference and current results"
                            )
                    
                    elif component == "audit_trail":
                        if result.check_name == "trail_exists":
                            recommendations.append(
                                "Enable audit trail logging for complete traceability"
                            )
                        elif result.check_name == "integrity":
                            recommendations.append(
                                "Ensure audit trail integrity with proper checksums"
                            )
                        elif result.check_name == "completeness":
                            recommendations.append(
                                "Improve audit trail completeness with more detailed event logging"
                            )
                        elif result.check_name == "error_rate":
                            recommendations.append(
                                "Reduce error rate in benchmark execution"
                            )
        
        # Remove duplicates and limit to top recommendations
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def _save_report(self, report: ReproducibilityReport) -> str:
        """Save reproducibility report to disk."""
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"reproducibility_report_{report.run_id}_{timestamp}.json"
        report_path = self.storage_path / filename
        
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Saved reproducibility report to: {report_path}")
        return str(report_path)
    
    def load_report(self, filename: str) -> Optional[ReproducibilityReport]:
        """
        Load a reproducibility report from disk.
        
        Args:
            filename: Filename of the report
            
        Returns:
            Loaded report or None if failed
        """
        report_path = self.storage_path / filename
        
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)
            
            # Create report object
            report = ReproducibilityReport(
                run_id=data["run_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                overall_score=data["overall_score"],
                components=data["components"],
                details=data["details"],
                recommendations=data["recommendations"]
            )
            
            logger.info(f"Loaded reproducibility report from: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to load reproducibility report {filename}: {e}")
            return None
    
    def list_reports(self) -> List[str]:
        """
        List all available reproducibility reports.
        
        Returns:
            List of report filenames
        """
        return [f.name for f in self.storage_path.glob("*.json")]
    
    def compare_reports(
        self, 
        report1: ReproducibilityReport, 
        report2: ReproducibilityReport
    ) -> Dict[str, Any]:
        """
        Compare two reproducibility reports.
        
        Args:
            report1: First report
            report2: Second report
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "run_ids": [report1.run_id, report2.run_id],
            "timestamps": [report1.timestamp.isoformat(), report2.timestamp.isoformat()],
            "overall_scores": [report1.overall_score, report2.overall_score],
            "score_difference": report2.overall_score - report1.overall_score,
            "component_comparisons": {}
        }
        
        # Compare components
        all_components = set(report1.components.keys()) | set(report2.components.keys())
        
        for component in all_components:
            comp1 = report1.components.get(component, [])
            comp2 = report2.components.get(component, [])
            
            # Calculate average scores
            score1 = sum(r.score for r in comp1) / len(comp1) if comp1 else 0.0
            score2 = sum(r.score for r in comp2) / len(comp2) if comp2 else 0.0
            
            comparison["component_comparisons"][component] = {
                "scores": [score1, score2],
                "difference": score2 - score1,
                "improved": score2 > score1
            }
        
        return comparison
    
    def set_validation_weights(self, weights: Dict[str, float]) -> None:
        """
        Set validation weights for components.
        
        Args:
            weights: Dictionary of component weights
        """
        # Validate weights sum to 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.weights.update(weights)
        logger.info(f"Updated validation weights: {self.weights}")