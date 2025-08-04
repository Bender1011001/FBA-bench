"""
Golden Master Testing System for FBA-Bench Reproducibility

Provides comprehensive baseline recording, regression detection, and diff analysis
to ensure bit-perfect reproducibility across simulation runs.
"""

import json
import gzip
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

from reproducibility.event_snapshots import EventSnapshot

logger = logging.getLogger(__name__)

@dataclass
class ToleranceConfig:
    """Configuration for acceptable differences in comparisons."""
    numeric_tolerance: float = 1e-10
    event_tolerance: int = 0  # Number of events that can differ
    timestamp_tolerance_ms: float = 1.0  # Milliseconds
    floating_point_epsilon: float = 1e-12
    ignore_fields: List[str] = field(default_factory=list)
    ignore_patterns: List[str] = field(default_factory=list)

@dataclass
class DiffDetail:
    """Detailed information about a specific difference."""
    path: str
    expected: Any
    actual: Any
    diff_type: str  # "missing", "extra", "different", "type_mismatch"
    severity: str   # "critical", "warning", "info"
    description: str

@dataclass
class ComparisonResult:
    """Result of comparing simulation runs."""
    is_identical: bool
    is_within_tolerance: bool
    differences: List[DiffDetail] = field(default_factory=list)
    critical_differences: List[DiffDetail] = field(default_factory=list)
    warnings: List[DiffDetail] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    comparison_time_ms: float = 0.0
    
    def has_critical_differences(self) -> bool:
        """Check if there are any critical differences."""
        return len(self.critical_differences) > 0
    
    def summary(self) -> str:
        """Get a human-readable summary."""
        if self.is_identical:
            return "✅ Runs are bit-perfect identical"
        elif self.is_within_tolerance:
            return f"⚠️ Runs differ but within tolerance ({len(self.differences)} differences)"
        else:
            return f"❌ Runs have significant differences ({len(self.critical_differences)} critical)"

@dataclass
class GoldenMasterRecord:
    """Record of a golden master baseline."""
    label: str
    timestamp: str
    simulation_data: Dict[str, Any]
    metadata: Dict[str, Any]
    data_hash: str
    event_hash: str
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldenMasterRecord':
        """Create from dictionary after deserialization."""
        return cls(**data)

class GoldenMasterTester:
    """
    Golden Master Testing system for capturing and validating simulation baselines.
    
    Provides:
    - Baseline recording with comprehensive metadata
    - Regression detection with configurable tolerances
    - Detailed diff analysis for troubleshooting
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        storage_dir: str = "golden_masters",
        tolerance_config: Optional[ToleranceConfig] = None,
        enable_compression: bool = True,
        enable_validation: bool = True
    ):
        """
        Initialize Golden Master testing system.
        
        Args:
            storage_dir: Directory for storing golden master files
            tolerance_config: Configuration for acceptable differences
            enable_compression: Whether to compress stored data
            enable_validation: Whether to validate data integrity
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.tolerance_config = tolerance_config or ToleranceConfig()
        self.enable_compression = enable_compression
        self.enable_validation = enable_validation
        
        # In-memory cache for performance
        self._golden_masters: Dict[str, GoldenMasterRecord] = {}
        
        logger.info(f"Golden Master Tester initialized: {self.storage_dir}")
    
    def record_golden_master(
        self,
        simulation_run: Dict[str, Any],
        label: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a simulation run as a golden master baseline.
        
        Args:
            simulation_run: Complete simulation data to record
            label: Unique label for this golden master
            metadata: Additional metadata about the run
            
        Returns:
            True if successfully recorded, False otherwise
        """
        try:
            # Prepare metadata
            full_metadata = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "label": label,
                "data_size": len(json.dumps(simulation_run)),
                **(metadata or {})
            }
            
            # Generate hashes for integrity checking
            data_json = json.dumps(simulation_run, sort_keys=True, separators=(',', ':'))
            data_hash = hashlib.sha256(data_json.encode()).hexdigest()
            
            # Generate event hash if events are present
            event_hash = ""
            if "events" in simulation_run:
                event_hash = EventSnapshot.generate_event_stream_hash(simulation_run["events"])
            
            # Create golden master record
            golden_master = GoldenMasterRecord(
                label=label,
                timestamp=datetime.now(timezone.utc).isoformat(),
                simulation_data=simulation_run,
                metadata=full_metadata,
                data_hash=data_hash,
                event_hash=event_hash
            )
            
            # Store to file
            file_path = self.storage_dir / f"{label}.golden"
            self._save_golden_master(golden_master, file_path)
            
            # Cache in memory
            self._golden_masters[label] = golden_master
            
            logger.info(f"Golden master '{label}' recorded: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record golden master '{label}': {e}")
            return False
    
    def compare_against_golden(
        self,
        new_run: Dict[str, Any],
        golden_label: str,
        tolerance_override: Optional[ToleranceConfig] = None
    ) -> ComparisonResult:
        """
        Compare a new simulation run against a golden master.
        
        Args:
            new_run: New simulation data to compare
            golden_label: Label of the golden master to compare against
            tolerance_override: Override default tolerance settings
            
        Returns:
            Detailed comparison result
        """
        start_time = time.time()
        
        try:
            # Load golden master
            golden_master = self._load_golden_master(golden_label)
            if not golden_master:
                return ComparisonResult(
                    is_identical=False,
                    is_within_tolerance=False,
                    critical_differences=[DiffDetail(
                        path="golden_master",
                        expected=golden_label,
                        actual="not_found",
                        diff_type="missing",
                        severity="critical",
                        description=f"Golden master '{golden_label}' not found"
                    )]
                )
            
            # Use provided tolerance or default
            tolerance = tolerance_override or self.tolerance_config
            
            # Perform detailed comparison
            result = self._deep_compare(
                golden_master.simulation_data,
                new_run,
                tolerance,
                path_prefix=""
            )
            
            # Add timing information
            result.comparison_time_ms = (time.time() - start_time) * 1000
            
            # Generate statistics
            result.statistics = {
                "golden_master_label": golden_label,
                "golden_master_timestamp": golden_master.timestamp,
                "comparison_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_differences": len(result.differences),
                "critical_differences": len(result.critical_differences),
                "warnings": len(result.warnings),
                "data_hash_match": golden_master.data_hash == self._calculate_hash(new_run),
                "tolerance_config": asdict(tolerance)
            }
            
            logger.info(f"Comparison complete: {result.summary()}")
            return result
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return ComparisonResult(
                is_identical=False,
                is_within_tolerance=False,
                critical_differences=[DiffDetail(
                    path="comparison",
                    expected="success",
                    actual="error",
                    diff_type="different",
                    severity="critical",
                    description=f"Comparison failed: {e}"
                )]
            )
    
    def analyze_differences(
        self,
        run1: Dict[str, Any],
        run2: Dict[str, Any],
        tolerance: Optional[ToleranceConfig] = None
    ) -> ComparisonResult:
        """
        Perform detailed diff analysis between two simulation runs.
        
        Args:
            run1: First simulation run
            run2: Second simulation run
            tolerance: Tolerance configuration
            
        Returns:
            Detailed comparison result
        """
        start_time = time.time()
        tolerance = tolerance or self.tolerance_config
        
        result = self._deep_compare(run1, run2, tolerance, "")
        result.comparison_time_ms = (time.time() - start_time) * 1000
        
        # Add analysis statistics
        result.statistics = {
            "run1_size": len(json.dumps(run1)),
            "run2_size": len(json.dumps(run2)),
            "comparison_timestamp": datetime.now(timezone.utc).isoformat(),
            "tolerance_config": asdict(tolerance)
        }
        
        return result
    
    def _deep_compare(
        self,
        expected: Any,
        actual: Any,
        tolerance: ToleranceConfig,
        path_prefix: str
    ) -> ComparisonResult:
        """
        Perform deep comparison of two data structures.
        
        Args:
            expected: Expected value
            actual: Actual value
            tolerance: Tolerance configuration
            path_prefix: Current path in the data structure
            
        Returns:
            Comparison result
        """
        result = ComparisonResult(is_identical=True, is_within_tolerance=True)
        
        # Check if path should be ignored
        if self._should_ignore_path(path_prefix, tolerance):
            return result
        
        # Type comparison
        if type(expected) != type(actual):
            diff = DiffDetail(
                path=path_prefix,
                expected=str(type(expected)),
                actual=str(type(actual)),
                diff_type="type_mismatch",
                severity="critical",
                description=f"Type mismatch at {path_prefix}"
            )
            result.differences.append(diff)
            result.critical_differences.append(diff)
            result.is_identical = False
            result.is_within_tolerance = False
            return result
        
        # Handle different data types
        if isinstance(expected, dict):
            return self._compare_dicts(expected, actual, tolerance, path_prefix)
        elif isinstance(expected, list):
            return self._compare_lists(expected, actual, tolerance, path_prefix)
        elif isinstance(expected, (int, float)):
            return self._compare_numbers(expected, actual, tolerance, path_prefix)
        elif isinstance(expected, str):
            return self._compare_strings(expected, actual, tolerance, path_prefix)
        else:
            # Direct comparison for other types
            if expected != actual:
                diff = DiffDetail(
                    path=path_prefix,
                    expected=expected,
                    actual=actual,
                    diff_type="different",
                    severity="warning",
                    description=f"Value difference at {path_prefix}"
                )
                result.differences.append(diff)
                result.warnings.append(diff)
                result.is_identical = False
        
        return result
    
    def _compare_dicts(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        tolerance: ToleranceConfig,
        path_prefix: str
    ) -> ComparisonResult:
        """Compare two dictionaries."""
        result = ComparisonResult(is_identical=True, is_within_tolerance=True)
        
        # Check for missing keys
        for key in expected:
            if key not in actual:
                diff = DiffDetail(
                    path=f"{path_prefix}.{key}" if path_prefix else key,
                    expected=expected[key],
                    actual="<missing>",
                    diff_type="missing",
                    severity="critical",
                    description=f"Missing key: {key}"
                )
                result.differences.append(diff)
                result.critical_differences.append(diff)
                result.is_identical = False
                result.is_within_tolerance = False
        
        # Check for extra keys
        for key in actual:
            if key not in expected:
                diff = DiffDetail(
                    path=f"{path_prefix}.{key}" if path_prefix else key,
                    expected="<missing>",
                    actual=actual[key],
                    diff_type="extra",
                    severity="warning",
                    description=f"Extra key: {key}"
                )
                result.differences.append(diff)
                result.warnings.append(diff)
                result.is_identical = False
        
        # Compare common keys
        for key in set(expected.keys()) & set(actual.keys()):
            key_path = f"{path_prefix}.{key}" if path_prefix else key
            sub_result = self._deep_compare(expected[key], actual[key], tolerance, key_path)
            
            # Merge results
            result.differences.extend(sub_result.differences)
            result.critical_differences.extend(sub_result.critical_differences)
            result.warnings.extend(sub_result.warnings)
            
            if not sub_result.is_identical:
                result.is_identical = False
            if not sub_result.is_within_tolerance:
                result.is_within_tolerance = False
        
        return result
    
    def _compare_lists(
        self,
        expected: List[Any],
        actual: List[Any],
        tolerance: ToleranceConfig,
        path_prefix: str
    ) -> ComparisonResult:
        """Compare two lists."""
        result = ComparisonResult(is_identical=True, is_within_tolerance=True)
        
        # Length comparison
        if len(expected) != len(actual):
            diff = DiffDetail(
                path=f"{path_prefix}.length",
                expected=len(expected),
                actual=len(actual),
                diff_type="different",
                severity="critical",
                description=f"List length mismatch: expected {len(expected)}, got {len(actual)}"
            )
            result.differences.append(diff)
            result.critical_differences.append(diff)
            result.is_identical = False
            result.is_within_tolerance = False
        
        # Element-wise comparison
        max_len = max(len(expected), len(actual))
        for i in range(max_len):
            element_path = f"{path_prefix}[{i}]"
            
            if i >= len(expected):
                diff = DiffDetail(
                    path=element_path,
                    expected="<missing>",
                    actual=actual[i],
                    diff_type="extra",
                    severity="warning",
                    description=f"Extra list element at index {i}"
                )
                result.differences.append(diff)
                result.warnings.append(diff)
                result.is_identical = False
            elif i >= len(actual):
                diff = DiffDetail(
                    path=element_path,
                    expected=expected[i],
                    actual="<missing>",
                    diff_type="missing",
                    severity="critical",
                    description=f"Missing list element at index {i}"
                )
                result.differences.append(diff)
                result.critical_differences.append(diff)
                result.is_identical = False
                result.is_within_tolerance = False
            else:
                sub_result = self._deep_compare(expected[i], actual[i], tolerance, element_path)
                
                # Merge results
                result.differences.extend(sub_result.differences)
                result.critical_differences.extend(sub_result.critical_differences)
                result.warnings.extend(sub_result.warnings)
                
                if not sub_result.is_identical:
                    result.is_identical = False
                if not sub_result.is_within_tolerance:
                    result.is_within_tolerance = False
        
        return result
    
    def _compare_numbers(
        self,
        expected: Union[int, float],
        actual: Union[int, float],
        tolerance: ToleranceConfig,
        path_prefix: str
    ) -> ComparisonResult:
        """Compare two numbers with tolerance."""
        result = ComparisonResult(is_identical=True, is_within_tolerance=True)
        
        if expected != actual:
            result.is_identical = False
            
            # Check if difference is within tolerance
            if isinstance(expected, float) or isinstance(actual, float):
                diff_value = abs(float(expected) - float(actual))
                relative_diff = diff_value / max(abs(float(expected)), abs(float(actual)), 1e-10)
                
                within_tolerance = (
                    diff_value <= tolerance.numeric_tolerance or
                    relative_diff <= tolerance.floating_point_epsilon
                )
            else:
                # Integer comparison
                diff_value = abs(expected - actual)
                within_tolerance = diff_value <= tolerance.numeric_tolerance
            
            severity = "warning" if within_tolerance else "critical"
            
            diff = DiffDetail(
                path=path_prefix,
                expected=expected,
                actual=actual,
                diff_type="different",
                severity=severity,
                description=f"Numeric difference: {expected} vs {actual} (diff: {abs(expected - actual)})"
            )
            
            result.differences.append(diff)
            
            if within_tolerance:
                result.warnings.append(diff)
            else:
                result.critical_differences.append(diff)
                result.is_within_tolerance = False
        
        return result
    
    def _compare_strings(
        self,
        expected: str,
        actual: str,
        tolerance: ToleranceConfig,
        path_prefix: str
    ) -> ComparisonResult:
        """Compare two strings."""
        result = ComparisonResult(is_identical=True, is_within_tolerance=True)
        
        if expected != actual:
            result.is_identical = False
            
            # Check if it's a timestamp difference within tolerance
            if self._is_timestamp_field(path_prefix):
                time_diff = self._calculate_timestamp_difference(expected, actual)
                if time_diff is not None and time_diff <= tolerance.timestamp_tolerance_ms:
                    severity = "warning"
                else:
                    severity = "critical"
                    result.is_within_tolerance = False
            else:
                severity = "warning"
            
            diff = DiffDetail(
                path=path_prefix,
                expected=expected,
                actual=actual,
                diff_type="different",
                severity=severity,
                description=f"String difference at {path_prefix}"
            )
            
            result.differences.append(diff)
            
            if severity == "warning":
                result.warnings.append(diff)
            else:
                result.critical_differences.append(diff)
        
        return result
    
    def _should_ignore_path(self, path: str, tolerance: ToleranceConfig) -> bool:
        """Check if a path should be ignored during comparison."""
        # Check ignore fields
        for field in tolerance.ignore_fields:
            if field in path:
                return True
        
        # Check ignore patterns
        for pattern in tolerance.ignore_patterns:
            if pattern in path:
                return True
        
        return False
    
    def _is_timestamp_field(self, path: str) -> bool:
        """Check if a field is likely a timestamp."""
        timestamp_indicators = ["timestamp", "time", "date", "created_at", "updated_at"]
        return any(indicator in path.lower() for indicator in timestamp_indicators)
    
    def _calculate_timestamp_difference(self, ts1: str, ts2: str) -> Optional[float]:
        """Calculate difference between timestamps in milliseconds."""
        try:
            from datetime import datetime
            
            # Try parsing ISO format timestamps
            dt1 = datetime.fromisoformat(ts1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(ts2.replace('Z', '+00:00'))
            
            return abs((dt1 - dt2).total_seconds() * 1000)
        except:
            return None
    
    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash of data for integrity checking."""
        data_json = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(data_json.encode()).hexdigest()
    
    def _save_golden_master(self, golden_master: GoldenMasterRecord, file_path: Path):
        """Save golden master to file."""
        data = golden_master.to_dict()
        json_data = json.dumps(data, separators=(',', ':')).encode('utf-8')
        
        if self.enable_compression:
            json_data = gzip.compress(json_data)
            file_path = file_path.with_suffix('.golden.gz')
        
        with open(file_path, 'wb') as f:
            f.write(json_data)
    
    def _load_golden_master(self, label: str) -> Optional[GoldenMasterRecord]:
        """Load golden master from file or cache."""
        # Check memory cache first
        if label in self._golden_masters:
            return self._golden_masters[label]
        
        # Try loading from file
        file_path = self.storage_dir / f"{label}.golden"
        compressed_path = self.storage_dir / f"{label}.golden.gz"
        
        target_path = compressed_path if compressed_path.exists() else file_path
        
        if not target_path.exists():
            logger.error(f"Golden master file not found: {target_path}")
            return None
        
        try:
            with open(target_path, 'rb') as f:
                data = f.read()
            
            if target_path.suffix == '.gz':
                data = gzip.decompress(data)
            
            golden_data = json.loads(data.decode('utf-8'))
            golden_master = GoldenMasterRecord.from_dict(golden_data)
            
            # Cache in memory
            self._golden_masters[label] = golden_master
            
            return golden_master
            
        except Exception as e:
            logger.error(f"Failed to load golden master '{label}': {e}")
            return None
    
    def set_tolerance_levels(
        self,
        numeric_tolerance: Optional[float] = None,
        event_tolerance: Optional[int] = None,
        timestamp_tolerance_ms: Optional[float] = None
    ):
        """
        Update tolerance levels for comparisons.
        
        Args:
            numeric_tolerance: Tolerance for numeric differences
            event_tolerance: Number of events that can differ
            timestamp_tolerance_ms: Tolerance for timestamp differences in milliseconds
        """
        if numeric_tolerance is not None:
            self.tolerance_config.numeric_tolerance = numeric_tolerance
        if event_tolerance is not None:
            self.tolerance_config.event_tolerance = event_tolerance
        if timestamp_tolerance_ms is not None:
            self.tolerance_config.timestamp_tolerance_ms = timestamp_tolerance_ms
        
        logger.info(f"Tolerance levels updated: {asdict(self.tolerance_config)}")
    
    def generate_reproducibility_report(self, comparison_results: List[ComparisonResult]) -> Dict[str, Any]:
        """
        Generate comprehensive reproducibility report from multiple comparisons.
        
        Args:
            comparison_results: List of comparison results
            
        Returns:
            Comprehensive report dictionary
        """
        total_comparisons = len(comparison_results)
        identical_runs = sum(1 for r in comparison_results if r.is_identical)
        within_tolerance = sum(1 for r in comparison_results if r.is_within_tolerance)
        critical_failures = sum(1 for r in comparison_results if r.has_critical_differences())
        
        # Aggregate statistics
        total_differences = sum(len(r.differences) for r in comparison_results)
        total_critical = sum(len(r.critical_differences) for r in comparison_results)
        total_warnings = sum(len(r.warnings) for r in comparison_results)
        
        # Common difference patterns
        difference_patterns = defaultdict(int)
        for result in comparison_results:
            for diff in result.differences:
                difference_patterns[diff.path] += 1
        
        report = {
            "summary": {
                "total_comparisons": total_comparisons,
                "identical_runs": identical_runs,
                "within_tolerance": within_tolerance,
                "critical_failures": critical_failures,
                "reproducibility_rate": (within_tolerance / total_comparisons) if total_comparisons > 0 else 0.0,
                "perfect_reproducibility_rate": (identical_runs / total_comparisons) if total_comparisons > 0 else 0.0
            },
            "statistics": {
                "total_differences": total_differences,
                "total_critical_differences": total_critical,
                "total_warnings": total_warnings,
                "average_differences_per_run": total_differences / total_comparisons if total_comparisons > 0 else 0,
                "average_comparison_time_ms": sum(r.comparison_time_ms for r in comparison_results) / total_comparisons if total_comparisons > 0 else 0
            },
            "patterns": {
                "common_difference_paths": dict(sorted(difference_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
                "most_problematic_areas": [path for path, count in difference_patterns.items() if count > len(comparison_results) * 0.5]
            },
            "recommendations": self._generate_recommendations(comparison_results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tolerance_config": asdict(self.tolerance_config)
        }
        
        logger.info(f"Reproducibility report generated: {report['summary']['reproducibility_rate']:.2%} success rate")
        
        return report
    
    def _generate_recommendations(self, comparison_results: List[ComparisonResult]) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        # Check for common patterns
        if any(r.has_critical_differences() for r in comparison_results):
            recommendations.append("Critical differences detected. Review simulation determinism.")
        
        # Check for numeric precision issues
        numeric_diffs = [
            diff for result in comparison_results
            for diff in result.differences
            if diff.diff_type == "different" and any(isinstance(val, (int, float)) for val in [diff.expected, diff.actual])
        ]
        
        if len(numeric_diffs) > len(comparison_results) * 0.3:
            recommendations.append("High number of numeric differences. Consider adjusting numeric tolerance.")
        
        # Check for timestamp issues
        timestamp_diffs = [
            diff for result in comparison_results
            for diff in result.differences
            if self._is_timestamp_field(diff.path)
        ]
        
        if timestamp_diffs:
            recommendations.append("Timestamp differences detected. Review time synchronization or increase timestamp tolerance.")
        
        # Check performance
        avg_time = sum(r.comparison_time_ms for r in comparison_results) / len(comparison_results) if comparison_results else 0
        if avg_time > 1000:  # > 1 second
            recommendations.append("Comparison performance is slow. Consider optimizing data structures or enabling compression.")
        
        return recommendations
    
    def list_golden_masters(self) -> List[str]:
        """List all available golden master labels."""
        labels = []
        
        # From memory cache
        labels.extend(self._golden_masters.keys())
        
        # From file system
        for file_path in self.storage_dir.glob("*.golden*"):
            label = file_path.stem.replace('.golden', '')
            if label not in labels:
                labels.append(label)
        
        return sorted(labels)
    
    def delete_golden_master(self, label: str) -> bool:
        """
        Delete a golden master.
        
        Args:
            label: Label of golden master to delete
            
        Returns:
            True if successfully deleted
        """
        try:
            # Remove from memory cache
            if label in self._golden_masters:
                del self._golden_masters[label]
            
            # Remove files
            file_path = self.storage_dir / f"{label}.golden"
            compressed_path = self.storage_dir / f"{label}.golden.gz"
            
            deleted = False
            if file_path.exists():
                file_path.unlink()
                deleted = True
            if compressed_path.exists():
                compressed_path.unlink()
                deleted = True
            
            if deleted:
                logger.info(f"Golden master '{label}' deleted")
            else:
                logger.warning(f"Golden master '{label}' not found for deletion")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete golden master '{label}': {e}")
            return False

# Alias for backward compatibility
GoldenMaster = GoldenMasterTester