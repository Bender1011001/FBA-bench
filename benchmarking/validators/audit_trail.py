"""
Audit trail management for benchmarking runs.

This module provides tools for creating and managing comprehensive audit trails
for all benchmark runs, ensuring complete traceability and reproducibility.
"""

import os
import json
import logging
import hashlib
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager
import uuid

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Single audit event."""
    timestamp: datetime
    event_type: str
    event_id: str
    run_id: str
    component: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # debug, info, warning, error, critical
    user: str = "system"
    session_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary."""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class AuditTrail:
    """Complete audit trail for a benchmark run."""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    events: List[AuditEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate checksum after initialization."""
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum of the audit trail."""
        trail_data = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "events": [event.to_dict() for event in self.events],
            "metadata": dict(sorted(self.metadata.items()))
        }
        
        trail_str = json.dumps(trail_data, sort_keys=True)
        return hashlib.sha256(trail_str.encode()).hexdigest()
    
    def add_event(self, event: AuditEvent) -> None:
        """Add an event to the audit trail."""
        self.events.append(event)
        self.checksum = self._calculate_checksum()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
            "checksum": self.checksum,
            "event_count": len(self.events)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditTrail':
        """Create from dictionary."""
        if "start_time" in data and isinstance(data["start_time"], str):
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        
        if "end_time" in data and data["end_time"] is not None:
            if isinstance(data["end_time"], str):
                data["end_time"] = datetime.fromisoformat(data["end_time"])
        
        if "events" in data:
            data["events"] = [AuditEvent.from_dict(event) for event in data["events"]]
        
        trail = cls(**data)
        # Recalculate checksum to ensure integrity
        trail.checksum = trail._calculate_checksum()
        return trail


class AuditTrailManager:
    """
    Manages audit trails for benchmarking runs.
    
    This class provides tools for creating, managing, and querying audit trails
    to ensure complete traceability of all benchmark operations.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the audit trail manager.
        
        Args:
            storage_path: Path to store audit trails
        """
        self.storage_path = Path(storage_path) if storage_path else Path.cwd() / "audit_trails"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._active_trails: Dict[str, AuditTrail] = {}
        self._lock = threading.Lock()
        
        logger.info(f"Initialized AuditTrailManager with storage at: {self.storage_path}")
    
    def create_trail(self, run_id: str, metadata: Optional[Dict[str, Any]] = None) -> AuditTrail:
        """
        Create a new audit trail.
        
        Args:
            run_id: Unique identifier for the benchmark run
            metadata: Additional metadata for the trail
            
        Returns:
            New audit trail
        """
        with self._lock:
            if run_id in self._active_trails:
                logger.warning(f"Audit trail already exists for run: {run_id}")
                return self._active_trails[run_id]
            
            trail = AuditTrail(
                run_id=run_id,
                start_time=datetime.now(),
                metadata=metadata or {}
            )
            
            self._active_trails[run_id] = trail
            
            logger.info(f"Created audit trail for run: {run_id}")
            return trail
    
    def get_active_trail(self, run_id: str) -> Optional[AuditTrail]:
        """Get an active audit trail."""
        with self._lock:
            return self._active_trails.get(run_id)
    
    def close_trail(self, run_id: str) -> Optional[AuditTrail]:
        """
        Close an active audit trail.
        
        Args:
            run_id: Run ID of the trail to close
            
        Returns:
            Closed audit trail or None if not found
        """
        with self._lock:
            if run_id not in self._active_trails:
                logger.warning(f"No active audit trail found for run: {run_id}")
                return None
            
            trail = self._active_trails[run_id]
            trail.end_time = datetime.now()
            trail.checksum = trail._calculate_checksum()
            
            # Save to disk
            self._save_trail(trail)
            
            # Remove from active trails
            del self._active_trails[run_id]
            
            logger.info(f"Closed audit trail for run: {run_id}")
            return trail
    
    def log_event(
        self,
        run_id: str,
        component: str,
        action: str,
        event_type: str = "general",
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        user: str = "system",
        session_id: str = ""
    ) -> bool:
        """
        Log an event to an audit trail.
        
        Args:
            run_id: Run ID of the audit trail
            component: Component generating the event
            action: Action being performed
            event_type: Type of event
            details: Additional event details
            severity: Event severity level
            user: User performing the action
            session_id: Session identifier
            
        Returns:
            True if event was logged successfully
        """
        with self._lock:
            trail = self._active_trails.get(run_id)
            if trail is None:
                logger.error(f"No active audit trail found for run: {run_id}")
                return False
            
            event = AuditEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                event_id=str(uuid.uuid4()),
                run_id=run_id,
                component=component,
                action=action,
                details=details or {},
                severity=severity,
                user=user,
                session_id=session_id
            )
            
            trail.add_event(event)
            
            logger.debug(f"Logged event to audit trail {run_id}: {component}.{action}")
            return True
    
    def save_trail(self, run_id: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Save an audit trail to disk.
        
        Args:
            run_id: Run ID of the trail to save
            filename: Filename to save (optional)
            
        Returns:
            Path to saved file or None if failed
        """
        with self._lock:
            trail = self._active_trails.get(run_id)
            if trail is None:
                logger.error(f"No active audit trail found for run: {run_id}")
                return None
            
            return self._save_trail(trail, filename)
    
    def _save_trail(self, trail: AuditTrail, filename: Optional[str] = None) -> str:
        """Save a trail to disk."""
        if filename is None:
            timestamp = trail.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"audit_trail_{trail.run_id}_{timestamp}.json"
        
        trail_path = self.storage_path / filename
        
        with open(trail_path, 'w') as f:
            json.dump(trail.to_dict(), f, indent=2)
        
        logger.info(f"Saved audit trail to: {trail_path}")
        return str(trail_path)
    
    def load_trail(self, filename: str) -> Optional[AuditTrail]:
        """
        Load an audit trail from disk.
        
        Args:
            filename: Filename of the audit trail
            
        Returns:
            Loaded audit trail or None if failed
        """
        trail_path = self.storage_path / filename
        
        try:
            with open(trail_path, 'r') as f:
                data = json.load(f)
            
            trail = AuditTrail.from_dict(data)
            
            # Verify checksum
            calculated_checksum = trail._calculate_checksum()
            if trail.checksum != calculated_checksum:
                logger.error(f"Audit trail checksum mismatch for: {filename}")
                return None
            
            logger.info(f"Loaded audit trail from: {trail_path}")
            return trail
            
        except Exception as e:
            logger.error(f"Failed to load audit trail {filename}: {e}")
            return None
    
    def list_trails(self) -> List[str]:
        """
        List all available audit trails.
        
        Returns:
            List of trail filenames
        """
        return [f.name for f in self.storage_path.glob("*.json")]
    
    def query_trails(
        self,
        run_id: Optional[str] = None,
        component: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query audit trails for specific events.
        
        Args:
            run_id: Filter by run ID
            component: Filter by component
            event_type: Filter by event type
            severity: Filter by severity
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of results
            
        Returns:
            List of matching events
        """
        results = []
        
        # Load all trails and filter
        for filename in self.list_trails():
            trail = self.load_trail(filename)
            if trail is None:
                continue
            
            # Filter by run_id
            if run_id is not None and trail.run_id != run_id:
                continue
            
            # Filter events
            for event in trail.events:
                # Apply filters
                if component is not None and event.component != component:
                    continue
                
                if event_type is not None and event.event_type != event_type:
                    continue
                
                if severity is not None and event.severity != severity:
                    continue
                
                if start_time is not None and event.timestamp < start_time:
                    continue
                
                if end_time is not None and event.timestamp > end_time:
                    continue
                
                # Add to results
                results.append(event.to_dict())
                
                # Check limit
                if limit is not None and len(results) >= limit:
                    break
            
            # Check limit
            if limit is not None and len(results) >= limit:
                break
        
        return results
    
    def get_trail_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of an audit trail.
        
        Args:
            run_id: Run ID of the trail
            
        Returns:
            Dictionary with trail summary or None if not found
        """
        # Check active trails first
        trail = self._active_trails.get(run_id)
        
        # If not active, try to load from disk
        if trail is None:
            # Find trail file for this run_id
            for filename in self.list_trails():
                loaded_trail = self.load_trail(filename)
                if loaded_trail and loaded_trail.run_id == run_id:
                    trail = loaded_trail
                    break
        
        if trail is None:
            return None
        
        # Calculate summary statistics
        event_counts = {}
        severity_counts = {}
        component_counts = {}
        
        for event in trail.events:
            # Count by event type
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            
            # Count by severity
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            
            # Count by component
            component_counts[event.component] = component_counts.get(event.component, 0) + 1
        
        # Calculate duration
        duration = None
        if trail.end_time:
            duration = (trail.end_time - trail.start_time).total_seconds()
        
        return {
            "run_id": trail.run_id,
            "start_time": trail.start_time.isoformat(),
            "end_time": trail.end_time.isoformat() if trail.end_time else None,
            "duration_seconds": duration,
            "event_counts": event_counts,
            "severity_counts": severity_counts,
            "component_counts": component_counts,
            "total_events": len(trail.events),
            "checksum": trail.checksum,
            "metadata": trail.metadata
        }
    
    def verify_trail_integrity(self, trail: AuditTrail) -> bool:
        """
        Verify the integrity of an audit trail.
        
        Args:
            trail: Audit trail to verify
            
        Returns:
            True if trail is intact
        """
        # Recalculate checksum
        calculated_checksum = trail._calculate_checksum()
        
        # Verify checksum
        if trail.checksum != calculated_checksum:
            logger.error(f"Audit trail checksum mismatch for run: {trail.run_id}")
            return False
        
        # Verify event timestamps are in order
        for i in range(1, len(trail.events)):
            if trail.events[i].timestamp < trail.events[i-1].timestamp:
                logger.error(f"Event timestamp out of order in trail: {trail.run_id}")
                return False
        
        # Verify all events belong to this run
        for event in trail.events:
            if event.run_id != trail.run_id:
                logger.error(f"Event with mismatched run_id in trail: {trail.run_id}")
                return False
        
        return True
    
    def export_trail(self, run_id: str, format: str = "json", filename: Optional[str] = None) -> Optional[str]:
        """
        Export an audit trail to a file.
        
        Args:
            run_id: Run ID of the trail to export
            format: Export format ('json', 'csv')
            filename: Filename to export to (optional)
            
        Returns:
            Path to exported file or None if failed
        """
        # Get the trail
        trail = self._active_trails.get(run_id)
        if trail is None:
            # Try to load from disk
            for trail_filename in self.list_trails():
                loaded_trail = self.load_trail(trail_filename)
                if loaded_trail and loaded_trail.run_id == run_id:
                    trail = loaded_trail
                    break
        
        if trail is None:
            logger.error(f"No audit trail found for run: {run_id}")
            return None
        
        # Generate filename if not provided
        if filename is None:
            timestamp = trail.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"audit_trail_{trail.run_id}_{timestamp}.{format}"
        
        export_path = self.storage_path / filename
        
        try:
            if format == "json":
                with open(export_path, 'w') as f:
                    json.dump(trail.to_dict(), f, indent=2)
            elif format == "csv":
                import csv
                
                with open(export_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        "timestamp", "event_id", "event_type", "component",
                        "action", "severity", "user", "session_id", "details"
                    ])
                    
                    # Write events
                    for event in trail.events:
                        writer.writerow([
                            event.timestamp.isoformat(),
                            event.event_id,
                            event.event_type,
                            event.component,
                            event.action,
                            event.severity,
                            event.user,
                            event.session_id,
                            json.dumps(event.details)
                        ])
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
            
            logger.info(f"Exported audit trail to: {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Failed to export audit trail: {e}")
            return None
    
    @contextmanager
    def audit_context(self, run_id: str, component: str, action: str, **kwargs):
        """
        Context manager for auditing operations.
        
        Args:
            run_id: Run ID for the audit trail
            component: Component being audited
            action: Action being performed
            **kwargs: Additional arguments for event logging
            
        Yields:
            None
        """
        # Log start event
        start_details = kwargs.get('start_details', {})
        start_details.update({"status": "started"})
        
        self.log_event(
            run_id=run_id,
            component=component,
            action=f"{action}_start",
            event_type="operation",
            details=start_details,
            severity=kwargs.get('severity', 'info'),
            user=kwargs.get('user', 'system'),
            session_id=kwargs.get('session_id', '')
        )
        
        try:
            yield
        except Exception as e:
            # Log error event
            error_details = kwargs.get('error_details', {})
            error_details.update({
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            self.log_event(
                run_id=run_id,
                component=component,
                action=f"{action}_error",
                event_type="error",
                details=error_details,
                severity="error",
                user=kwargs.get('user', 'system'),
                session_id=kwargs.get('session_id', '')
            )
            
            raise
        else:
            # Log completion event
            completion_details = kwargs.get('completion_details', {})
            completion_details.update({"status": "completed"})
            
            self.log_event(
                run_id=run_id,
                component=component,
                action=f"{action}_complete",
                event_type="operation",
                details=completion_details,
                severity=kwargs.get('severity', 'info'),
                user=kwargs.get('user', 'system'),
                session_id=kwargs.get('session_id', '')
            )