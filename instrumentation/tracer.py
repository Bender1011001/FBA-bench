from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

# Lightweight, dependency-free tracer for unit tests.
# Provides a stable import and simple span lifecycle without requiring OpenTelemetry.


@dataclass
class Span:
    span_id: str
    operation: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    tags: Dict[str, Any] = field(default_factory=dict)


class Tracer:
    """
    Minimal tracer implementation for tests expecting:
      - Tracer()
      - start_span(operation: str, trace_id: Optional[str] = None) -> str
      - end_span(span_id: str) -> None
      - get_trace(span_id: str) -> Optional[Span]
      - export_trace(span_id: str) -> Dict[str, Any]
    """

    def __init__(self) -> None:
        self._spans: Dict[str, Span] = {}
        # trace_id -> list of span_ids
        self._traces: Dict[str, List[str]] = {}

    # Tests call start_span(operation, trace_id)
    def start_span(self, operation: str, trace_id: Optional[str] = None) -> str:
        if trace_id is None:
            trace_id = str(uuid4())
        sid = str(uuid4())
        self._spans[sid] = Span(span_id=sid, operation=operation, started_at=datetime.utcnow())
        self._traces.setdefault(trace_id, []).append(sid)
        return sid

    def end_span(self, span_id: str) -> None:
        s = self._spans.get(span_id)
        if s and s.ended_at is None:
            s.ended_at = datetime.utcnow()

    def get_trace(self, span_id: str) -> Optional[Span]:
        return self._spans.get(span_id)

    def export_trace(self, span_id: str) -> Dict[str, Any]:
        s = self._spans[span_id]
        return {
            "span_id": s.span_id,
            "operation": s.operation,
            "started_at": s.started_at.isoformat(),
            "ended_at": s.ended_at.isoformat() if s.ended_at else None,
            "tags": s.tags,
        }