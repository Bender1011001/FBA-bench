from opentelemetry import trace
from opentelemetry.trace import Tracer, NonRecordingSpan
from typing import Optional, Dict, Any

class AgentTracer:
    """
    Handles OpenTelemetry tracing for agent-specific actions (observe, think, tool_call).
    Each tracing method returns a context manager that should be used with a 'with' statement.
    """
    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def trace_agent_turn(self, agent_id: str, tick: int, agent_type: str = "advanced_agent"):
        """
        Returns a context manager for tracing a complete agent turn.
        """
        span_name = f"agent_turn_{agent_id}_tick_{tick}"
        parent_span = trace.get_current_span()
        
        span = self.tracer.start_as_current_span(
            span_name,
            attributes={
                "agent.id": agent_id,
                "simulation.tick": tick,
                "agent.type": agent_type,
                "parent_span_id": str(parent_span.context.span_id) if parent_span and parent_span.context else "N/A"
            }
        )
        span.add_event("Agent turn started")
        return _SpanContextManager(span)

    def trace_observe_phase(self, current_tick: int, event_count: int):
        """
        Returns a context manager for tracing the agent's observation phase.
        """
        span = self.tracer.start_as_current_span("observe")
        span.set_attributes({
            "observe.tick": current_tick,
            "observe.event_count": event_count
        })
        return _SpanContextManager(span)

    def trace_think_phase(self, current_tick: int, llm_model: str, llm_tokens_used: int, decision_confidence: Optional[float] = None, parsed_response: Optional[Dict[str, Any]] = None):
        """
        Returns a context manager for tracing the agent's thinking/decision-making phase.
        """
        span = self.tracer.start_as_current_span("think")
        attributes = {
            "think.tick": current_tick,
            "llm.model": llm_model,
            "llm.tokens_used": llm_tokens_used,
        }
        if decision_confidence is not None:
            attributes["decision.confidence"] = decision_confidence
        if parsed_response:
            # Add general attributes from the parsed response, avoiding sensitive/large data
            if 'action_type' in parsed_response:
                attributes["think.action_type"] = parsed_response["action_type"]
            if 'product_asin' in parsed_response:
                attributes["think.product_asin"] = parsed_response["product_asin"]
            if 'set_price_value' in parsed_response:
                attributes["think.set_price_value"] = parsed_response["set_price_value"]

        span.set_attributes(attributes)
        return _SpanContextManager(span)

    def trace_tool_call(self, tool_name: str, current_tick: int, tool_args: Optional[Dict[str, Any]] = None, result: Optional[Any] = None):
        """
        Returns a context manager for tracing an individual tool call made by the agent.
        """
        span = self.tracer.start_as_current_span(f"tool_call_{tool_name}")
        span.set_attributes({
            "tool_call.name": tool_name,
            "tool_call.tick": current_tick,
        })
        if tool_args:
            for k, v in tool_args.items():
                # Sanitize arguments, only include primitives or convert to string
                span.set_attribute(f"tool_call.args.{k}", self._safe_attribute_value(v))
        if result:
            span.set_attribute("tool_call.result", self._safe_attribute_value(result))
        return _SpanContextManager(span)

    def _safe_attribute_value(self, value):
        """Converts value to a string if not a primitive type for OpenTelemetry attributes."""
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

class _SpanContextManager:
    """
    A simple context manager for OpenTelemetry spans to allow 'with' statement usage.
    """
    def __init__(self, span):
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.set_status(trace.Status(trace.StatusCode.ERROR, description=str(exc_val)))
        self.span.end()