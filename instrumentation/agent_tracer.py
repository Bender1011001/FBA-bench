from opentelemetry import trace
from opentelemetry.trace import Tracer, NonRecordingSpan
from typing import Optional, Dict, Any, List
import json

class AgentTracer:
    """
    Handles OpenTelemetry tracing for agent-specific actions (observe, think, tool_call).
    Each tracing method returns a context manager that should be used with a 'with' statement.
    Includes enhancements for LLM reasoning, tool usage, error context, and performance.
    """
    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def trace_agent_turn(self, agent_id: str, tick: int, agent_type: str = "advanced_agent"):
        """
        Returns a context manager for tracing a complete agent turn.
        """
        span_name = f"agent_turn_{agent_id}_tick_{tick}"
        parent_span = trace.get_current_span()
        
        # Create a span context manager with the attributes
        context_manager = self.tracer.start_as_current_span(
            span_name,
            attributes={
                "agent.id": agent_id,
                "simulation.tick": tick,
                "agent.type": agent_type,
                "parent_span_id": str(parent_span._context.span_id) if parent_span and hasattr(parent_span, '_context') else "N/A"
            }
        )
        
        # Get the actual span from the context manager
        span = context_manager.__enter__() if hasattr(context_manager, '__enter__') else None
        if span:
            try:
                span.add_event("Agent turn started")
            except AttributeError:
                # If span doesn't have add_event method, just continue
                pass
        
        return _SpanContextManager(span or context_manager)

    def trace_observe_phase(self, current_tick: int, event_count: int, observed_events_summary: Optional[str] = None):
        """
        Returns a context manager for tracing the agent's observation phase.

        Args:
            current_tick: The current simulation tick.
            event_count: The number of events observed.
            observed_events_summary: A concise summary of the observed events.
        """
        span = self.tracer.start_as_current_span("observe")
        attributes = {
            "observe.tick": current_tick,
            "observe.event_count": event_count
        }
        if observed_events_summary:
            attributes["observe.events_summary"] = observed_events_summary
        span.set_attributes(attributes)
        return _SpanContextManager(span)

    def trace_think_phase(self, current_tick: int, llm_model: str, llm_tokens_used: int,
                          decision_confidence: Optional[float] = None, parsed_response: Optional[Dict[str, Any]] = None,
                          llm_reasoning: Optional[str] = None):
        """
        Returns a context manager for tracing the agent's thinking/decision-making phase.

        Args:
            current_tick: The current simulation tick.
            llm_model: The LLM model used for thinking.
            llm_tokens_used: The number of tokens used by the LLM.
            decision_confidence: Confidence score of the agent's decision.
            parsed_response: The structured response from the LLM.
            llm_reasoning: The raw LLM thought process or reasoning chain.
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
            # Capture full parsed response as a JSON string for detailed analysis if needed
            attributes["think.parsed_response_json"] = json.dumps(parsed_response)

        if llm_reasoning:
            attributes["llm.reasoning"] = llm_reasoning # Capture LLM thought process

        span.set_attributes(attributes)
        return _SpanContextManager(span)

    def trace_tool_call(self, tool_name: str, current_tick: int, tool_args: Optional[Dict[str, Any]] = None,
                        result: Optional[Any] = None, success: Optional[bool] = None, error_details: Optional[Dict[str, Any]] = None):
        """
        Returns a context manager for tracing an individual tool call made by the agent.

        Args:
            tool_name: The name of the tool called.
            current_tick: The current simulation tick.
            tool_args: The arguments passed to the tool.
            result: The result returned by the tool (if successful).
            success: Boolean indicating if the tool call was successful.
            error_details: Dictionary containing error information if the call failed.
        """
        span = self.tracer.start_as_current_span(f"tool_call_{tool_name}")
        attributes = {
            "tool_call.name": tool_name,
            "tool_call.tick": current_tick,
        }
        if success is not None:
            attributes["tool_call.success"] = success
        if tool_args:
            # Sanitize arguments, only include primitives or convert to string
            for k, v in tool_args.items():
                attributes[f"tool_call.args.{k}"] = self._safe_attribute_value(v)
        if result:
            attributes["tool_call.result_summary"] = self._safe_attribute_value(result) # Summary to avoid large fields
            attributes["tool_call.result_json"] = json.dumps(result) # Full result as JSON string if detailed capture is needed

        if error_details:
            attributes["tool_call.error_type"] = error_details.get("type", "unknown")
            attributes["tool_call.error_message"] = error_details.get("message", "N/A")
            attributes["tool_call.error_full_details"] = json.dumps(error_details) # Full error context

        span.set_attributes(attributes)
        return _SpanContextManager(span)

    def _safe_attribute_value(self, value):
        """Converts value to a string if not a primitive type for OpenTelemetry attributes."""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (dict, list)): # Convert dicts/lists to JSON string
            try:
                return json.dumps(value)
            except TypeError:
                return repr(value) # Fallback if not JSON serializable
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