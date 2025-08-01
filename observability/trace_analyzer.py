from typing import Dict, Any, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TraceAnalyzer:
    """
    Analyzes simulation trace data to identify root causes of failures,
    detect patterns, and pinpoint performance bottlenecks.
    """

    def analyze_simulation_failure(self, trace_data: List[Dict[str, Any]], failure_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs automated post-mortem analysis of failed simulations to identify root causes.

        Args:
            trace_data: A list of chronologically ordered trace events.
            failure_point: The specific event or state where the failure occurred.

        Returns:
            A dictionary containing the analysis report.
        """
        analysis_report = {
            "root_cause_candidates": [],
            "failure_point_details": failure_point,
            "nearby_events_context": []
        }

        failure_timestamp = failure_point.get("timestamp")
        if not failure_timestamp:
            logging.warning("Failure point has no timestamp, limiting context analysis.")
            return analysis_report

        # Look for events immediately preceding the failure
        context_window_seconds = 5  # Analyze events in the 5 seconds leading to failure
        for event in reversed(trace_data):
            event_timestamp = event.get("timestamp")
            if event_timestamp and (failure_timestamp - event_timestamp) <= context_window_seconds and event_timestamp <= failure_timestamp:
                analysis_report["nearby_events_context"].insert(0, event)
            if event_timestamp and (failure_timestamp - event_timestamp) > context_window_seconds:
                break # Passed our context window

        # Simple heuristic for root cause: Look for errors/warnings in the context
        for event in analysis_report["nearby_events_context"]:
            if event.get("event_type") in ["error", "warning"] or "error" in event.get("details", {}):
                analysis_report["root_cause_candidates"].append(event)
        
        if not analysis_report["root_cause_candidates"]:
            analysis_report["root_cause_candidates"].append({"message": "No explicit error/warning events found immediately preceding failure. Further manual investigation recommended."})

        logging.info(f"Analyzed simulation failure. Root cause candidates: {len(analysis_report['root_cause_candidates'])}")
        return analysis_report

    def detect_behavioral_patterns(self, agent_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identifies common failure modes and success patterns from agent decision traces.

        Args:
            agent_traces: A list of traces, each representing an agent's decisions/actions.

        Returns:
            A dictionary summarizing detected behavioral patterns.
        """
        patterns = {
            "common_actions": {},
            "common_errors": {},
            "successful_sequences": [],
            "failed_sequences": []
        }

        for trace in agent_traces:
            agent_id = trace.get("agent_id", "unknown_agent")
            decisions = trace.get("decisions", []) # Assuming decisions is a list of dicts with 'action' and 'status'
            
            sequence_of_actions = []
            for decision in decisions:
                action = decision.get("action")
                status = decision.get("status")
                
                if action:
                    patterns["common_actions"][action] = patterns["common_actions"].get(action, 0) + 1
                    sequence_of_actions.append(action)

                if status == "failure" and decision.get("error_details"):
                    error_type = decision.get("error_details", {}).get("type", "generic_error")
                    patterns["common_errors"][error_type] = patterns["common_errors"].get(error_type, 0) + 1
            
            # Simple sequence tracking; can be enhanced with n-gram analysis
            if decisions and decisions[-1].get("status") == "success":
                patterns["successful_sequences"].append(sequence_of_actions)
            elif decisions and decisions[-1].get("status") == "failure":
                patterns["failed_sequences"].append(sequence_of_actions)

        logging.info("Detected behavioral patterns from agent traces.")
        return patterns

    def identify_performance_bottlenecks(self, timing_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Locates slow operations and suggests optimizations based on timing data.

        Args:
            timing_data: A list of dictionaries, each with 'operation', 'duration_ms', 'timestamp'.

        Returns:
            A dictionary summarizing identified bottlenecks.
        """
        bottlenecks = {
            "slow_operations": [],
            "average_durations": {}
        }

        operation_durations: Dict[str, List[float]] = {}
        for entry in timing_data:
            op = entry.get("operation")
            duration = entry.get("duration_ms")
            if op and isinstance(duration, (int, float)):
                operation_durations.setdefault(op, []).append(duration)
        
        for op, durations in operation_durations.items():
            avg_duration = sum(durations) / len(durations)
            bottlenecks["average_durations"][op] = avg_duration
            
            # Simple threshold for slow operations (e.g., >100ms average)
            if avg_duration > 100:
                bottlenecks["slow_operations"].append({"operation": op, "average_duration_ms": avg_duration})
        
        bottlenecks["slow_operations"].sort(key=lambda x: x["average_duration_ms"], reverse=True)
        logging.info(f"Identified {len(bottlenecks['slow_operations'])} performance bottlenecks.")
        return bottlenecks

    def generate_insight_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Creates a human-readable analysis report from various analysis results.
        """
        report_parts = ["--- Trace Analysis Insight Report ---"]

        # Failure Analysis Summary
        failure_analysis = analysis_results.get("failure_analysis", {})
        if failure_analysis:
            report_parts.append("\n### 1. Simulation Failure Analysis ###")
            report_parts.append(f"Failure Point: {json.dumps(failure_analysis.get('failure_point_details'), indent=2)}")
            report_parts.append("\nRoot Cause Candidates:")
            for rc in failure_analysis.get("root_cause_candidates", []):
                report_parts.append(f"- {json.dumps(rc, indent=2)}")
            report_parts.append("\nNearby Events Context:")
            for event in failure_analysis.get("nearby_events_context", []):
                report_parts.append(f"- [{event.get('timestamp')}] {event.get('event_type')}: {event.get('details')}")

        # Behavioral Patterns Summary
        behavioral_patterns = analysis_results.get("behavioral_patterns", {})
        if behavioral_patterns:
            report_parts.append("\n### 2. Agent Behavioral Patterns ###")
            report_parts.append("\nCommon Actions:")
            for action, count in behavioral_patterns.get("common_actions", {}).items():
                report_parts.append(f"- {action}: {count} times")
            report_parts.append("\nCommon Errors:")
            for error_type, count in behavioral_patterns.get("common_errors", {}).items():
                report_parts.append(f"- {error_type}: {count} times")
            
            # Limited display for sequences to avoid overwhelming
            report_parts.append("\nSample Successful Sequences:")
            for seq in behavioral_patterns.get("successful_sequences", [])[:3]: # Show top 3
                report_parts.append(f"- {'>'.join(seq)}")
            report_parts.append("\nSample Failed Sequences:")
            for seq in behavioral_patterns.get("failed_sequences", [])[:3]: # Show top 3
                report_parts.append(f"- {'>'.join(seq)}")

        # Performance Bottlenecks Summary
        performance_bottlenecks = analysis_results.get("performance_bottlenecks", {})
        if performance_bottlenecks:
            report_parts.append("\n### 3. Performance Bottlenecks ###")
            if performance_bottlenecks.get("slow_operations"):
                report_parts.append("\nTop Slow Operations (Avg Duration):")
                for op in performance_bottlenecks.get("slow_operations", [])[:5]: # Show top 5
                    report_parts.append(f"- {op['operation']}: {op['average_duration_ms']:.2f} ms")
            else:
                report_parts.append("No significant slow operations detected above threshold.")

            report_parts.append("\nAverage Operation Durations (selected):")
            avg_ops = list(performance_bottlenecks.get("average_durations", {}).items())[:5] # Show first 5
            for op, duration in avg_ops:
                report_parts.append(f"- {op}: {duration:.2f} ms")


        report_parts.append("\n--- End of Report ---")
        logging.info("Generated insight report.")
        return "\n".join(report_parts)

    def recommend_optimizations(self, bottlenecks: Dict[str, Any]) -> List[str]:
        """
        Suggests performance improvements based on identified bottlenecks.
        """
        recommendations = []
        slow_ops = bottlenecks.get("slow_operations", [])

        if not slow_ops:
            recommendations.append("No significant bottlenecks identified. System appears performant.")
            return recommendations

        recommendations.append("Recommended Optimizations based on Performance Bottlenecks:")
        for op_info in slow_ops:
            op_name = op_info.get("operation")
            avg_duration = op_info.get("average_duration_ms")

            if "database" in op_name.lower():
                recommendations.append(f"- Optimize database queries for '{op_name}'. Consider indexing, caching, or reducing query complexity.")
            elif "llm" in op_name.lower() or "inference" in op_name.lower():
                recommendations.append(f"- For '{op_name}' (LLM/Inference): Explore model quantization, batching, using smaller models for less critical tasks, or optimizing prompt structure.")
            elif "api_call" in op_name.lower() or "external_service" in op_name.lower():
                recommendations.append(f"- For '{op_name}' (API/External Service): Implement aggressive caching, retry mechanisms, or consider parallelizing calls if possible.")
            elif "data_processing" in op_name.lower():
                recommendations.append(f"- For '{op_name}' (Data Processing): Optimize algorithms, use more efficient data structures, or consider parallel processing.")
            else:
                recommendations.append(f"- Investigate '{op_name}' (avg {avg_duration:.2f}ms) for potential algorithmic or I/O inefficiencies.")

        logging.info("Generated optimization recommendations.")
        return recommendations
