import json
import logging
from typing import Dict, Any, List
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentErrorHandler:
    """
    Provides robust error handling for LLM agent commands, including pre-validation,
    graceful degradation, educational error feedback, and recovery suggestions.
    """

    def validate_command_syntax(self, command_json: str) -> tuple[bool, str]:
        """
        Checks the JSON structure and required fields of an agent command.
        Returns True if valid, False otherwise with an error message.
        """
        try:
            command = json.loads(command_json)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {e}"

        if not isinstance(command, dict):
            return False, "Command must be a JSON object."
        if "tool_name" not in command:
            return False, "Missing 'tool_name' field."
        if "parameters" not in command:
            return False, "Missing 'parameters' field."
        if not isinstance(command["parameters"], dict):
            return False, "'parameters' must be a JSON object."

        return True, "Syntax is valid."

    def handle_invalid_command(self, command: str, error_type: str) -> Dict[str, Any]:
        """
        Processes malformed commands gracefully without simulation crashes.
        Logs the error and returns a structured error response, ensuring continuous operation
        even with malformed LLM agent inputs.
        """
        logging.error(f"Malformed command received (Type: {error_type}): {command}")
        return {
            "status": "error",
            "error_type": error_type,
            "message": f"The command provided was malformed or invalid. Error type: {error_type}.",
            "original_command": command
        }

    def generate_error_feedback(self, error: Exception, context: Dict[str, Any]) -> str:
        """
        Generates actionable and educational error messages to help LLM agents learn from mistakes.
        This structured feedback is crucial for iterative improvement of agent behavior.
        """
        error_message = str(error)
        tool_name = context.get("tool_name", "an unknown tool")
        command_params = context.get("parameters", {})

        feedback = f"Error during execution of tool '{tool_name}'. Details: {error_message}. "

        if "missing required" in error_message.lower() or "invalid type" in error_message.lower():
            feedback += "This often means you are missing a required parameter or provided an incorrect data type. " \
                        "Please check the tool's schema and usage examples carefully."
        elif "authentication" in error_message.lower():
            feedback += "There was an authentication issue. Ensure you have the necessary permissions. "
        elif "rate limit" in error_message.lower():
            feedback += "You have hit a rate limit. Please try again after some time."
        elif "network" in error_message.lower():
            feedback += "A network error occurred. Check your connection or try again."
        elif "value out of range" in error_message.lower():
            feedback += "One of the provided values is out of an acceptable range. Review valid ranges for parameters."
        else:
            feedback += "Review the tool's documentation for correct usage."

        feedback += f"\nYour parameters were: {json.dumps(command_params)}."
        return feedback

    def suggest_command_corrections(self, invalid_command: str) -> List[str]:
        """
        Provides specific, actionable recommendations for fixing common mistakes in agent commands.
        This aims to guide the LLM agent towards a correct invocation pattern.
        """
        suggestions = []
        try:
            # Attempt to parse to identify common malformations
            command = json.loads(invalid_command)
            if not isinstance(command, dict):
                suggestions.append("Command should be a JSON object, e.g., `{\"tool_name\": \"...\", \"parameters\": {...}}`.")
            if "tool_name" not in command:
                suggestions.append("Missing `\"tool_name\"` key. Ensure commands are structured like `{\"tool_name\": \"your_tool\", \"parameters\": {...}}`.")
            if "parameters" not in command:
                suggestions.append("Missing `\"parameters\"` key. Ensure parameters are nested under `\"parameters\": {...}`.")
            elif not isinstance(command.get("parameters"), dict):
                suggestions.append("The `\"parameters\"` value must be a JSON object.")

            if not suggestions: # Only add general suggestions if more specific ones weren't found
                suggestions.append("Ensure all string values are enclosed in double quotes and commas separate elements in lists/objects.")

        except json.JSONDecodeError:
            suggestions.append("Double check that your command is valid JSON. Pay attention to missing commas, unclosed quotes, or improper brackets/braces.")
            suggestions.append("Ensure keys and string values are always enclosed in double quotes, not single quotes.")
            suggestions.append("If you are experiencing issues with newlines, try to remove them or escape them properly.")

        if not suggestions:
            suggestions.append("Review the tool's expected JSON format and parameter types.")

        return suggestions


def handle_common_errors_for_agent(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Centralized error handling function for LLM agents, providing standardized, educational feedback.
    This function acts as a uniform interface for all errors encountered during agent operations.
    
    Args:
        error: The exception that occurred
        context: Context information about the error, typically including 'tool_name' and 'parameters'.
        
    Returns:
        A structured dictionary containing error details, educational feedback, and actionable suggestions.
    """
    error_handler = AgentErrorHandler()
    
    feedback = error_handler.generate_error_feedback(error, context)
    
    error_response = {
        "status": "error",
        "error_type": type(error).__name__,
        "message": str(error),
        "feedback": feedback,
        "context": context,
        "suggestions": []
    }
    
    if isinstance(error, (ValueError, TypeError)):
        error_response["suggestions"].append("Check parameter types and values. Ensure numbers are numeric, and strings are properly quoted.")
    elif isinstance(error, KeyError):
        error_response["suggestions"].append("Verify all required keys are present in the command's 'parameters' object.")
    elif isinstance(error, AttributeError):
        error_response["suggestions"].append("Check if the attribute or method exists on the target object. This may indicate a schema mismatch.")
    elif isinstance(error, json.JSONDecodeError):
        error_response["suggestions"].append("Review the JSON syntax of your command. Look for unclosed quotes, missing commas, or incorrect bracing.")
    elif "connection" in str(error).lower() or "network" in str(error).lower():
         error_response["suggestions"].append("A network issue prevented the operation. Verify your internet connection or API endpoint accessibility.")
    elif "authentication" in str(error).lower() or "api key" in str(error).lower():
         error_response["suggestions"].append("Authentication failed. Ensure your API key is correct and has the necessary permissions.")
    
    return error_response

    def log_error_patterns(self, agent_id: str, error_history: List[Dict[str, Any]]):
        """
        Analyzes and logs recurring error patterns for a given agent from its error history.
        This provides insights into common agent mistakes and informs agent design improvements.
        """
        if not error_history:
            logging.info(f"Agent {agent_id} has no error history to analyze.")
            return
            
        logging.info(f"Analyzing error history for Agent {agent_id}: {len(error_history)} errors recorded.")
        
        error_types = [e.get("error_type", "unknown") for e in error_history if e.get("error_type")]
        count_errors = Counter(error_types)
        
        if count_errors:
            logging.info(f"Top 5 most common error types for Agent {agent_id}:")
            for error_type, count in count_errors.most_common(5):
                logging.info(f"  - {error_type}: {count} occurrences")
        
        timestamps = [e.get("timestamp") for e in error_history if e.get("timestamp")]
        if timestamps:
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            if time_diffs:
                avg_interval = sum(time_diffs) / len(time_diffs)
                logging.info(f"Average interval between errors for Agent {agent_id}: {avg_interval:.2f} seconds.")
            else:
                logging.info(f"Only one error recorded for Agent {agent_id}, cannot determine interval.")
        
        # Further analysis could include:
        # - Correlation of errors with specific tools or contexts.
        # - Trend analysis over longer periods.
        # - Automated alerting for persistent critical errors.
        # This data would typically be stored in a time-series database for robust analytics.


class BenchmarkError(Exception):
    """Custom exception raised for benchmark-related errors during execution."""
    pass