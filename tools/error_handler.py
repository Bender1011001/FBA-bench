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
        Logs the error and returns a structured error response.
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
        Generates educational error messages to help agents learn from mistakes.
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
        Recommends fixes for common mistakes in agent commands.
        (Simplified: focuses on common JSON errors; can be expanded with more NLP for intent)
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

            # Add general structure suggestion if none of the above caught specific issues
            if not suggestions:
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
    Handle common errors for agents in a standardized way.
    
    Args:
        error: The exception that occurred
        context: Context information about the error
        
    Returns:
        Dictionary with error handling information
    """
    error_handler = AgentErrorHandler()
    
    # Generate educational feedback
    feedback = error_handler.generate_error_feedback(error, context)
    
    # Create standardized error response
    error_response = {
        "status": "error",
        "error_type": type(error).__name__,
        "message": str(error),
        "feedback": feedback,
        "context": context,
        "suggestions": []
    }
    
    # Add specific suggestions based on error type
    if isinstance(error, (ValueError, TypeError)):
        error_response["suggestions"].append("Check parameter types and values")
    elif isinstance(error, KeyError):
        error_response["suggestions"].append("Verify all required keys are present")
    elif isinstance(error, AttributeError):
        error_response["suggestions"].append("Check if the attribute exists on the object")
    
    return error_response

    def log_error_patterns(self, agent_id: str, error_history: List[Dict[str, Any]]):
        """
        Tracks recurring agent mistakes for later analysis and learning.
        Aggregates error patterns and logs them for monitoring and improvement.
        """
        if not error_history:
            logging.info(f"Agent {agent_id} has no error history to analyze.")
            return
            
        logging.info(f"Agent {agent_id} error history snapshot: {len(error_history)} errors recorded.")
        
        # Count common error types
        error_types = [e.get("error_type", "unknown") for e in error_history if e.get("error_type")]
        count_errors = Counter(error_types)
        
        # Log the most common errors
        if count_errors:
            logging.info(f"Most common errors for Agent {agent_id}:")
            for error_type, count in count_errors.most_common(5):  # Log top 5 errors
                logging.info(f"  - {error_type}: {count} occurrences")
        
        # Log error frequency over time if timestamps are available
        timestamps = [e.get("timestamp") for e in error_history if e.get("timestamp")]
        if timestamps:
            logging.info(f"Error timestamps available for {len(timestamps)} errors. "
                        "Consider implementing time-based analysis for error patterns.")
        
        # Note: In a production system, this data would be stored to a database
        # for more sophisticated analysis and potential alerting.


class BenchmarkError(Exception):
    """Exception raised for benchmark-related errors."""
    pass