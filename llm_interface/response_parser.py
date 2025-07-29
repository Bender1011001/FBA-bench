import json
import logging
from typing import Dict, Any, Optional, Tuple

from llm_interface.schema_validator import validate_llm_response, LLM_RESPONSE_SCHEMA, ACTION_SCHEMAS
from metrics.trust_metrics import TrustMetrics # Assuming TrustMetrics can be imported and has an apply_penalty method

logger = logging.getLogger(__name__)

class LLMResponseParser:
    """
    Parses and validates LLM responses, handling schema violations and
    applying trust score penalties.
    """
    def __init__(self, trust_metrics: TrustMetrics):
        self.trust_metrics = trust_metrics
        self.LLM_RESPONSE_SCHEMA = LLM_RESPONSE_SCHEMA # Expose for reference

    def parse_and_validate(self, raw_llm_response: str, agent_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Parses the raw LLM string response into JSON and validates it against the schema.
        Applies trust score penalties for invalid responses.

        Args:
            raw_llm_response: The raw string response received from the LLM.
            agent_id: The ID of the agent that generated this response, for logging/penalties.

        Returns:
            A tuple: (parsed_json, error_details).
            parsed_json will be the validated dictionary if successful, None otherwise.
            error_details will be a dictionary with error info if invalid, None otherwise.
        """
        parsed_json = None
        error_details = None

        try:
            # Attempt to parse the raw response as JSON
            parsed_json = json.loads(raw_llm_response)
        except json.JSONDecodeError as e:
            error_details = {
                "error_type": "JSONParsingError",
                "message": f"Malformed JSON response: {e.msg}. Raw response: {raw_llm_response[:200]}...",
                "suggested_fix": "Ensure the entire response is a single, valid JSON object.",
                "trust_score_penalty": -0.1, # Penalty for severe parsing errors
                "parsing_error": str(e)
            }
            logger.error(f"[{agent_id}] JSON Parsing Error: {error_details['message']}")
            self._apply_trust_penalty(agent_id, error_details["trust_score_penalty"], error_details["message"])
            return None, error_details
        except Exception as e:
            error_details = {
                "error_type": "UnexpectedParsingError",
                "message": f"An unexpected error occurred during JSON parsing: {str(e)}. Raw response: {raw_llm_response[:200]}...",
                "suggested_fix": "Review the LLM's raw output for unexpected characters or formatting.",
                "trust_score_penalty": -0.15, # Higher penalty for unexpected system errors
                "parsing_error": str(e)
            }
            logger.error(f"[{agent_id}] Unexpected Parsing Error: {error_details['message']}")
            self._apply_trust_penalty(agent_id, error_details["trust_score_penalty"], error_details["message"])
            return None, error_details

        # If JSON parsing was successful, validate against the schema
        is_valid, validation_error_info = validate_llm_response(parsed_json)

        if not is_valid:
            error_details = {
                "error_type": "SchemaViolation",
                "message": f"LLM response failed schema validation: {validation_error_info.get('message', 'N/A')}",
                "path": validation_error_info.get('path', 'N/A'),
                "invalid_value": validation_error_info.get('invalid_value', 'N/A'),
                "suggested_fix": validation_error_info.get('suggested_fix', 'Review schema documentation.'),
                "trust_score_penalty": -0.05, # Penalty for schema violations
                "validation_details": validation_error_info # Full original validation error details
            }
            logger.warning(f"[{agent_id}] Schema Violation: {error_details['message']} at {error_details['path']}. Value: {error_details['invalid_value']}")
            self._apply_trust_penalty(agent_id, error_details["trust_score_penalty"], error_details["message"])
            return None, error_details
        
        logger.info(f"[{agent_id}] LLM response successfully parsed and validated.")
        return parsed_json, None

    def _apply_trust_penalty(self, agent_id: str, penalty_amount: float, reason: str):
        """
        Applies a penalty to the agent's trust score.
        """
        # Assuming trust_metrics has a method to apply penalties
        # The agent_id could be used by TrustMetrics to track individual agent performance
        self.trust_metrics.apply_penalty(agent_id=agent_id, penalty_amount=penalty_amount, reason=reason)
        logger.info(f"[{agent_id}] Applied trust score penalty of {penalty_amount} for reason: {reason}")