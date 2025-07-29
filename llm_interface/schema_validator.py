import json
from jsonschema import validate, ValidationError
from typing import Dict, Any, Optional, Tuple

# JSON schema for the LLM agent's output
LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "asin": {"type": "string"},
                    "price": {"type": "number"},
                    # Other potential action parameters for future actions
                },
                "required": ["type"]
            },
            "minItems": 1
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the agent's decision"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Agent's confidence in its decision (0.0-1.0)"
        }
    },
    "required": ["actions", "reasoning", "confidence"],
    "additionalProperties": False # Ensure no extra fields are added
}

# Define specific action schemas
ACTION_SCHEMAS = {
    "set_price": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["set_price"]},
            "asin": {"type": "string", "pattern": "^B0[0-9A-Z]{8}$"}, # Example ASIN pattern: B0 followed by 8 alphanumeric chars
            "price": {"type": "number", "exclusiveMinimum": 0.0}
        },
        "required": ["type", "asin", "price"],
        "additionalProperties": False
    },
    "wait_next_day": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["wait_next_day"]}
        },
        "required": ["type"],
        "additionalProperties": False
    }
    # Add other action schemas here as they are defined e.g., "adjust_ad_spend", "restock_inventory"
}

def validate_llm_response(response_json: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validates the LLM's response against the predefined schema and action-specific schemas.

    Args:
        response_json: The JSON response from the LLM.

    Returns:
        A tuple: (True, None) if valid, or (False, error_details) if invalid.
        error_details will be a dictionary with 'message', 'path', 'validator', 'validation_error'.
    """
    try:
        # Validate against the overall LLM response schema
        validate(instance=response_json, schema=LLM_RESPONSE_SCHEMA)

        # Validate each action against its specific schema
        for i, action in enumerate(response_json.get("actions", [])):
            action_type = action.get("type")
            if action_type not in ACTION_SCHEMAS:
                return False, {
                    "error_type": "SchemaViolation",
                    "message": f"Unknown action type: '{action_type}'",
                    "path": f"actions/{i}/type",
                    "invalid_value": action_type,
                    "suggested_fix": f"Action type '{action_type}' is not recognized. Check available actions.",
                }
            
            try:
                validate(instance=action, schema=ACTION_SCHEMAS[action_type])
            except ValidationError as e:
                # Extract more user-friendly error details
                return False, {
                    "error_type": "SchemaViolation",
                    "message": e.message,
                    "path": "/".join([str(p) for p in e.path]),
                    "validator": e.validator,
                    "validator_value": e.validator_value,
                    "invalid_value": e.instance,
                    "suggested_fix": f"Ensure '{e.path[-1]}' field is correctly formatted for '{action_type}' action. Expected: {e.validator_value}",
                    "validation_error": str(e) # Keep original error for deeper debugging
                }

    except ValidationError as e:
        # Handle overall schema validation errors
        return False, {
            "error_type": "SchemaViolation",
            "message": e.message,
            "path": "/".join([str(p) for p in e.path]),
            "validator": e.validator,
            "validator_value": e.validator_value,
            "invalid_value": e.instance,
            "suggested_fix": f"Ensure outermost JSON structure conforms to expected format. Field '{e.path[-1] or 'root'}' is incorrect.",
            "validation_error": str(e)
        }
    except Exception as e:
        # Catch any other unexpected errors during validation
        return False, {
            "error_type": "InternalValidationError",
            "message": f"An unexpected error occurred during validation: {str(e)}",
            "validation_error": str(e)
        }
    
    return True, None # Response is valid