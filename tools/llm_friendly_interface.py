import json

class LLMFriendlyToolWrapper:
    """
    Enhances tool interfaces for LLM agents by providing human-readable summaries,
    contextual examples, schema validation, and formatted responses.
    """

    def wrap_tool_response(self, json_data: dict, tool_name: str) -> str:
        """
        Adds a human-readable summary alongside JSON responses for LLM interpretation.
        """
        summary = self._generate_human_readable_summary(json_data, tool_name)
        return json.dumps({"summary": summary, "data": json_data}, indent=2)

    def generate_usage_examples(self, tool_name: str) -> str:
        """
        Creates few-shot usage examples for agent prompts based on the tool name.
        These are fictional examples for demonstration.
        """
        examples = {
            "create_product": [
                "Example 1: To create a new product named 'EcoWidget' with price 15.99 and stock 100:\n"
                "```json\n"
                "{\"tool_name\": \"create_product\", \"parameters\": {\"name\": \"EcoWidget\", \"price\": 15.99, \"initial_stock\": 100}}\n"
                "```",
                "Example 2: To launch a marketing campaign for 'EcoWidget' with budget 500:\n"
                "```json\n"
                "{\"tool_name\": \"launch_campaign\", \"parameters\": {\"product_id\": \"EcoWidget\", \"budget\": 500, \"type\": \"social_media\"}}\n"
                "```"
            ],
            "get_market_data": [
                "Example 1: To retrieve market data for the 'technology' sector:\n"
                "```json\n"
                "{\"tool_name\": \"get_market_data\", \"parameters\": {\"sector\": \"technology\"}}\n"
                "```"
            ],
            "update_stock": [
                "Example 1: To update the stock of product 'EcoWidget' to 120:\n"
                "```json\n"
                "{\"tool_name\": \"update_stock\", \"parameters\": {\"product_id\": \"EcoWidget\", \"new_stock\": 120}}\n"
                "```"
            ]
        }
        return "\n\n".join(examples.get(tool_name, ["No usage examples available for this tool."]))

    def validate_agent_command(self, command: dict, schema: dict) -> tuple[bool, str]:
        """
        Validates agent commands against a given JSON schema with helpful error messages.
        (Simplified validation, assumes schema is a basic dict with 'required' and 'properties')
        """
        if not isinstance(command, dict):
            return False, "Command must be a JSON object."

        # Check for required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in command:
                return False, f"Missing required field: '{field}'."

        # Check parameter types and presence if properties are defined
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key in command:
                expected_type = prop_schema.get("type")
                if expected_type and not isinstance(command[key], self._get_python_type(expected_type)):
                    return False, f"Invalid type for field '{key}'. Expected '{expected_type}', got '{type(command[key]).__name__}'."
            elif key in required_fields: # This is already covered by the first required field check, but good for clarity on schema validation
                return False, f"Missing required parameter '{key}' in command."

        return True, "Command is valid."

    def format_complex_data(self, data: dict, complexity_level: str = "medium") -> dict:
        """
        Structures complex data for easier LLM interpretation based on complexity level.
        (Placeholder for more sophisticated data summarization/simplification logic)
        """
        if complexity_level == "low":
            return {"overview": "Data simplified to key metrics."}
        elif complexity_level == "medium":
            return {"summary": "Detailed summary of data with key fields.", "data": data}
        elif complexity_level == "high":
            return {"full_data": data}
        return data

    def inject_contextual_help(self, tool_description: str) -> str:
        """
        Adds LLM-specific guidance to a tool description, e.g., on expected formats.
        """
        return (f"{tool_description}\n\n"
                "IMPORTANT: When using this tool, please ensure your parameters are valid JSON "
                "and adhere to the specific types and formats required. "
                "Consider the numerical precision for financial values and string lengths for names.")

    def _generate_human_readable_summary(self, json_data: dict, tool_name: str) -> str:
        """
        Generates a human-readable summary of the tool's JSON response.
        (Simplified, can be expanded with more sophisticated NLP or rule-based logic)
        """
        if tool_name == "create_product":
            if json_data.get("success"):
                product_name = json_data.get("product_name", "a product")
                product_id = json_data.get("product_id", "N/A")
                return f"Successfully created product '{product_name}' with ID '{product_id}'."
            else:
                error = json_data.get("error", "unknown error")
                return f"Failed to create product: {error}."
        elif tool_name == "get_market_data":
            data_count = len(json_data.get("data", []))
            return f"Retrieved market data containing {data_count} entries. Key metrics include average price and demand."
        elif tool_name == "update_stock":
            if json_data.get("success"):
                product_id = json_data.get("product_id", "N/A")
                new_stock = json_data.get("new_stock", "N/A")
                return f"Successfully updated stock for product '{product_id}' to {new_stock}."
            else:
                error = json_data.get("error", "unknown error")
                return f"Failed to update stock: {error}."
        elif "success" in json_data:
            status = "succeeded" if json_data["success"] else "failed"
            return f"The operation for tool '{tool_name}' {status}."
        return f"Response for '{tool_name}': {json.dumps(json_data)}"

    def _get_python_type(self, schema_type: str):
        """Maps JSON schema types to Python types."""
        if schema_type == "string":
            return str
        elif schema_type == "integer":
            return int
        elif schema_type == "number":
            return (int, float)
        elif schema_type == "boolean":
            return bool
        elif schema_type == "array":
            return list
        elif schema_type == "object":
            return dict
        else:
            return None # Unknown type
