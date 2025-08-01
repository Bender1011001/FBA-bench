import inspect
import json
import os
import re
from typing import Any, Dict, List, Type, Union

class ToolDocGenerator:
    """
    Auto-generates LLM-friendly tool documentation from code, including
    usage examples, error scenarios, and integration guides.
    """

    def generate_tool_docs(self, tool_module: Any) -> Dict[str, Any]:
        """
        Creates comprehensive documentation for all functions within a given tool module.
        """
        docs = {"module_name": tool_module.__name__, "tools": []}
        
        for name, obj in inspect.getmembers(tool_module):
            if inspect.isfunction(obj) and not name.startswith("_"): # Exclude private/helper functions
                tool_doc = self._document_function(obj)
                docs["tools"].append(tool_doc)
        
        return docs

    def _document_function(self, func: Any) -> Dict[str, Any]:
        """Documents a single function, extracting its name, docstring, and parameters."""
        docstring = inspect.getdoc(func) or "No description available."
        signature = inspect.signature(func)
        parameters = []
        for param_name, param in signature.parameters.items():
            if param_name == 'self': # Skip 'self' for methods
                continue
            param_type = str(param.annotation).replace("<class '", "").replace("'>", "")
            parameters.append({
                "name": param_name,
                "type": param_type,
                "default": str(param.default) if param.default is not inspect.Parameter.empty else "N/A"
            })
        
        return {
            "name": func.__name__,
            "description": docstring,
            "parameters": parameters,
            "return_type": str(signature.return_annotation).replace("<class '", "").replace("'>", "")
        }

    def extract_usage_examples(self, tool_functions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Generates practical usage examples for each tool function.
        (Simplified: uses placeholders/mock examples)
        """
        examples = {}
        for tool in tool_functions:
            tool_name = tool["name"]
            example_list = []
            
            # Basic example structure based on tool name
            if "create_product" in tool_name:
                example_list.append(f"Example: `{{\"tool_name\": \"{tool_name}\", \"parameters\": {{\"name\": \"NewGadget\", \"price\": 29.99, \"stock\": 100}}}}` - Creates a new product.")
            elif "update_stock" in tool_name:
                example_list.append(f"Example: `{{\"tool_name\": \"{tool_name}\", \"parameters\": {{\"product_id\": \"prod123\", \"new_quantity\": 50}}}}` - Updates product stock.")
            elif "get_data" in tool_name or "fetch_data" in tool_name:
                example_list.append(f"Example: `{{\"tool_name\": \"{tool_name}\", \"parameters\": {{\"query\": \"sales_metrics\", \"timeframe\": \"last_month\"}}}}` - Retrieves data.")
            else:
                example_list.append(f"Example: `{{\"tool_name\": \"{tool_name}\", \"parameters\": {{...}}}}` - Basic usage.")
            
            examples[tool_name] = example_list
        return examples

    def document_error_scenarios(self, error_patterns: Dict[str, int]) -> Dict[str, str]:
        """
        Documents common mistakes and how to avoid them based on observed error patterns.
        (error_patterns would come from AgentErrorHandler.log_error_patterns)
        """
        scenarios = {}
        if "Invalid JSON format" in error_patterns:
            scenarios["Invalid JSON Format"] = "**Description**: The command sent by the agent was not valid JSON. **How to Avoid**: Always ensure your command is a properly formatted JSON string, with all keys and string values enclosed in double quotes. Use online JSON validators if unsure."
        if "Missing 'tool_name' field" in error_patterns:
            scenarios["Missing Tool Name"] = "**Description**: The command JSON was missing the `tool_name` field. **How to Avoid**: Every command must specify the `\"tool_name\"` key at the top level of the JSON object."
        if "Missing 'parameters' field" in error_patterns:
            scenarios["Missing Parameters Object"] = "**Description**: The command JSON was missing the `parameters` field. **How to Avoid**: All tool arguments must be nested within a `\"parameters\": {{...}}` object."
        if "Invalid type for field" in error_patterns:
            scenarios["Incorrect Parameter Type"] = "**Description**: A parameter was provided with an incorrect data type (e.g., string instead of number). **How to Avoid**: Refer to the tool's schema to understand the expected data type for each parameter (e.g., `integer`, `string`, `number`, `boolean`)."
        if "Rate Limit Exceeded" in error_patterns:
            scenarios["Rate Limit Exceeded"] = "**Description**: The agent made too many requests to a tool within a short period. **How to Avoid**: Implement back-off strategies or re-evaluate the necessity of frequent calls."
        
        if not scenarios:
            scenarios["No Common Errors"] = "No specific common error patterns detected yet. Ensure robust error logging is in place."

        return scenarios

    def create_integration_guide(self, tool_set: Dict[str, Any]) -> str:
        """
        Generates a step-by-step guide for integrating and using a set of tools.
        """
        guide_parts = ["--- Tool Integration Guide ---"]
        guide_parts.append("\nThis guide provides instructions for integrating and effectively using the available tools.\n")
        
        guide_parts.append("### 1. Understanding Tool Structure ###")
        guide_parts.append("All tool commands should be sent as JSON objects with two top-level keys:")
        guide_parts.append("- `tool_name`: A string identifying the tool to be used (e.g., \"create_product\").")
        guide_parts.append("- `parameters`: A JSON object containing all arguments for the tool's function. Each key-value pair inside `parameters` corresponds to a function argument and its value.")
        guide_parts.append("\nExample Generic Command:")
        guide_parts.append("```json")
        guide_parts.append("{\n  \"tool_name\": \"your_tool_name\",\n  \"parameters\": {\n    \"arg1\": \"value1\",\n    \"arg2\": 123,\n    \"optional_arg\": true\n  }\n}")
        guide_parts.append("```")

        guide_parts.append("\n### 2. Available Tool Functions ###")
        for module_name, module_docs in tool_set.items():
            guide_parts.append(f"\n#### Module: `{module_name}` ####")
            for tool in module_docs.get("tools", []):
                guide_parts.append(f"\n##### Tool: `{tool['name']}` #####")
                guide_parts.append(f"Description: {tool['description']}")
                guide_parts.append("Parameters:")
                for param in tool["parameters"]:
                    guide_parts.append(f"- `{param['name']}`: Type: `{param['type']}`, Default: `{param['default']}`")
                
                # Add usage examples specific to this tool
                examples = self.extract_usage_examples([tool]).get(tool['name'], [])
                if examples:
                    guide_parts.append("Usage Examples:")
                    for ex in examples:
                        guide_parts.append(f"  {ex}")
                
        guide_parts.append("\n### 3. Error Handling and Recovery ###")
        guide_parts.append("When a tool execution fails, the system provides structured error feedback. Pay attention to:")
        guide_parts.append("- **`error_type`**: Indicates the category of error (e.g., `JSON_SCHEMA_ERROR`, `RUNTIME_ERROR`).")
        guide_parts.append("- **`message`**: A detailed description of what went wrong.")
        guide_parts.append("- **`suggestions`**: Often includes specific recommendations on how to correct your command or approach.")
        guide_parts.append("Example Error Response (simplified):")
        guide_parts.append("```json")
        guide_parts.append("{\n  \"status\": \"error\",\n  \"error_type\": \"JSON_SCHEMA_ERROR\",\n  \"message\": \"Missing required field: 'product_id'.\",\n  \"suggestions\": [\"Ensure 'product_id' is present in parameters.\"]\n}")
        guide_parts.append("```")

        guide_parts.append("\n### 4. Best Practices for LLM Agents ###")
        guide_parts.append("- **Be Precise**: Always provide exact JSON as required.")
        guide_parts.append("- **Validate Inputs**: Before forming a command, ensure you have all necessary information and that values are of the correct type and within valid ranges.")
        guide_parts.append("- **Learn from Errors**: Utilize the error feedback and suggestions to refine your future commands.")
        guide_parts.append("- **Use Few-Shot Examples**: Leverage the provided usage examples to guide your command generation, especially for new or complex tools.")
        
        guide_parts.append("\n--- End of Guide ---")
        return "\n".join(guide_parts)

    def export_documentation(self, docs_content: str, format: str = "markdown", output_path: str = "tool_documentation") -> str:
        """
        Exports documentation content to a specified format and output path.
        Supports markdown by default.
        """
        full_path = ""
        if format == "markdown":
            full_path = f"{output_path}.md"
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(docs_content)
            return f"Documentation exported to {full_path}"
        elif format == "json":
            full_path = f"{output_path}.json"
            # Assuming docs_content is already a dict or can be converted
            try:
                if isinstance(docs_content, str):
                    parsed_content = json.loads(docs_content) # Assuming JSON string
                else:
                    parsed_content = docs_content
                with open(full_path, "w", encoding="utf-8") as f:
                    json.dump(parsed_content, f, indent=2)
                return f"Documentation exported to {full_path}"
            except json.JSONDecodeError:
                return "Error: Could not export to JSON. Content is not valid JSON."
        else:
            return f"Error: Unsupported format '{format}'. Supported formats: 'markdown', 'json'."
