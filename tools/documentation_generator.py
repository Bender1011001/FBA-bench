import inspect
import json
import os
import re
from typing import Any, Dict, List, Type, Union, Tuple

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
        
        Creates realistic, contextually relevant examples based on tool signatures,
        parameter types, and naming conventions. Includes basic usage, advanced scenarios,
        and edge cases to provide comprehensive documentation.
        """
        examples = {}
        for tool in tool_functions:
            tool_name = tool["name"]
            parameters = tool.get("parameters", [])
            example_list = []
            
            # Generate basic usage example
            basic_example = self._create_basic_usage_example(tool_name, parameters)
            example_list.append(f"Example 1: Basic usage - {basic_example['description']}")
            example_list.append(f"```json\n{json.dumps(basic_example['json'], indent=2)}\n```")
            
            # Generate advanced usage example with optional parameters
            if len(parameters) > 1:  # Only if there are multiple parameters
                advanced_example = self._create_advanced_usage_example(tool_name, parameters)
                example_list.append(f"Example 2: Advanced usage - {advanced_example['description']}")
                example_list.append(f"```json\n{json.dumps(advanced_example['json'], indent=2)}\n```")
            
            # Generate edge case example if applicable
            edge_case_example = self._create_edge_case_example(tool_name, parameters)
            if edge_case_example:
                example_list.append(f"Example 3: Edge case - {edge_case_example['description']}")
                example_list.append(f"```json\n{json.dumps(edge_case_example['json'], indent=2)}\n```")
            
            examples[tool_name] = example_list
        return examples
    
    def _create_basic_usage_example(self, tool_name: str, parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a basic usage example with required parameters."""
        # Extract tool action and object from name
        action, obj = self._parse_tool_name(tool_name)
        
        # Generate basic parameters
        basic_params = {}
        for param in parameters:
            param_name = param["name"]
            param_type = param["type"]
            
            # Skip optional parameters for basic example
            if param["default"] != "N/A":
                continue
                
            basic_params[param_name] = self._generate_sample_value(param_name, param_type, obj)
        
        return {
            "description": f"Simple {action} operation with required parameters",
            "json": {
                "tool_name": tool_name,
                "parameters": basic_params
            }
        }
    
    def _create_advanced_usage_example(self, tool_name: str, parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an advanced usage example with optional parameters."""
        # Extract tool action and object from name
        action, obj = self._parse_tool_name(tool_name)
        
        # Generate advanced parameters including optional ones
        advanced_params = {}
        for param in parameters:
            param_name = param["name"]
            param_type = param["type"]
            
            # Include all parameters with realistic values
            advanced_params[param_name] = self._generate_sample_value(param_name, param_type, obj, advanced=True)
        
        return {
            "description": f"Advanced {action} operation with all parameters",
            "json": {
                "tool_name": tool_name,
                "parameters": advanced_params
            }
        }
    
    def _create_edge_case_example(self, tool_name: str, parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an edge case example if applicable."""
        action, obj = self._parse_tool_name(tool_name)
        
        # Only create edge cases for certain tool types
        if action in ["create", "update"]:
            edge_params = {}
            for param in parameters:
                param_name = param["name"]
                param_type = param["type"]
                
                # Create edge case values
                if "name" in param_name:
                    edge_params[param_name] = f"Special {obj.replace('_', ' ').title()} with Unicode: ñoño"
                elif "price" in param_name or "amount" in param_name:
                    edge_params[param_name] = 999999.99  # High value
                elif "quantity" in param_name or "count" in param_name:
                    edge_params[param_name] = 0  # Edge case value
                elif "date" in param_name or "time" in param_name:
                    edge_params[param_name] = "2024-12-31T23:59:59Z"  # End of year
                elif "id" in param_name:
                    edge_params[param_name] = "special-case-id-123"
                else:
                    edge_params[param_name] = self._generate_sample_value(param_name, param_type, obj)
            
            return {
                "description": f"Edge case handling with special values",
                "json": {
                    "tool_name": tool_name,
                    "parameters": edge_params
                }
            }
        
        return None
    
    def _parse_tool_name(self, tool_name: str) -> Tuple[str, str]:
        """Parse tool name to extract action and object."""
        parts = tool_name.split('_')
        if len(parts) >= 2:
            action = parts[0]
            obj = '_'.join(parts[1:])
            return action, obj
        else:
            return "use", tool_name
    
    def _generate_sample_value(self, param_name: str, param_type: str, obj: str = "", advanced: bool = False) -> Any:
        """Generate a sample value for a parameter based on its name and type."""
        param_name_lower = param_name.lower()
        
        # Handle different parameter types
        if "str" in param_type:
            return self._generate_string_value(param_name_lower, obj, advanced)
        elif "int" in param_type or "float" in param_type or "number" in param_type:
            return self._generate_numeric_value(param_name_lower, obj, advanced)
        elif "bool" in param_type:
            return self._generate_boolean_value(param_name_lower, advanced)
        elif "list" in param_type or "array" in param_type:
            return self._generate_array_value(param_name_lower, obj, advanced)
        elif "dict" in param_type or "object" in param_type:
            return self._generate_object_value(param_name_lower, obj, advanced)
        else:
            return "sample_value"
    
    def _generate_string_value(self, param_name: str, obj: str = "", advanced: bool = False) -> str:
        """Generate a sample string value."""
        if "name" in param_name:
            if obj:
                return f"Sample {obj.replace('_', ' ').title()}"
            return "Sample Name"
        elif "id" in param_name:
            return f"{obj or 'item'}_123"
        elif "email" in param_name:
            return "user@example.com"
        elif "description" in param_name:
            return "A detailed description of the item or operation"
        elif "query" in param_name or "search" in param_name:
            return "search terms or query string"
        elif "type" in param_name:
            return "standard"
        elif "status" in param_name:
            return "active"
        elif "category" in param_name:
            return "general"
        elif "format" in param_name:
            return "json"
        elif "url" in param_name or "link" in param_name:
            return "https://example.com/resource"
        elif "date" in param_name or "time" in param_name:
            return "2024-01-01T00:00:00Z"
        elif "token" in param_name or "key" in param_name:
            return "sample_token_or_key"
        elif "message" in param_name or "note" in param_name:
            return "Additional information or notes"
        else:
            return "sample_value"
    
    def _generate_numeric_value(self, param_name: str, obj: str = "", advanced: bool = False) -> Union[int, float]:
        """Generate a sample numeric value."""
        if "price" in param_name or "cost" in param_name or "amount" in param_name:
            return 99.99 if advanced else 19.99
        elif "quantity" in param_name or "count" in param_name or "number" in param_name:
            return 100 if advanced else 10
        elif "limit" in param_name or "max" in param_name:
            return 50 if advanced else 10
        elif "offset" in param_name or "page" in param_name:
            return 0
        elif "percentage" in param_name or "rate" in param_name:
            return 0.15 if advanced else 0.05
        elif "score" in param_name or "rating" in param_name:
            return 4.5 if advanced else 3.0
        elif "size" in param_name or "length" in param_name:
            return 1000 if advanced else 100
        elif "priority" in param_name:
            return 1
        elif "version" in param_name:
            return 2
        elif "year" in param_name:
            return 2024
        elif "month" in param_name:
            return 1
        elif "day" in param_name:
            return 1
        else:
            return 42
    
    def _generate_boolean_value(self, param_name: str, advanced: bool = False) -> bool:
        """Generate a sample boolean value."""
        if "active" in param_name or "enabled" in param_name or "is_" in param_name:
            return True
        elif "required" in param_name or "mandatory" in param_name:
            return False
        elif "verified" in param_name or "approved" in param_name:
            return True if advanced else False
        else:
            return True
    
    def _generate_array_value(self, param_name: str, obj: str = "", advanced: bool = False) -> List[Any]:
        """Generate a sample array value."""
        if "tag" in param_name:
            return ["important", "featured", "new"] if advanced else ["sample"]
        elif "item" in param_name or "element" in param_name:
            return ["item1", "item2", "item3"] if advanced else ["sample_item"]
        elif "id" in param_name:
            return ["id_001", "id_002", "id_003"] if advanced else ["sample_id"]
        elif "category" in param_name:
            return ["electronics", "books", "clothing"] if advanced else ["general"]
        elif "option" in param_name or "choice" in param_name:
            return ["option_a", "option_b", "option_c"] if advanced else ["default_option"]
        elif "field" in param_name or "column" in param_name:
            return ["name", "description", "price"] if advanced else ["field1"]
        elif "filter" in param_name:
            return [{"field": "status", "value": "active"}, {"field": "date", "operator": ">=", "value": "2024-01-01"}] if advanced else [{"field": "status", "value": "active"}]
        else:
            return ["value1", "value2"] if advanced else ["sample_value"]
    
    def _generate_object_value(self, param_name: str, obj: str = "", advanced: bool = False) -> Dict[str, Any]:
        """Generate a sample object value."""
        if "metadata" in param_name:
            return {
                "created_by": "system",
                "created_at": "2024-01-01T00:00:00Z",
                "version": 1,
                "tags": ["auto-generated"]
            } if advanced else {
                "created_by": "system"
            }
        elif "config" in param_name or "settings" in param_name:
            return {
                "enabled": True,
                "timeout": 30,
                "retry_count": 3,
                "log_level": "info"
            } if advanced else {
                "enabled": True
            }
        elif "filter" in param_name:
            return {
                "field": "status",
                "operator": "=",
                "value": "active"
            }
        elif "sort" in param_name:
            return {
                "field": "created_at",
                "direction": "desc"
            }
        elif "pagination" in param_name:
            return {
                "page": 1,
                "page_size": 20
            }
        elif "data" in param_name:
            return {
                "key1": "value1",
                "key2": "value2"
            }
        else:
            return {
                "key": "value"
            }

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
