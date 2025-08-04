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
        
        Generates realistic, contextually relevant examples that demonstrate proper usage
        of the tool with various parameter combinations and edge cases.
        """
        # Dynamic example generation based on tool name patterns
        examples = []
        
        # Extract tool action and object from name
        parts = tool_name.split('_')
        if len(parts) >= 2:
            action = parts[0]
            obj = '_'.join(parts[1:])
            
            # Generate basic example
            basic_example = self._create_basic_example(action, obj)
            examples.append(f"Example 1: Basic {action} operation:\n{basic_example}")
            
            # Generate advanced example with additional parameters
            advanced_example = self._create_advanced_example(action, obj)
            examples.append(f"Example 2: Advanced {action} with additional parameters:\n{advanced_example}")
            
            # Generate context-specific example
            context_example = self._create_contextual_example(action, obj)
            examples.append(f"Example 3: Context-specific usage:\n{context_example}")
        else:
            # Fallback for tools that don't follow the action_object pattern
            examples.append(f"Example: Basic usage of {tool_name}:\n```json\n{{\"tool_name\": \"{tool_name}\", \"parameters\": {{}}}}\n```")
        
        return "\n\n".join(examples)
    
    def _create_basic_example(self, action: str, obj: str) -> str:
        """Create a basic example for the given action and object."""
        params = {}
        
        if action == "create":
            params = {
                "name": f"New {obj.replace('_', ' ').title()}",
                "description": f"Description for {obj.replace('_', ' ')}"
            }
        elif action == "get" or action == "fetch":
            params = {
                "id": f"{obj}_123",
                "include_details": True
            }
        elif action == "update":
            params = {
                "id": f"{obj}_123",
                "updates": {"status": "active"}
            }
        elif action == "delete":
            params = {
                "id": f"{obj}_123",
                "confirm": True
            }
        
        return self._format_example(f"{action}_{obj}", params)
    
    def _create_advanced_example(self, action: str, obj: str) -> str:
        """Create an advanced example with more parameters."""
        params = self._create_basic_example(action, obj)
        
        # Add advanced parameters
        if action == "create":
            params.update({
                "metadata": {"created_by": "system", "priority": "high"},
                "tags": ["important", "new"],
                "settings": {"auto_approve": False, "notify_stakeholders": True}
            })
        elif action == "get" or action == "fetch":
            params.update({
                "filters": {"date_range": {"start": "2024-01-01", "end": "2024-12-31"}},
                "sort_by": "created_date",
                "sort_order": "desc",
                "limit": 50
            })
        
        return self._format_example(f"{action}_{obj}", params)
    
    def _create_contextual_example(self, action: str, obj: str) -> str:
        """Create a context-specific example based on the object type."""
        params = {}
        
        # Add context-specific parameters based on object type
        if "product" in obj:
            params.update({
                "name": "Premium Wireless Headphones",
                "category": "Electronics",
                "price": 199.99,
                "description": "High-quality wireless headphones with noise cancellation"
            })
        elif "market" in obj or "data" in obj:
            params.update({
                "timeframe": "last_30_days",
                "metrics": ["revenue", "units_sold", "customer_satisfaction"],
                "filters": {"region": "North America", "product_category": "Electronics"}
            })
        elif "campaign" in obj:
            params.update({
                "name": "Summer Sale 2024",
                "budget": 50000,
                "target_audience": "tech_enthusiasts",
                "channels": ["social_media", "email", "search_ads"],
                "duration_days": 30
            })
        
        return self._format_example(f"{action}_{obj}", params)
    
    def _format_example(self, tool_name: str, parameters: dict) -> str:
        """Format parameters as a JSON example."""
        import json
        example_json = {
            "tool_name": tool_name,
            "parameters": parameters
        }
        return f"```json\n{json.dumps(example_json, indent=2)}\n```"

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
        
        Implements data summarization and formatting based on the specified complexity level.
        """
        if not isinstance(data, dict):
            return {"formatted_data": str(data)}
        
        if complexity_level == "low":
            return self._create_low_complexity_summary(data)
        elif complexity_level == "medium":
            return self._create_medium_complexity_summary(data)
        elif complexity_level == "high":
            return self._create_high_complexity_summary(data)
        else:
            return data
    
    def _create_low_complexity_summary(self, data: dict) -> dict:
        """Create a simplified overview focusing on key metrics."""
        summary = {
            "overview": "Key metrics and summary",
            "data_type": self._identify_data_type(data),
            "total_items": self._count_items(data),
            "key_metrics": self._extract_key_metrics(data)
        }
        
        # Add temporal summary if date fields are present
        date_range = self._extract_date_range(data)
        if date_range:
            summary["time_period"] = date_range
        
        return summary
    
    def _create_medium_complexity_summary(self, data: dict) -> dict:
        """Create a detailed summary with important fields highlighted."""
        return {
            "summary": "Detailed summary with key fields",
            "structure": self._analyze_structure(data),
            "important_fields": self._extract_important_fields(data),
            "sample_data": self._extract_sample_data(data, max_samples=3),
            "statistics": self._calculate_basic_statistics(data)
        }
    
    def _create_high_complexity_summary(self, data: dict) -> dict:
        """Create a comprehensive view with full data and metadata."""
        return {
            "full_data": data,
            "metadata": {
                "structure_analysis": self._analyze_structure(data),
                "data_quality": self._assess_data_quality(data),
                "relationships": self._identify_relationships(data),
                "statistics": self._calculate_comprehensive_statistics(data)
            }
        }
    
    def _identify_data_type(self, data: dict) -> str:
        """Identify the type of data structure."""
        if "products" in data or "items" in data:
            return "product_catalog"
        elif "sales" in data or "revenue" in data:
            return "sales_data"
        elif "users" in data or "customers" in data:
            return "user_data"
        elif "metrics" in data or "statistics" in data:
            return "analytics_data"
        else:
            return "general_data"
    
    def _count_items(self, data: dict) -> int:
        """Count the total number of items in the data structure."""
        count = 0
        for key, value in data.items():
            if isinstance(value, list):
                count += len(value)
            elif isinstance(value, dict):
                count += self._count_items(value)
            else:
                count += 1
        return count
    
    def _extract_key_metrics(self, data: dict) -> dict:
        """Extract key numerical metrics from the data."""
        metrics = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
            elif isinstance(value, list) and value:
                if all(isinstance(item, (int, float)) for item in value):
                    metrics[f"{key}_avg"] = sum(value) / len(value)
                    metrics[f"{key}_sum"] = sum(value)
                    metrics[f"{key}_count"] = len(value)
        
        return metrics
    
    def _extract_date_range(self, data: dict) -> dict:
        """Extract date range information if available."""
        import re
        dates = []
        
        def extract_dates(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    extract_dates(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_dates(item)
            elif isinstance(obj, str) and re.match(r'\d{4}-\d{2}-\d{2}', obj):
                dates.append(obj)
        
        extract_dates(data)
        
        if dates:
            return {"earliest": min(dates), "latest": max(dates)}
        return None
    
    def _analyze_structure(self, data: dict) -> dict:
        """Analyze the structure of the data."""
        structure = {
            "top_level_keys": list(data.keys()),
            "nested_levels": 0,
            "array_fields": [],
            "object_fields": []
        }
        
        def analyze(obj, level=0):
            if isinstance(obj, dict):
                structure["nested_levels"] = max(structure["nested_levels"], level)
                for value in obj.values():
                    analyze(value, level + 1)
            elif isinstance(obj, list):
                structure["array_fields"].append(f"level_{level}")
                if obj:
                    analyze(obj[0], level + 1)
            else:
                structure["object_fields"].append(f"level_{level}")
        
        analyze(data)
        return structure
    
    def _extract_important_fields(self, data: dict) -> dict:
        """Extract fields that are likely important based on naming conventions."""
        important_fields = {}
        important_patterns = [
            r'id', r'name', r'title', r'description', r'status', r'created_at',
            r'updated_at', r'price', r'amount', r'total', r'count', r'value'
        ]
        
        import re
        for key, value in data.items():
            if any(re.search(pattern, key, re.IGNORECASE) for pattern in important_patterns):
                important_fields[key] = value
        
        return important_fields
    
    def _extract_sample_data(self, data: dict, max_samples: int = 3) -> dict:
        """Extract sample data from arrays in the structure."""
        samples = {}
        
        for key, value in data.items():
            if isinstance(value, list) and value:
                samples[key] = value[:max_samples]
            elif isinstance(value, dict):
                nested_samples = self._extract_sample_data(value, max_samples)
                if nested_samples:
                    samples[key] = nested_samples
        
        return samples
    
    def _calculate_basic_statistics(self, data: dict) -> dict:
        """Calculate basic statistics for numerical fields."""
        stats = {}
        
        def collect_numbers(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    collect_numbers(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    collect_numbers(item, f"{path}[{i}]")
            elif isinstance(obj, (int, float)):
                if path not in stats:
                    stats[path] = []
                stats[path].append(obj)
        
        collect_numbers(data)
        
        result = {}
        for path, numbers in stats.items():
            result[path] = {
                "count": len(numbers),
                "sum": sum(numbers),
                "average": sum(numbers) / len(numbers),
                "min": min(numbers),
                "max": max(numbers)
            }
        
        return result
    
    def _assess_data_quality(self, data: dict) -> dict:
        """Assess the quality of the data."""
        quality = {
            "completeness": 1.0,
            "consistency": 1.0,
            "validity": 1.0,
            "issues": []
        }
        
        # Check for missing values
        total_fields = 0
        missing_fields = 0
        
        def check_missing(obj):
            nonlocal total_fields, missing_fields
            if isinstance(obj, dict):
                for key, value in obj.items():
                    total_fields += 1
                    if value is None or value == "":
                        missing_fields += 1
                    check_missing(value)
            elif isinstance(obj, list):
                for item in obj:
                    check_missing(item)
        
        check_missing(data)
        quality["completeness"] = (total_fields - missing_fields) / total_fields if total_fields > 0 else 1.0
        
        if missing_fields > 0:
            quality["issues"].append(f"Found {missing_fields} missing or empty fields")
        
        return quality
    
    def _identify_relationships(self, data: dict) -> dict:
        """Identify relationships between different parts of the data."""
        relationships = {
            "references": [],
            "hierarchies": [],
            "dependencies": []
        }
        
        # Look for ID references
        id_fields = []
        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if "id" in sub_key.lower() and isinstance(sub_value, str):
                        id_fields.append(f"{key}.{sub_key}")
        
        if id_fields:
            relationships["references"] = id_fields
        
        return relationships
    
    def _calculate_comprehensive_statistics(self, data: dict) -> dict:
        """Calculate comprehensive statistics for the data."""
        stats = self._calculate_basic_statistics(data)
        
        # Add distribution information
        for path, stat_info in stats.items():
            numbers = []
            
            def extract_numbers(obj):
                if isinstance(obj, dict):
                    for value in obj.values():
                        extract_numbers(value)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_numbers(item)
                elif isinstance(obj, (int, float)):
                    numbers.append(obj)
            
            extract_numbers(data)
            
            if len(numbers) > 1:
                sorted_numbers = sorted(numbers)
                n = len(sorted_numbers)
                stat_info["median"] = sorted_numbers[n // 2]
                stat_info["percentile_25"] = sorted_numbers[n // 4]
                stat_info["percentile_75"] = sorted_numbers[3 * n // 4]
        
        return stats

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
