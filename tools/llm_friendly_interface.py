import json
import re
import math
from datetime import datetime
from typing import Any, Dict, List, Tuple, Set

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
        Implements data summarization and formatting based on the specified complexity level to ensure
        LLMs receive optimal information density without being overwhelmed.
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
        """Create a concise overview focusing on primary metrics and high-level status."""
        summary = {
            "overview": "Key metrics and summary",
            "data_type": self._identify_data_type(data),
            "total_items": self._count_items(data),
            "key_metrics": self._extract_key_metrics(data)
        }
        
        date_range = self._extract_date_range(data)
        if date_range:
            summary["time_period"] = date_range
        
        return summary
    
    def _create_medium_complexity_summary(self, data: dict) -> dict:
        """Create a detailed summary highlighting important fields, structure, and basic statistics."""
        return {
            "summary": "Detailed summary with key fields",
            "structure": self._analyze_structure(data),
            "important_fields": self._extract_important_fields(data),
            "sample_data": self._extract_sample_data(data, max_samples=3),
            "statistics": self._calculate_basic_statistics(data)
        }
    
    def _create_high_complexity_summary(self, data: dict) -> dict:
        """Create a comprehensive view including full data, detailed metadata, and advanced statistics."""
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
        """Dynamically identifies the primary type of data structure based on common keys."""
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
        """Recursively counts items within list or dict structures."""
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
        """Extracts significant numerical metrics from the data for high-level summaries."""
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
        """Identifies and extracts date range information from string fields within the data."""
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
        """Performs a structural analysis of the data, identifying key patterns and nesting levels."""
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
        """Extracts fields considered important based on a predefined list of naming conventions."""
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
        """Extracts a limited number of sample items from lists within the data structure."""
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
        """Calculates basic descriptive statistics (count, sum, average, min, max) for numerical fields."""
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
        """Assess data quality: completeness, consistency, validity with concrete heuristics and issues list."""
        issues: List[str] = []

        # 1) Completeness: missing/empty values ratio across entire structure
        total_fields = 0
        missing_fields = 0

        def walk_and_count(obj: Any) -> None:
            nonlocal total_fields, missing_fields
            if isinstance(obj, dict):
                for _, v in obj.items():
                    total_fields += 1
                    if v is None or (isinstance(v, str) and v.strip() == ""):
                        missing_fields += 1
                    walk_and_count(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk_and_count(item)

        walk_and_count(data)
        completeness = (total_fields - missing_fields) / total_fields if total_fields > 0 else 1.0
        if missing_fields > 0:
            issues.append(f"Missing/empty fields detected: {missing_fields} of {total_fields} scanned")

        # 2) Consistency: for lists of dicts, check that repeated keys maintain stable types
        # Collect per-key observed types under list-of-dicts collections
        key_types: Dict[str, Set[type]] = {}

        def collect_key_types(obj: Any) -> None:
            if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
                for row in obj:
                    for k, v in row.items():
                        t = type(v)
                        key_types.setdefault(k, set()).add(t)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect_key_types(v)
            elif isinstance(obj, list):
                for v in obj:
                    collect_key_types(v)

        collect_key_types(data)
        inconsistent_keys = [k for k, ts in key_types.items() if len(ts) > 1]
        consistency = 1.0 if not inconsistent_keys else max(0.0, 1.0 - (len(inconsistent_keys) / max(1, len(key_types))))
        if inconsistent_keys:
            issues.append(f"Inconsistent types for keys: {', '.join(sorted(inconsistent_keys))}")

        # 3) Validity: heuristics
        # - Non-negative for fields like price, amount, total, count, quantity, units, revenue
        # - Parsable ISO dates for *_at, date, timestamp keys
        # - Numeric sanity (no NaN/inf)
        invalid_numeric: List[str] = []
        negative_fields: List[str] = []
        bad_dates: List[str] = []

        numeric_key_patterns = re.compile(r"(price|amount|total|count|quantity|units|revenue|cost|profit|value)$", re.IGNORECASE)
        date_key_patterns = re.compile(r"(created_at|updated_at|date|timestamp)$", re.IGNORECASE)

        def validate(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    p = f"{path}.{k}" if path else k
                    # Numeric checks
                    if isinstance(v, (int, float)):
                        if math.isnan(v) or math.isinf(v):  # type: ignore[arg-type]
                            invalid_numeric.append(p)
                        if numeric_key_patterns.search(k) and v < 0:
                            negative_fields.append(p)
                    # Date checks
                    if isinstance(v, str) and date_key_patterns.search(k):
                        # Accept ISO 8601 basic heuristic
                        try:
                            # Flexible parse: allow date or datetime
                            if "T" in v:
                                datetime.fromisoformat(v.replace("Z", "+00:00"))
                            else:
                                datetime.fromisoformat(v)
                        except Exception:
                            bad_dates.append(p)
                    validate(v, p)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    validate(item, f"{path}[{i}]")

        validate(data)

        if invalid_numeric:
            issues.append(f"Invalid numeric values (NaN/inf) at: {', '.join(invalid_numeric[:10])}{'…' if len(invalid_numeric) > 10 else ''}")
        if negative_fields:
            issues.append(f"Negative values in non-negative fields at: {', '.join(negative_fields[:10])}{'…' if len(negative_fields) > 10 else ''}")
        if bad_dates:
            issues.append(f"Unparseable date/timestamp fields at: {', '.join(bad_dates[:10])}{'…' if len(bad_dates) > 10 else ''}")

        # Validity score: penalize per category proportionally
        problem_count = len(invalid_numeric) + len(negative_fields) + len(bad_dates)
        # Scale by total fields scanned to avoid over-penalizing small samples
        validity = 1.0 if total_fields == 0 else max(0.0, 1.0 - min(1.0, problem_count / max(1, total_fields)))

        return {
            "completeness": round(completeness, 6),
            "consistency": round(consistency, 6),
            "validity": round(validity, 6),
            "issues": issues,
        }
    
    def _identify_relationships(self, data: dict) -> dict:
        """Identify references, hierarchies, and dependencies via structural and key-based heuristics."""
        references: List[str] = []
        hierarchies: List[str] = []
        dependencies: List[Dict[str, str]] = []

        # Collect object IDs and their paths; and collect all fields ending with *_id
        object_ids: Dict[str, List[str]] = {}  # id_value -> paths
        foreign_keys: List[Tuple[str, str]] = []  # (path, key_name)

        def scan(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                # Hierarchy recording
                hierarchies.append(path or "<root>")
                # Record id-like fields and their values
                for k, v in obj.items():
                    p = f"{path}.{k}" if path else k
                    if isinstance(v, (str, int)) and re.search(r"(?:^|_)id$", k, re.IGNORECASE):
                        object_ids.setdefault(str(v), []).append(p)
                    if isinstance(v, (str, int)) and re.search(r"_id$", k, re.IGNORECASE) and not re.fullmatch(r"(?:^|_)id$", k, re.IGNORECASE):
                        foreign_keys.append((p, k))
                    scan(v, p)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    scan(item, f"{path}[{i}]")

        scan(data)

        # References: list all paths that define IDs
        for id_value, paths in object_ids.items():
            for p in paths:
                references.append(p)

        # Dependencies: if a *_id field's value matches some object's id value, record dependency
        def value_at_path(obj: Any, path: str) -> Any:
            # Best effort: walk dict/list by tokens; only used for small lookups
            tokens = re.split(r"\.(?![^\[]*\])", path)
            cur = obj
            for tok in tokens:
                if tok == "":
                    continue
                m = re.match(r"(.+)\[(\d+)\]$", tok)
                if m:
                    key, idx = m.group(1), int(m.group(2))
                    if isinstance(cur, dict):
                        cur = cur.get(key)
                    else:
                        return None
                    if isinstance(cur, list) and 0 <= idx < len(cur):
                        cur = cur[idx]
                    else:
                        return None
                else:
                    if isinstance(cur, dict):
                        cur = cur.get(tok)
                    else:
                        return None
            return cur

        for fk_path, fk_key in foreign_keys:
            fk_val = value_at_path(data, fk_path)
            if fk_val is None:
                continue
            hit_paths = object_ids.get(str(fk_val), [])
            for target_path in hit_paths:
                if target_path == fk_path:
                    continue
                dependencies.append({
                    "from": fk_path,
                    "to": target_path,
                    "via": fk_key
                })

        # Deduplicate simple lists
        references = sorted(set(references))
        hierarchies = sorted(set(hierarchies))

        return {
            "references": references,
            "hierarchies": hierarchies,
            "dependencies": dependencies
        }
    
    def _calculate_comprehensive_statistics(self, data: dict) -> dict:
        """Calculates comprehensive statistics including distribution metrics like median and percentiles."""
        stats = self._calculate_basic_statistics(data)
        
        # Add distribution information where applicable
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
            
            # Re-collect numbers for this path only to ensure correct sorting/median calculation
            collected_numbers_for_path = []
            def re_collect_path_numbers(obj, current_path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        re_collect_path_numbers(value, f"{current_path}.{key}" if current_path else key)
                elif isinstance(obj, list):
                    for item in obj:
                        re_collect_path_numbers(item, current_path)
                elif isinstance(obj, (int, float)) and current_path == path:
                    collected_numbers_for_path.append(obj)

            re_collect_path_numbers(data)
            
            if len(collected_numbers_for_path) > 1:
                sorted_numbers = sorted(collected_numbers_for_path)
                n = len(sorted_numbers)
                stat_info["median"] = sorted_numbers[n // 2]
                stat_info["percentile_25"] = sorted_numbers[n // 4]
                stat_info["percentile_75"] = sorted_numbers[3 * n // 4]
        
        return stats

    def inject_contextual_help(self, tool_description: str) -> str:
        """
        Enhances a tool description with LLM-specific guidance on expected data formats and best practices.
        """
        return (f"{tool_description}\n\n"
                "IMPORTANT: When using this tool, please ensure your parameters are valid JSON "
                "and adhere to the specific types and formats required. "
                "Consider the numerical precision for financial values and string lengths for names. Ensure all API keys are correctly configured in the environment.")

    def _generate_human_readable_summary(self, json_data: dict, tool_name: str) -> str:
        """
        Generates a concise human-readable summary of a tool's JSON response,
        aiding LLM interpretation without processing the full data payload.
        """
        if tool_name == "create_product":
            if json_data.get("success"):
                product_name = json_data.get("product_name", "a product")
                product_id = json_data.get("product_id", "N/A")
                return f"Successfully created product '{product_name}' with ID '{product_id}'."
            else:
                error = json_data.get("error", "unknown error")
                return f"Failed to create product: {error}. Check required parameters and data types."
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
                return f"Failed to update stock: {error}. Verify product ID and stock quantity."
        elif "success" in json_data:
            status = "succeeded" if json_data["success"] else "failed"
            message = json_data.get("message", "operation completed.")
            return f"The operation for tool '{tool_name}' {status}. Message: {message}"
        return f"Response for '{tool_name}': {json.dumps(json_data)}"

    def _get_python_type(self, schema_type: str):
        """Maps JSON schema types to Python types for runtime validation."""
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
            return type(None) # Unknown type, matches None
