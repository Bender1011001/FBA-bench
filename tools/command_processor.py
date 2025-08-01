import json
import logging
import re
from typing import Dict, Any, Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SmartCommandProcessor:
    """
    Intelligently processes agent commands, recognizing intent, auto-correcting,
    scoring confidence, and suggesting fallback mechanisms.
    """

    def process_agent_command(self, raw_command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently processes raw agent commands.
        Attempts to parse it as JSON, then applies auto-correction and intent recognition.
        """
        processed_command = {
            "original_command": raw_command,
            "parsed_command": None,
            "corrected_command": None,
            "intent": None,
            "confidence_score": 0.0,
            "status": "failed",
            "error_message": "Unknown processing error."
        }

        # Try to parse as JSON first
        try:
            parsed_json = json.loads(raw_command)
            processed_command["parsed_command"] = parsed_json
            processed_command["corrected_command"] = parsed_json # If valid, no correction needed initially
            processed_command["status"] = "success"
            processed_command["confidence_score"] = self.calculate_confidence_score(parsed_json)
            processed_command["intent"] = self.recognize_intent(raw_command) # Still recognize intent from raw if meaningful
            processed_command["error_message"] = None
        except json.JSONDecodeError as e:
            processed_command["error_message"] = f"JSON decoding error: {e}. Attempting auto-correction..."
            logging.debug(f"JSON decoding failed: {raw_command} - {e}")

            # Attempt auto-correction if JSON parsing failed
            corrected_command_text = self.auto_correct_command(raw_command)
            try:
                corrected_json = json.loads(corrected_command_text)
                processed_command["corrected_command"] = corrected_json
                processed_command["parsed_command"] = corrected_json # Use corrected as parsed if successful
                processed_command["status"] = "corrected"
                processed_command["confidence_score"] = self.calculate_confidence_score(corrected_json) * 0.8 # Lower confidence for corrected
                processed_command["intent"] = self.recognize_intent(corrected_command_text)
                processed_command["error_message"] = None
            except json.JSONDecodeError as e_corrected:
                processed_command["error_message"] += f" Auto-correction also failed: {e_corrected}. "
                processed_command["status"] = "failed_correction"
                logging.warning(f"Auto-correction failed for command: {raw_command} - {e_corrected}")

        # If still no successful parsing (even after correction), try to recognize intent broadly
        if processed_command["parsed_command"] is None:
            if not processed_command["intent"]: # If intent wasn't recognized from initial JSON attempt
                processed_command["intent"] = self.recognize_intent(raw_command)
            if processed_command["intent"]:
                # If intent recognized but not parsable JSON, it's a "partial_understanding"
                processed_command["status"] = "partial_understanding"
                processed_command["confidence_score"] = max(processed_command["confidence_score"], 0.3) # Give some confidence for intent
                processed_command["error_message"] = "Command could not be fully parsed, but intent was recognized."
            else:
                processed_command["status"] = "failed"
                processed_command["confidence_score"] = 0.0
                processed_command["error_message"] += "Neither JSON parsing nor intent recognition was successful."


        logging.info(f"Command processed. Status: {processed_command['status']}, Confidence: {processed_command['confidence_score']:.2f}")
        return processed_command

    def recognize_intent(self, command_text: str) -> Optional[str]:
        """
        Extracts the intended action from natural language or malformed commands.
        Uses simple regex patterns for demonstration; can be enhanced with an LLM.
        """
        command_text_lower = command_text.lower()

        # Check for tool_name from JSON-like structure
        tool_name_match = re.search(r'"tool_name"\s*:\s*["\']?([a-zA-Z_]+)["\']?', command_text_lower)
        if tool_name_match:
            return tool_name_match.group(1)

        # Keyword-based intent recognition
        if "create product" in command_text_lower or "add new product" in command_text_lower:
            return "create_product"
        if "update stock" in command_text_lower or "adjust inventory" in command_text_lower:
            return "update_stock"
        if "get market data" in command_text_lower or "fetch market data" in command_text_lower:
            return "get_market_data"
        if "launch campaign" in command_text_lower or "start marketing" in command_text_lower:
            return "launch_campaign"
        
        logging.debug(f"Could not recognize intent for command: {command_text}")
        return None

    def auto_correct_command(self, malformed_command: str) -> str:
        """
        Fixes common typos and formatting errors automatically.
        (Simplified: focuses on basic JSON structural fixes)
        """
        corrected_command = malformed_command

        # Common JSON corrections
        # 1. Replace single quotes with double quotes for keys and string values
        corrected_command = re.sub(r"'([^']+)'\s*:", r'"\1":', corrected_command) # keys
        corrected_command = re.sub(r":\s*'([^']+)'", r': "\1"', corrected_command) # string values

        # 2. Add missing commas between key-value pairs or list items (basic)
        # This is very rudimentary and might apply incorrectly in complex cases
        corrected_command = re.sub(r'}\s*"(?=\w+":)', r'},"', corrected_command) # between objects
        corrected_command = re.sub(r']\s*"', r'],"', corrected_command) # after arrays, before key
        corrected_command = re.sub(r'(?<=\w|[\]}])\s*{\s*"', r',{"', corrected_command) # before new object if missing comma

        # 3. Handle unclosed quotes (simple case: add a quote at the end if string looks incomplete)
        # This is a risky correction and might require more advanced parsing for robustness
        if corrected_command.count('"') % 2 != 0:
            if corrected_command.strip().endswith('{') or corrected_command.strip().endswith('['):
                # Don't add if it's an unclosed bracket/brace for object/array parsing
                pass
            else:
                # Attempt to close an open string at the end
                corrected_command += '"}' if corrected_command.endswith('{') else '}' # Assuming it might be trailing JSON
                corrected_command += '"' if not corrected_command.endswith('"') else ''


        # 4. Try to wrap the whole thing in {} if it looks like a list of KV pairs
        if not corrected_command.strip().startswith('{') and not corrected_command.strip().endswith('}'):
            # This is a very rough heuristic
            if 'tool_name' in corrected_command and 'parameters' in corrected_command:
                corrected_command = "{" + corrected_command + "}"

        # 5. Remove trailing commas before closing braces/brackets directly
        corrected_command = re.sub(r',(\s*[\]}])', r'\1', corrected_command)


        # Basic JSON-like structure reconstruction if completely broken but contains key words
        if not corrected_command.strip().startswith("{") and "tool_name" in corrected_command and "parameters" in corrected_command:
            logging.debug(f"Attempting aggressive reconstruction for: {malformed_command}")
            # This part would typically be much more complex, potentially involving a grammar parser
            # For now, it's a very simple wrap-around
            try:
                # A very aggressive attempt to make it valid JSON by wrapping
                temp_wrapped_command = "{" + corrected_command + "}"
                json.loads(temp_wrapped_command) # Test if this makes it valid
                corrected_command = temp_wrapped_command
            except json.JSONDecodeError:
                pass


        logging.debug(f"Auto-corrected '{malformed_command}' to '{corrected_command}'")
        return corrected_command

    def calculate_confidence_score(self, processed_command: Dict[str, Any]) -> float:
        """
        Rates how well the processor understood the command based on parsing success,
        presence of key fields, and whether auto-correction was needed.
        """
        if not processed_command:
            return 0.0

        score = 0.0
        # Base score for successful parsing (direct or corrected)
        if processed_command.get("parsed_command"):
            score += 0.6 # Base for being parsable JSON
            if processed_command.get("corrected_command") and processed_command["corrected_command"] != processed_command["parsed_command"]:
                score -= 0.2 # Penalty if auto-correction was needed

            # Increase score if essential fields are present
            if "tool_name" in processed_command["parsed_command"]:
                score += 0.2
            if "parameters" in processed_command["parsed_command"]:
                score += 0.2
        
        # Add a small score if intent was recognized even if JSON failed
        if processed_command.get("intent") and not processed_command.get("parsed_command"):
            score += 0.3 # Partial understanding

        return max(0.0, min(1.0, score)) # Ensure score is between 0 and 1

    def suggest_fallback_actions(self, failed_command_info: Dict[str, Any]) -> List[str]:
        """
        Provides alternative approaches or simplified suggestions when a primary command fails.
        """
        suggestions = []
        original_command = failed_command_info.get("original_command", "")
        error_message = failed_command_info.get("error_message", "")
        intent = failed_command_info.get("intent")

        suggestions.append("Consider the following alternative actions or ways to phrase your command:")

        if "JSONDecodeError" in error_message or failed_command_info.get("status") == "failed_correction":
            suggestions.append("- Double-check your JSON syntax. Ensure all keys and string values are in double quotes, and commas are correctly placed.")
            suggestions.append("- Provide a simpler command, focusing only on the `tool_name` and essential `parameters`.")
            if intent:
                suggestions.append(f"- Instead of complex JSON, try stating your intent clearly: 'I want to use the {intent} tool.' ")

        elif "Missing 'tool_name'" in error_message:
            suggestions.append("- Ensure your command explicitly specifies `\"tool_name\": \"YourToolName\"`.")
        elif "Missing 'parameters'" in error_message:
            suggestions.append("- Ensure your command includes a `\"parameters\": {{...}}` object, even if empty.")
        
        if intent:
            # General fallback related to the detected intent
            suggestions.append(f"- If targeting the '{intent}' tool, review its specific required parameters.")
            suggestions.append(f"- Try a direct command for '{intent}' without nested logic, e.g., `{{\"tool_name\": \"{intent}\", \"parameters\": {{...}} }}`.")
        
        if not suggestions:
            suggestions.append("- Try rephrasing your command in a very simple, direct manner.")
            suggestions.append("- Consult the tool documentation for exact syntax and required parameters.")
            suggestions.append("- If you were trying a complex operation, break it down into smaller, simpler steps.")

        return suggestions