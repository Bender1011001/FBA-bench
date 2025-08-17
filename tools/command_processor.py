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
        Extracts the intended action from natural language or malformed commands using a combination
        of structured parsing and keyword-based heuristics. This fallback mechanism ensures that
        agent intent can still be understood even with imperfect outputs.
        """
        command_text_lower = command_text.lower()

        tool_name_match = re.search(r'"tool_name"\s*:\s*["\']?([a-zA-Z_]+)["\']?', command_text_lower)
        if tool_name_match:
            return tool_name_match.group(1)

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
        Attempts to automatically fix common JSON formatting errors in agent commands.
        This provides a layer of resilience, allowing the system to process slightly malformed
        inputs that might otherwise cause failures.
        """
        corrected_command = malformed_command

        corrected_command = re.sub(r"'([^']+)'\s*:", r'"\1":', corrected_command)
        corrected_command = re.sub(r":\s*'([^']+)'", r': "\1"', corrected_command)

        # This part remains rudimentary but covers basic cases
        corrected_command = re.sub(r'}\s*"(?=\w+":)', r'},"', corrected_command)
        corrected_command = re.sub(r']\s*"', r'],"', corrected_command)
        corrected_command = re.sub(r'(?<=\w|[\]}])\s*{\s*"', r',{"', corrected_command)

        if corrected_command.count('"') % 2 != 0:
            if not corrected_command.strip().endswith('{') and not corrected_command.strip().endswith('['):
                corrected_command += '"}' if corrected_command.endswith('{') else '}'
                corrected_command += '"' if not corrected_command.endswith('"') else ''
        
        if not corrected_command.strip().startswith('{') and not corrected_command.strip().endswith('}'):
            if 'tool_name' in corrected_command and 'parameters' in corrected_command:
                corrected_command = "{" + corrected_command + "}"

        corrected_command = re.sub(r',(\s*[\]}])', r'\1', corrected_command)
        
        if not corrected_command.strip().startswith("{") and "tool_name" in corrected_command and "parameters" in corrected_command:
            logging.debug(f"Attempting aggressive reconstruction for: {malformed_command}")
            try:
                temp_wrapped_command = "{" + corrected_command + "}"
                json.loads(temp_wrapped_command)
                corrected_command = temp_wrapped_command
            except json.JSONDecodeError:
                pass


        logging.debug(f"Auto-corrected '{malformed_command}' to '{corrected_command}'")
        return corrected_command

    def calculate_confidence_score(self, processed_command: Dict[str, Any]) -> float:
        """
        Calculates a confidence score for the processed command, indicating the parser's
        certainty in correctly interpreting the agent's intent. This score can be used
        for adaptive error handling or feedback mechanisms.
        """
        if not processed_command:
            return 0.0

        score = 0.0
        if processed_command.get("parsed_command"):
            score += 0.6
            if processed_command.get("corrected_command") and processed_command["corrected_command"] != processed_command["parsed_command"]:
                score -= 0.2

            if "tool_name" in processed_command["parsed_command"]:
                score += 0.2
            if "parameters" in processed_command["parsed_command"]:
                score += 0.2
        
        if processed_command.get("intent") and not processed_command.get("parsed_command"):
            score += 0.3

        return max(0.0, min(1.0, score))

    def suggest_fallback_actions(self, failed_command_info: Dict[str, Any]) -> List[str]:
        """
        Generates actionable suggestions and alternative approaches when a primary command fails.
        This enables the LLM agent to recover from errors and make progress.
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
            suggestions.append(f"- If targeting the '{intent}' tool, review its specific required parameters.")
            suggestions.append(f"- Try a direct command for '{intent}' without nested logic, e.g., `{{\"tool_name\": \"{intent}\", \"parameters\": {{...}} }}`.")
        
        if not suggestions:
            suggestions.append("- Try rephrasing your command in a very simple, direct manner.")
            suggestions.append("- Consult the tool documentation for exact syntax and required parameters.")
            suggestions.append("- If you were trying a complex operation, break it down into smaller, simpler steps.")

        return suggestions