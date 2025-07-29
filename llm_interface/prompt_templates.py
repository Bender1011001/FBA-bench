from typing import Dict, Any, List

class PromptTemplates:
    """
    Manages different prompt templates for various simulation scenarios.
    """

    @staticmethod
    def initial_setup_template(context: Dict[str, Any]) -> str:
        """
        Template for the very first prompt to the LLM agent, providing full initial context.
        """
        template = f"""
You are an autonomous FBA (Fulfillment by Amazon) seller agent. Your goal is to manage your product portfolio to maximize profit over time within a dynamic market simulation.

You will receive simulation state updates and current events, and you must respond with actions in a strict JSON format.

=== FBA SIMULATION STATE ===
Tick: {context['current_tick']} | Day: {context['simulation_time'].day} | Time: {context['simulation_time'].strftime('%H:%M UTC')}

{context['budget_status']}

PRODUCT PORTFOLIO:
{context['product_portfolio']}

RECENT EVENTS:
{context['recent_events']}

AVAILABLE ACTIONS:
{context['available_actions']}

REQUIRED OUTPUT FORMAT:
{context['required_output_format']}

---
Your response MUST be a single JSON object conforming strictly to the 'REQUIRED OUTPUT FORMAT'. Do NOT include any other text or formatting outside the JSON.
Analyze the initial market conditions and your product portfolio. Formulate a strategy and decide on your first action.
"""
        return template.strip()

    @staticmethod
    def regular_update_template(context: Dict[str, Any]) -> str:
        """
        Template for regular simulation updates, focusing on incremental changes and recent events.
        """
        template = f"""
You are an autonomous FBA seller agent. Your goal is to manage your product portfolio to maximize profit over time within a dynamic market simulation.

=== FBA SIMULATION STATE ===
Tick: {context['current_tick']} | Day: {context['simulation_time'].day} | Time: {context['simulation_time'].strftime('%H:%M UTC')}

{context['budget_status']}

PRODUCT PORTFOLIO:
{context['product_portfolio']}

RECENT EVENTS:
{context['recent_events']}

AVAILABLE ACTIONS:
{context['available_actions']}

REQUIRED OUTPUT FORMAT:
{context['required_output_format']}

---
Your response MUST be a single JSON object conforming strictly to the 'REQUIRED OUTPUT FORMAT'. Do NOT include any other text or formatting outside the JSON.
Analyze the latest market data and recent events. Adjust your strategy and decide on your next action to optimize profit.
"""
        return template.strip()

    @staticmethod
    def shock_event_template(context: Dict[str, Any], shock_description: str) -> str:
        """
        Template for crisis response scenarios (e.g., sudden market changes, competitor actions).
        """
        template = f"""
You are an autonomous FBA seller agent. Your goal is to manage your product portfolio to maximize profit over time within a dynamic market simulation.

=== FBA SIMULATION STATE ===
Tick: {context['current_tick']} | Day: {context['simulation_time'].day} | Time: {context['simulation_time'].strftime('%H:%M UTC')}

{context['budget_status']}

PRODUCT PORTFOLIO:
{context['product_portfolio']}

RECENT EVENTS:
{context['recent_events']}

---
EMERGENCY ALERT: A significant market shock has occurred!
Description: {shock_description}
---

AVAILABLE ACTIONS:
{context['available_actions']}

REQUIRED OUTPUT FORMAT:
{context['required_output_format']}

---
Your response MUST be a single JSON object conforming strictly to the 'REQUIRED OUTPUT FORMAT'. Do NOT include any other text or formatting outside the JSON.
Critically evaluate the {shock_description} and its immediate impact. Respond decisively to mitigate negative effects or capitalize on new opportunities. Prioritize short-term stability and long-term recovery.
"""
        return template.strip()

    @staticmethod
    def budget_warning_template(context: Dict[str, Any], warning_message: str) -> str:
        """
        Template for notifying the LLM about budget constraints.
        """
        template = f"""
You are an autonomous FBA seller agent. Your goal is to manage your product portfolio to maximize profit over time within a dynamic market simulation.

=== FBA SIMULATION STATE ===
Tick: {context['current_tick']} | Day: {context['simulation_time'].day} | Time: {context['simulation_time'].strftime('%H:%M UTC')}

---
BUDGET WARNING: Your token usage is nearing its limit!
Message: {warning_message}
{context['budget_status']}
---

PRODUCT PORTFOLIO:
{context['product_portfolio']}

RECENT EVENTS:
{context['recent_events']}

AVAILABLE ACTIONS:
{context['available_actions']}

REQUIRED OUTPUT FORMAT:
{context['required_output_format']}

---
Your response MUST be a single JSON object conforming strictly to the 'REQUIRED OUTPUT FORMAT'. Do NOT include any other text or formatting outside the JSON.
Consider the budget warning and optimize your actions to be more token-efficient. Focus on essential decisions.
"""
        return template.strip()

    @staticmethod
    def get_template(template_name: str, context: Dict[str, Any], **kwargs) -> str:
        """
        Fetches and renders a specified prompt template.
        """
        if template_name == "initial_setup":
            return PromptTemplates.initial_setup_template(context)
        elif template_name == "regular_update":
            return PromptTemplates.regular_update_template(context)
        elif template_name == "shock_event":
            return PromptTemplates.shock_event_template(context, kwargs.get("shock_description", "An unknown market event."))
        elif template_name == "budget_warning":
            return PromptTemplates.budget_warning_template(context, kwargs.get("warning_message", "Token usage is high."))
        else:
            raise ValueError(f"Unknown prompt template name: {template_name}")
