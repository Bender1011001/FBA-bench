# my_simple_agent_plugin/my_simple_agent_plugin.py

from fba_bench.plugins.agent_plugins.base_agent_plugin import BaseAgentPlugin
from .my_simple_skill import MySimpleSkill # Import the custom skill implemented in this plugin

class MySimpleAgentPlugin(BaseAgentPlugin):
    """
    An example FBA-Bench Agent Plugin that registers a new custom skill.
    """
    PLUGIN_NAME = "MySimpleAgentPlugin"
    VERSION = "1.0.0"
    DESCRIPTION = "This plugin adds 'MySimpleSkill' to FBA-Bench agents."

    def register(self, core_system):
        """
        Registers components provided by this plugin with the core FBA-Bench system.
        This method is called by the Plugin Framework during startup.
        """
        self.core_system = core_system
        self.logger.info(f"[{self.PLUGIN_NAME}] Registering MySimpleSkill...")
        
        # Example: Register a custom skill class with the skill system
        # Assuming core_system has a method to register skill classes (e.g., in SkillFactory)
        # In a real scenario, this might involve a more formal registration API.
        try:
            # Assuming core_system is an object that can register components.
            # This is a simplified representation of how a plugin might interact.
            if hasattr(core_system, 'register_skill_class'):
                core_system.register_skill_class(MySimpleSkill.skill_name, MySimpleSkill)
                self.logger.info(f"[{self.PLUGIN_NAME}] MySimpleSkill registered successfully.")
            else:
                self.logger.warning(f"[{self.PLUGIN_NAME}] Could not find 'register_skill_class' method on core_system. Skill may not be registered.")
        except Exception as e:
            self.logger.error(f"[{self.PLUGIN_NAME}] Error registering skill: {e}")

    def activate(self):
        """
        Performs any setup or startup tasks when the plugin is activated.
        """
        self.logger.info(f"[{self.PLUGIN_NAME}] Activated.")

    def deactivate(self):
        """
        Performs any cleanup tasks when the plugin is deactivated or FBA-Bench shuts down.
        """
        self.logger.info(f"[{self.PLUGIN_NAME}] Deactivated.")

# Note: The presence of this file within a discovered plugin path
# and its inheritance from BaseAgentPlugin will allow the PluginFramework
# to automatically discover and load it.