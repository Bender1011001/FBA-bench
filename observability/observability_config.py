from dataclasses import dataclass, field

@dataclass
class ObservabilityConfig:
    """
    Configuration settings for observability features in FBA-Bench.
    """
    enable_trace_analysis: bool = True
    llm_friendly_tools: bool = True
    auto_error_correction: bool = True
    real_time_alerts: bool = False
    insight_generation_interval: int = 100 # ticks
    error_pattern_window: int = 50 # events
    performance_alert_threshold: float = 0.8 # e.g., for CPU/memory utilization percentage
    trace_retention_days: int = 30 # days

    # Future fields could include:
    # dashboard_refresh_interval_ms: int = 1000
    # alert_notification_channels: List[str] = field(default_factory=list) # e.g., ["console", "email", "webhook"]
    # data_storage_backend: str = "sqlite"

    def validate(self):
        """Validates the configuration settings."""
        if not (self.insight_generation_interval > 0):
            raise ValueError("insight_generation_interval must be greater than 0.")
        if not (self.error_pattern_window > 0):
            raise ValueError("error_pattern_window must be greater than 0.")
        if not (0.0 <= self.performance_alert_threshold <= 1.0):
            raise ValueError("performance_alert_threshold must be between 0.0 and 1.0.")
        if not (self.trace_retention_days >= 0):
            raise ValueError("trace_retention_days cannot be negative.")

# Example usage (can be removed or placed in a test)
if __name__ == "__main__":
    config = ObservabilityConfig()
    print("Default Observability Config:")
    print(config)

    try:
        invalid_config = ObservabilityConfig(insight_generation_interval=0)
        invalid_config.validate()
    except ValueError as e:
        print(f"\nCaught expected error: {e}")

    try:
        invalid_config_threshold = ObservabilityConfig(performance_alert_threshold=1.5)
        invalid_config_threshold.validate()
    except ValueError as e:
        print(f"Caught expected error: {e}")
</content
<line_count>39</line_count>
</write_to_file>
<thinking>
I've created the `observability_config.py` file. Now I will update the todo list.
</thinking>