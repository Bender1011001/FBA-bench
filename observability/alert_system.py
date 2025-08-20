import logging
import time
import json
from collections import deque
from typing import Callable, Dict, Any, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ObservabilityAlertSystem:
    """
    Manages real-time observability alerts based on anomaly detection,
    performance monitoring, and error rate tracking.
    """

    def __init__(self, notification_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None):
        """
        Initializes the alert system.
        Args:
            notification_callback: A function to call when an alert is triggered.
                                   Expected signature: (alert_type: str, severity: str, details: Dict[str, Any])
        """
        self.notification_callback = notification_callback
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.metric_baselines: Dict[str, Any] = {}
        self.error_history: Dict[str, deque] = {} # {agent_id: deque of error timestamps}
        logging.info("ObservabilityAlertSystem initialized.")

    def detect_anomalies(self, metric_stream: Dict[str, float], baseline_key: str) -> Optional[Dict[str, Any]]:
        """
        Identifies unusual patterns in agent behavior or system metrics compared to a baseline.
        This is a simplified anomaly detection mechanism.

        Args:
            metric_stream: Current metric values (e.g., {"cpu_usage": 0.7, "memory_usage": 0.6}).
            baseline_key: Key to fetch the stored baseline (e.g., "agent_performance_baseline").

        Returns:
            A dictionary with anomaly details if detected, otherwise None.
        """
        baseline = self.metric_baselines.get(baseline_key)
        if not baseline:
            logging.warning(f"No baseline found for '{baseline_key}'. Cannot perform anomaly detection.")
            return None

        anomalies = {}
        for metric, current_value in metric_stream.items():
            if metric in baseline and baseline[metric] != 0:
                deviation = abs((current_value - baseline[metric]) / baseline[metric])
                # Simple rule: deviation > 20% from baseline is an anomaly
                if deviation > 0.2:
                    anomalies[metric] = {
                        "current_value": current_value,
                        "baseline_value": baseline[metric],
                        "deviation_percent": f"{deviation:.2%}"
                    }
        
        if anomalies:
            alert_details = {
                "message": f"Anomaly detected for baseline '{baseline_key}'.",
                "anomalies": anomalies,
                "timestamp": time.time()
            }
            self.send_alert("anomaly_detection", "High", alert_details)
            return alert_details
        
        return None

    def monitor_performance_metrics(self, current_metrics: Dict[str, float], thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Notifies about system performance degradation based on predefined thresholds.

        Args:
            current_metrics: Dictionary of current performance metrics (e.g., {"latency_ms": 150.5}).
            thresholds: Dictionary of {metric_name: max_allowed_value} thresholds.

        Returns:
            A list of triggered alerts.
        """
        triggered_alerts = []
        for metric, current_value in current_metrics.items():
            threshold = thresholds.get(metric)
            if threshold is not None and current_value > threshold:
                alert_details = {
                    "metric": metric,
                    "current_value": current_value,
                    "threshold": threshold,
                    "message": f"Performance for '{metric}' ({current_value:.2f}) exceeded threshold ({threshold:.2f}).",
                    "timestamp": time.time()
                }
                self.send_alert("performance_alert", "Critical", alert_details)
                triggered_alerts.append(alert_details)
        
        return triggered_alerts

    def track_error_rates(self, agent_id: str, error_event_timestamp: float, time_window_seconds: int = 600, alert_threshold: int = 5) -> Optional[Dict[str, Any]]:
        """
        Tracks and alerts on increasing error rates for a given agent within a time window.

        Args:
            agent_id: Identifier for the agent.
            error_event_timestamp: Timestamp of the latest error event.
            time_window_seconds: The duration (in seconds) to consider for calculating error rate.
            alert_threshold: The number of errors within the window to trigger an alert.

        Returns:
            A dictionary with alert details if triggered, otherwise None.
        """
        if agent_id not in self.error_history:
            self.error_history[agent_id] = deque()
        
        self.error_history[agent_id].append(error_event_timestamp)

        # Remove old errors outside the time window
        while self.error_history[agent_id] and \
              (error_event_timestamp - self.error_history[agent_id][0]) > time_window_seconds:
            self.error_history[agent_id].popleft()
        
        current_error_count = len(self.error_history[agent_id])

        if current_error_count >= alert_threshold:
            alert_details = {
                "agent_id": agent_id,
                "error_count": current_error_count,
                "time_window_seconds": time_window_seconds,
                "alert_threshold": alert_threshold,
                "message": f"High error rate detected for Agent '{agent_id}': {current_error_count} errors in {time_window_seconds} seconds.",
                "timestamp": time.time()
            }
            self.send_alert("error_rate_alert", "High", alert_details)
            return alert_details
        
        return None

    def configure_alert_rules(self, rule_definitions: List[Dict[str, Any]]):
        """
        Sets up custom alerting conditions based on a list of rule definitions.
        Example rule_definition:
        {
            "rule_name": "high_cpu_usage",
            "metric": "cpu_usage",
            "type": "threshold",
            "threshold": 0.9,
            "severity": "Critical"
        }
        """
        for rule in rule_definitions:
            rule_name = rule.get("rule_name")
            if rule_name:
                self.alert_rules[rule_name] = rule
                logging.info(f"Configured alert rule: {rule_name}")
            else:
                logging.warning(f"Rule definition missing 'rule_name': {rule}")
        logging.info(f"Loaded {len(self.alert_rules)} alert rules.")

    def send_alert(self, alert_type: str, severity: str, details: Dict[str, Any]):
        """
        Dispatches notifications for triggered alerts.
        If a notification_callback is provided, it will be used.
        """
        logging.critical(f"ALERT TRIGGERED - Type: {alert_type}, Severity: {severity}, Details: {json.dumps(details)}")
        if self.notification_callback:
            try:
                self.notification_callback(alert_type, severity, details)
                logging.info("Alert dispatched via callback.")
            except Exception as e:
                logging.error(f"Error sending alert via callback: {e}")
        else:
            logging.info("No notification callback configured. Alert logged only.")

    def set_baseline(self, key: str, value: Any):
        """Sets a baseline value for anomaly detection."""
        self.metric_baselines[key] = value
        logging.info(f"Baseline '{key}' set to: {value}")


# Example Notification Callback
def console_notifier(alert_type: str, severity: str, details: Dict[str, Any]):
    print(f"\n--- !!! ALERT !!! ---")
    print(f"Type: {alert_type}")
    print(f"Severity: {severity}")
    print(f"Details: {json.dumps(details, indent=2)}")
    print(f"----------------------\n")

# Backwards-compat alias expected by tests
class AlertSystem(ObservabilityAlertSystem):
    pass


if __name__ == "__main__":
    alert_system = ObservabilityAlertSystem(notification_callback=console_notifier)

    # Configure some rules
    alert_system.configure_alert_rules([
        {"rule_name": "high_latency", "metric": "api_latency_ms", "type": "threshold", "threshold": 200, "severity": "Warning"},
        {"rule_name": "critical_latency", "metric": "api_latency_ms", "type": "threshold", "threshold": 500, "severity": "Critical"}
    ])

    # Simulate performance monitoring
    print("Monitoring performance...")
    alert_system.monitor_performance_metrics({"api_latency_ms": 100}, {"api_latency_ms": 200}) # No alert
    alert_system.monitor_performance_metrics({"api_latency_ms": 250}, {"api_latency_ms": 200}) # Warning alert
    alert_system.monitor_performance_metrics({"api_latency_ms": 550}, {"api_latency_ms": 500}) # Critical alert

    # Simulate error rate tracking
    print("\nTracking error rates...")
    agent_id = "Agent_Alpha"
    current_time = time.time()
    for i in range(7): # 7 errors in a short window
        alert_system.track_error_rates(agent_id, current_time + i, time_window_seconds=10, alert_threshold=5)
        time.sleep(0.5) # Simulate some time passing

    # Simulate anomaly detection
    print("\nDetecting anomalies...")
    # Set a baseline for CPU usage
    alert_system.set_baseline("system_metrics_baseline", {"avg_cpu": 0.3, "avg_memory": 0.5})
    alert_system.detect_anomalies({"avg_cpu": 0.35, "avg_memory": 0.55}, "system_metrics_baseline") # No anomaly
    alert_system.detect_anomalies({"avg_cpu": 0.6, "avg_memory": 0.5}, "system_metrics_baseline") # Anomaly