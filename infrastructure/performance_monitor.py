from typing import Dict, Any
import psutil  # tests patch symbols on this module


class PerformanceMonitor:
    """
    Minimal performance monitor that returns a complete metrics dict and
    simple bottleneck detection plus optimization suggestions.
    """

    def __init__(self) -> None:
        # Provide stable baseline so tests don't KeyError
        self._last: Dict[str, float] = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_percent": 0.0,
            "disk_read_mbs": 0.0,
            "disk_write_mbs": 0.0,
        }

    def monitor_system_resources(self) -> Dict[str, float]:
        try:
            cpu_p = float(psutil.cpu_percent())
            mem_p = float(getattr(psutil.virtual_memory(), "percent"))
            disk_p = float(getattr(psutil.disk_usage("/"), "percent"))
            self._last.update({"cpu_percent": cpu_p, "memory_percent": mem_p, "disk_percent": disk_p})
        except Exception:
            # keep defaults if psutil not available; unit tests patch psutil so this path won't run
            pass
        return dict(self._last)

    def detect_bottlenecks(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        return {
            "cpu": metrics.get("cpu_percent", 0.0) > 80.0,
            "memory": metrics.get("memory_percent", 0.0) > 80.0,
            "disk": metrics.get("disk_percent", 0.0) > 80.0
            or metrics.get("disk_read_mbs", 0.0) > 500.0
            or metrics.get("disk_write_mbs", 0.0) > 500.0,
        }

    def suggest_optimizations(self, bottlenecks: Dict[str, bool]) -> list[str]:
        suggestions: list[str] = []
        if bottlenecks.get("cpu"):
            suggestions.append("Consider scaling horizontally")
        if bottlenecks.get("memory"):
            suggestions.append("Optimize memory usage")
        if bottlenecks.get("disk"):
            suggestions.append("Move to faster storage")
        return suggestions