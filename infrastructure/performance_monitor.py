import asyncio
import time
import os
from collections import deque, defaultdict
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging # Ensure logging is imported directly

# Attempt to import psutil, but make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. System resource monitoring will be limited. Please install it with 'pip install psutil'.")

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitors system performance, identifies bottlenecks, and suggests optimizations.

    - Bottleneck detection: Identifies performance issues in real-time.
    - Resource utilization: Tracks CPU, memory, and network usage.
    - Optimization suggestions: Recommends configuration improvements.
    - Performance reporting: Generates detailed performance analysis.
    """

    def __init__(self, resource_manager: Any = None): # Use Any to avoid circular deps with ResourceManager
        self.resource_manager = resource_manager # Reference to ResourceManager for integrated metrics
        self._metrics_history: deque = deque(maxlen=60) # Store last 60 seconds of metrics
        self._monitoring_interval: float = 1.0 # seconds between metric collection
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None

        # Default thresholds (can be set via set_performance_thresholds)
        self._cpu_threshold: float = 85.0 # percent
        self._memory_threshold_percent: float = 85.0 # system memory percent
        self._network_threshold_mbs: float = 50.0 # MB/s
        self._disk_threshold_mbs: float = 100.0 # MB/s

        logger.info("PerformanceMonitor initialized.")

    async def start(self):
        """Starts the background monitoring task."""
        if self._running:
            logger.warning("PerformanceMonitor already running.")
            return
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("PerformanceMonitor started.")

    async def stop(self):
        """Stops the background monitoring task."""
        if not self._running:
            return
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("PerformanceMonitor stopped.")

    async def _monitoring_loop(self):
        """Periodically collects metrics and checks for bottlenecks."""
        logger.info("Performance monitoring loop started.")
        while self._running:
            try:
                metrics = self.monitor_system_resources()
                self._metrics_history.append(metrics)
                
                bottlenecks = self.detect_bottlenecks(metrics)
                if bottlenecks:
                    suggestions = self.suggest_optimizations(bottlenecks)
                    logger.warning(f"Detected bottlenecks: {bottlenecks}. Suggestions: {suggestions}")
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}", exc_info=True)
            
            await asyncio.sleep(self._monitoring_interval)
        logger.info("Performance monitoring loop stopped.")

    def monitor_system_resources(self) -> Dict[str, Any]:
        """
        Tracks CPU, memory, network, and disk usage.
        Integrates with ResourceManager if available.
        """
        metrics = {}
        if self.resource_manager:
            metrics = self.resource_manager.get_resource_metrics()
            logger.debug("Collected metrics from ResourceManager.")
        else:
            # Fallback to direct psutil if ResourceManager is not provided
            metrics["cpu_percent"] = psutil.cpu_percent(interval=None) # Non-blocking call
            mem_info = psutil.virtual_memory()
            metrics["memory_percent"] = mem_info.percent
            process = psutil.Process(os.getpid())
            metrics["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
            # Disk and network I/O require tracking deltas, which ResourceManager handles.
            # For simplicity, if no ResourceManager, we'll just report instantaneous or 0
            metrics["disk_read_mbs"] = 0.0
            metrics["disk_write_mbs"] = 0.0
            metrics["net_sent_mbs"] = 0.0
            metrics["net_recv_mbs"] = 0.0
            metrics["total_llm_tokens"] = 0 # Not trackable without RM
            metrics["total_llm_cost"] = 0.0 # Not trackable without RM
            metrics["timestamp"] = time.time()
            logger.debug("Collected metrics directly via psutil.")
        
        return metrics

    def detect_bottlenecks(self, metrics: Dict[str, Any]) -> List[str]:
        """Identifies performance issues based on current metrics and configured thresholds."""
        bottlenecks = []
        if metrics["cpu_percent"] > self._cpu_threshold:
            bottlenecks.append(f"High CPU utilization ({metrics['cpu_percent']:.1f}%)")
        
        if metrics["memory_percent"] > self._memory_threshold_percent:
            bottlenecks.append(f"High System Memory Usage ({metrics['memory_percent']:.1f}%)")

        if metrics.get("process_memory_mb", 0) > 2000: # General threshold, can be configured
             bottlenecks.append(f"High Process Memory Usage ({metrics['process_memory_mb']:.1f}MB)")

        if metrics["disk_read_mbs"] > self._disk_threshold_mbs or metrics["disk_write_mbs"] > self._disk_threshold_mbs:
            bottlenecks.append(f"High Disk I/O (Read: {metrics['disk_read_mbs']:.1f}MB/s, Write: {metrics['disk_write_mbs']:.1f}MB/s)")
        
        if metrics["net_sent_mbs"] > self._network_threshold_mbs or metrics["net_recv_mbs"] > self._network_threshold_mbs:
            bottlenecks.append(f"High Network I/O (Sent: {metrics['net_sent_mbs']:.1f}MB/s, Recv: {metrics['net_recv_mbs']:.1f}MB/s)")
        
        # If ResourceManager is available, check cost limits
        if self.resource_manager:
            if self.resource_manager.get_total_api_cost() >= self.resource_manager._spending_limit * 0.9:
                bottlenecks.append("Approaching LLM Cost Limit")

        return bottlenecks

    def suggest_optimizations(self, bottlenecks: List[str]) -> List[str]:
        """Recommends configuration improvements and actions based on identified bottlenecks."""
        suggestions = []
        if "High CPU utilization" in "".join(bottlenecks):
            suggestions.append("Consider reducing concurrent agent activity or using more efficient algorithms.")
            suggestions.append("Distribute simulation across more workers/cores.")
        if "High System Memory Usage" in "".join(bottlenecks) or "High Process Memory Usage" in "".join(bottlenecks):
            suggestions.append("Trigger garbage collection more frequently (via ResourceManager).")
            suggestions.append("Optimize data structures to reduce memory footprint.")
            suggestions.append("Implement event compression/persistence to offload old events (via EventBus).")
        if "High Disk I/O" in "".join(bottlenecks):
            suggestions.append("Optimize data serialization/deserialization for persistence layers.")
            suggestions.append("Consider faster storage solutions (e.g., SSD).")
        if "High Network I/O" in "".join(bottlenecks):
            suggestions.append("Optimize cross-process communication protocols.")
            suggestions.append("Batch smaller network requests into larger ones (especially for LLM calls).")
        if "Approaching LLM Cost Limit" in "".join(bottlenecks):
            suggestions.append("Increase LLM batching aggressiveness (larger batches, longer timeouts).")
            suggestions.append("Review agent prompts for verbosity and unnecessary calls.")
            suggestions.append("Utilize fast-forward simulation during idle periods.")
            suggestions.append("Consider switching to cheaper LLM models for less critical tasks.")

        return suggestions

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generates a detailed performance analysis report."""
        if not self._metrics_history:
            return {"status": "No metrics collected yet."}

        # Calculate averages, min/max from history
        report_metrics = defaultdict(float)
        max_metrics = defaultdict(float)
        min_metrics = defaultdict(lambda: float('inf'))

        for metrics_snapshot in self._metrics_history:
            for key, value in metrics_snapshot.items():
                if isinstance(value, (int, float)):
                    report_metrics[key] += value
                    max_metrics[key] = max(max_metrics[key], value)
                    min_metrics[key] = min(min_metrics[key], value)
        
        num_snapshots = len(self._metrics_history)
        for key in report_metrics:
            report_metrics[key] /= num_snapshots # Calculate average

        report = {
            "summary": "Performance Report",
            "timestamp": datetime.now().isoformat(),
            "num_data_points": num_snapshots,
            "avg_metrics": {k: f"{v:.2f}" for k, v in report_metrics.items()},
            "max_metrics": {k: f"{v:.2f}" for k, v in max_metrics.items() if max_metrics[k] != 0.0},
            "min_metrics": {k: f"{v:.2f}" for k, v in min_metrics.items() if min_metrics[k] != float('inf')},
            "identified_bottlenecks": self.detect_bottlenecks(report_metrics), # Run detection on averages
            "optimization_suggestions": self.suggest_optimizations(self.detect_bottlenecks(report_metrics)),
            "monitoring_interval_seconds": self._monitoring_interval
        }
        logger.info("Generated performance report.")
        return report

    def set_performance_thresholds(self, cpu: Optional[float] = None, memory: Optional[float] = None, network: Optional[float] = None, disk: Optional[float] = None):
        """
        Configures thresholds for performance alerts.
        
        Args:
            cpu: CPU utilization threshold percentage (0-100).
            memory: System memory utilization threshold percentage (0-100).
            network: Network I/O threshold in MB/s.
            disk: Disk I/O threshold in MB/s.
        """
        if cpu is not None: self._cpu_threshold = cpu
        if memory is not None: self._memory_threshold_percent = memory
        if network is not None: self._network_threshold_mbs = network
        if disk is not None: self._disk_threshold_mbs = disk
        logger.info(f"Performance thresholds updated: CPU={self._cpu_threshold}%, Mem={self._memory_threshold_percent}%, Net={self._network_threshold_mbs}MB/s, Disk={self._disk_threshold_mbs}MB/s.")