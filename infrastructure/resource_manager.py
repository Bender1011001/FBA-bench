import logging
import time
import psutil # For system resource monitoring
from collections import defaultdict
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages and monitors simulation resources, including LLM tokens, memory, and performance.

    - Token budgeting: Manages LLM token allocation across agents and operations.
    - Memory monitoring: Tracks memory usage and triggers cleanup.
    - Performance optimization: Identifies bottlenecks and optimization opportunities.
    - Cost tracking: Monitors API costs and enforces spending limits.
    """

    def __init__(self):
        self._token_budgets: Dict[str, float] = defaultdict(lambda: float('inf')) # agent_id -> remaining_tokens
        self._global_token_cap: float = float('inf')
        self._token_usage_tracking: Dict[str, float] = defaultdict(float) # agent_id or operation_type -> tokens used
        self._current_memory_usage: float = 0.0 # in MB
        self._llm_api_costs: Dict[str, float] = defaultdict(float) # model_name -> total cost
        self._total_api_cost: float = 0.0
        self._spending_limit: float = float('inf')
        
        self.stats = {
            "llm_tokens_allocated": 0,
            "llm_api_cost_incurred": 0.0,
            "current_memory_mb": 0.0,
            "max_memory_mb": 0.0,
            "cpu_utilization_percent": 0.0,
            "disk_io_mbs": 0.0,
            "network_io_mbs": 0.0,
            "performance_bottlenecks": [],
            "optimization_suggestions": []
        }
        logger.info("ResourceManager initialized.")

    def set_token_budget(self, agent_id: str, budget: float):
        """Sets an LLM token budget for a specific agent."""
        if budget < 0:
            raise ValueError("Token budget cannot be negative.")
        self._token_budgets[agent_id] = budget
        logger.info(f"Set token budget for {agent_id}: {budget} tokens.")

    def set_global_token_cap(self, cap: float):
        """Sets a global LLM token cap for all operations."""
        if cap < 0:
            raise ValueError("Global token cap cannot be negative.")
        self._global_token_cap = cap
        logger.info(f"Set global LLM token cap: {cap} tokens.")

    def allocate_tokens(self, agent_id: str = "system", operation_type: str = "general", requested_tokens: int = 0) -> bool:
        """
        Manages LLM token allocation, checking against agent and global budgets.

        Args:
            agent_id: The ID of the agent requesting tokens (or "system").
            operation_type: Type of operation (e.g., "prompt", "response", "reflection").
            requested_tokens: Number of tokens requested.

        Returns:
            True if tokens are allocated, False otherwise.
        """
        if requested_tokens < 0:
            logger.warning(f"Invalid token allocation request: negative tokens {requested_tokens}.")
            return False

        # Check agent-specific budget
        if self._token_budgets[agent_id] != float('inf') and self._token_budgets[agent_id] < requested_tokens:
            logger.warning(f"Agent {agent_id} budget exceeded. Requested: {requested_tokens}, Available: {self._token_budgets[agent_id]}.")
            return False
        
        # Check global token cap
        if self._global_token_cap != float('inf') and self.stats["llm_tokens_allocated"] + requested_tokens > self._global_token_cap:
            logger.warning(f"Global token cap of {self._global_token_cap} exceeded. Cannot allocate {requested_tokens} tokens.")
            return False

        # Check total API cost limit
        if self._spending_limit != float('inf') and self._total_api_cost >= self._spending_limit:
            logger.warning(f"Global spending limit of ${self._spending_limit:.2f} reached. Cannot allocate more tokens.")
            return False


        # Allocate tokens
        self._token_budgets[agent_id] -= requested_tokens
        self.stats["llm_tokens_allocated"] += requested_tokens
        self._token_usage_tracking[agent_id] += requested_tokens
        self._token_usage_tracking[operation_type] += requested_tokens
        logger.debug(f"Allocated {requested_tokens} tokens for {agent_id}/{operation_type}. Remaining budget for {agent_id}: {self._token_budgets[agent_id]}")
        return True

    def record_llm_cost(self, model_name: str, cost: float, tokens: int):
        """Records the actual cost incurred from an LLM API call."""
        if cost < 0 or tokens < 0:
            logger.warning(f"Invalid cost or token value recorded: cost={cost}, tokens={tokens}.")
            return
            
        self._llm_api_costs[model_name] += cost
        self._total_api_cost += cost
        self.stats["llm_api_cost_incurred"] += cost
        self.stats["llm_tokens_allocated"] += tokens # Also track actual tokens
        
        # Enforce cost limits immediately upon recording
        if self._spending_limit != float('inf') and self._total_api_cost > self._spending_limit:
            logger.error(f"WARNING: Spending limit of ${self._spending_limit:.2f} exceeded! Current cost: ${self._total_api_cost:.2f}. Consider stopping simulation.")

        logger.info(f"Recorded LLM cost for {model_name}: ${cost:.6f} ({tokens} tokens). Total cost: ${self._total_api_cost:.2f}")

    def get_current_token_usage(self, key: str = "total") -> float:
        """Returns token usage for a specific agent/operation, or total."""
        if key == "total":
            return self.stats["llm_tokens_allocated"]
        return self._token_usage_tracking.get(key, 0.0)

    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Tracks system memory consumption (process specific and total system).
        Returns: Dict with 'process_memory_mb', 'system_total_gb', 'system_available_gb', 'system_percent'
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        self._current_memory_usage = mem_info.rss / (1024 * 1024)  # Resident Set Size in MB
        
        system_memory = psutil.virtual_memory()
        
        self.stats["current_memory_mb"] = self._current_memory_usage
        self.stats["max_memory_mb"] = max(self.stats["max_memory_mb"], self._current_memory_usage)
        
        mem_metrics = {
            "process_memory_mb": self._current_memory_usage,
            "system_total_gb": system_memory.total / (1024**3),
            "system_available_gb": system_memory.available / (1024**3),
            "system_percent": system_memory.percent
        }
        logger.debug(f"Memory usage: Process: {mem_metrics['process_memory_mb']:.2f}MB, System: {mem_metrics['system_percent']}%")
        return mem_metrics

    def trigger_garbage_collection(self, threshold_percent: float = 0.8):
        """
        Triggers garbage collection if memory usage exceeds a threshold.
        This is a soft suggestion; actual GC behavior is Python's discretion.
        """
        mem_metrics = self.monitor_memory_usage()
        if mem_metrics["system_percent"] > threshold_percent * 100:
            logger.warning(f"System memory usage {mem_metrics['system_percent']}% exceeds threshold {threshold_percent*100}%. Triggering garbage collection.")
            import gc
            gc.collect()
            logger.info("Garbage collection triggered.")
            # Re-monitor after GC
            self.monitor_memory_usage()


    def optimize_performance(self) -> List[str]:
        """
        Identifies bottlenecks and suggests optimization opportunities.
        This method is heuristic-based and would typically leverage historical
        performance data and predefined rules.
        """
        bottlenecks = self.detect_bottlenecks(self.get_resource_metrics())
        suggestions = self.suggest_optimizations(bottlenecks)
        self.stats["performance_bottlenecks"] = bottlenecks
        self.stats["optimization_suggestions"] = suggestions
        if suggestions:
            logger.info(f"Performance optimization suggestions: {', '.join(suggestions)}")
        return suggestions

    def enforce_cost_limits(self, spending_limit: float):
        """
        Sets a total spending limit for LLM API calls and prevents budget overruns.
        Returns True if within limits, False if limit is exceeded.
        """
        if spending_limit < 0:
            raise ValueError("Spending limit cannot be negative.")
        self._spending_limit = spending_limit
        logger.info(f"Set global LLM API spending limit to ${spending_limit:.2f}.")
        
        if self._total_api_cost > self._spending_limit:
            logger.error(f"Current total API cost ${self._total_api_cost:.2f} already exceeds spending limit ${self._spending_limit:.2f}!")
            return False
        return True

    def get_total_api_cost(self) -> float:
        """Returns the total estimated LLM API cost incurred."""
        return self._total_api_cost

    def get_resource_metrics(self) -> Dict[str, Any]:
        """Collects and returns key resource metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1) # Non-blocking for an interval
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()

        # Calculate deltas for I/O
        # Store previous values to calculate deltas. For first call, just return current.
        if not hasattr(self, '_prev_disk_io'):
            self._prev_disk_io = disk_io
            self._prev_net_io = net_io
            disk_read_mbs = 0.0
            disk_write_mbs = 0.0
            net_sent_mbs = 0.0
            net_recv_mbs = 0.0
        else:
            time_delta = 0.1 # This should match the interval for cpu_percent or be measured precisely
            disk_read_mbs = (disk_io.read_bytes - self._prev_disk_io.read_bytes) / (1024 * 1024) / time_delta
            disk_write_mbs = (disk_io.write_bytes - self._prev_disk_io.write_bytes) / (1024 * 1024) / time_delta
            net_sent_mbs = (net_io.bytes_sent - self._prev_net_io.bytes_sent) / (1024 * 1024) / time_delta
            net_recv_mbs = (net_io.bytes_recv - self._prev_net_io.bytes_recv) / (1024 * 1024) / time_delta
            self._prev_disk_io = disk_io
            self._prev_net_io = net_io

        self.stats["cpu_utilization_percent"] = cpu_percent
        self.stats["disk_io_mbs"] = disk_read_mbs + disk_write_mbs
        self.stats["network_io_mbs"] = net_sent_mbs + net_recv_mbs

        mem_metrics = self.monitor_memory_usage() # Updates self.stats["current_memory_mb"] and "max_memory_mb"]

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": mem_metrics["system_percent"],
            "process_memory_mb": mem_metrics["process_memory_mb"],
            "disk_read_mbs": disk_read_mbs,
            "disk_write_mbs": disk_write_mbs,
            "net_sent_mbs": net_sent_mbs,
            "net_recv_mbs": net_recv_mbs,
            "total_llm_tokens": self.stats["llm_tokens_allocated"],
            "total_llm_cost": self.stats["llm_api_cost_incurred"],
            "timestamp": time.time()
        }

    def detect_bottlenecks(self, metrics: Dict[str, Any]) -> List[str]:
        """Identifies performance issues based on current metrics and predefined thresholds."""
        bottlenecks = []
        if metrics["cpu_percent"] > 90:
            bottlenecks.append("High CPU utilization")
        if metrics["memory_percent"] > 85: # System-wide memory
            bottlenecks.append("High System Memory Usage")
        elif metrics["process_memory_mb"] > 2000: # Example threshold for process
            bottlenecks.append("High Process Memory Usage")
        if metrics["disk_read_mbs"] > 100 or metrics["disk_write_mbs"] > 100:
            bottlenecks.append("High Disk I/O")
        if metrics["net_sent_mbs"] > 50 or metrics["net_recv_mbs"] > 50:
            bottlenecks.append("High Network I/O")
        if self._total_api_cost >= self._spending_limit * 0.9: # Approaching limit
            bottlenecks.append("Approaching LLM Cost Limit")
        
        return bottlenecks

    def suggest_optimizations(self, bottlenecks: List[str]) -> List[str]:
        """Recommends configuration improvements based on identified bottlenecks."""
        suggestions = []
        if "High CPU utilization" in bottlenecks:
            suggestions.append("Consider reducing concurrent agent activity or using more efficient algorithms.")
            suggestions.append("Distribute simulation across more workers/cores.")
        if "High System Memory Usage" in bottlenecks or "High Process Memory Usage" in bottlenecks:
            suggestions.append("Trigger garbage collection more frequently.")
            suggestions.append("Optimize data structures to reduce memory footprint.")
            suggestions.append("Implement event compression/persistence to offload old events.")
        if "High Disk I/O" in bottlenecks:
            suggestions.append("Optimize data serialization/deserialization for persistence layers.")
            suggestions.append("Consider faster storage solutions.")
        if "High Network I/O" in bottlenecks:
            suggestions.append("Optimize cross-process communication protocols.")
            suggestions.append("Batch smaller network requests into larger ones.")
        if "Approaching LLM Cost Limit" in bottlenecks:
            suggestions.append("Increase LLM batching aggressiveness (larger batches, longer timeouts).")
            suggestions.append("Review agent prompts for verbosity and unnecessary calls.")
            suggestions.append("Utilize fast-forward simulation during idle periods.")
            suggestions.append("Consider switching to cheaper LLM models for less critical tasks.")

        return suggestions

    def set_performance_thresholds(self, cpu: float = None, memory: float = None, network: float = None) -> None:
        """
        Configures thresholds for performance alerts.
        (Future expansion: these thresholds would be used by a background monitoring task)
        """
        # For simplicity, these values are directly used in detect_bottlenecks now.
        # In a more advanced system, these would update internal state used by a periodic monitor.
        logger.info(f"Performance thresholds (conceptual) set: CPU={cpu}%, Memory={memory}%, Network={network}MB/s.")