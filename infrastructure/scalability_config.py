from dataclasses import dataclass

@dataclass
class ScalabilityConfig:
    """
    Configuration for infrastructure scalability features.
    """
    enable_distributed_mode: bool = False
    max_workers: int = 4
    enable_llm_batching: bool = True
    batch_timeout_ms: int = 500
    max_batch_size: int = 10
    enable_fast_forward: bool = True
    fast_forward_threshold: float = 0.1 # Threshold for detecting idle periods
    memory_cleanup_threshold: float = 0.8 # Memory usage percentage to trigger GC
    cost_limit_per_run: float = 100.0 # Maximum LLM API spending limit per run