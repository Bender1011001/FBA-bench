from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Deque, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A single meta-learning experience sample with arbitrary metrics."""
    metrics: Dict[str, Any]
    ts: float


class MetaLearning:
    """
    Production-ready meta-learning utility with a simple, test-friendly API.

    Capabilities:
    - Record experience dictionaries with metrics
    - Compute robust summaries
    - Adapt hyperparameters via simple rules or weighted heuristics
    - Optional persistence to JSON file
    """

    def __init__(self, window_size: int = 200, persist_path: Optional[str] = None) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._window_size = window_size
        self._buffer: Deque[Experience] = deque(maxlen=window_size)
        self._hyperparams: Dict[str, Any] = {
            "learning_rate": 0.05,
            "exploration_rate": 0.1,
            "regularization": 0.0,
        }
        self._persist_path = persist_path
        self._load()

    def record_experience(self, metrics: Dict[str, Any]) -> None:
        """
        Record an experience sample.
        Expected keys typically include: reward, loss, accuracy, latency_ms, etc.
        """
        ts_val = metrics.get("ts", 0.0)
        try:
            ts_float = float(ts_val)
        except Exception:
            ts_float = 0.0
        exp = Experience(metrics=dict(metrics or {}), ts=ts_float)
        self._buffer.append(exp)
        logger.debug("Recorded experience: %s", metrics)
        self._persist()

    def adapt_hyperparameters(self) -> Dict[str, Any]:
        """
        Adapt hyperparameters using simple heuristics:
        - If average reward decreases and loss increases, slightly increase learning_rate and regularization
        - If performance good and stable, reduce exploration_rate and regularization
        - If latency high, bump regularization (proxy for stability/resource constraints)
        """
        summary = self.summarize()
        avg_reward = summary.get("avg_reward")
        avg_loss = summary.get("avg_loss")
        avg_latency = summary.get("avg_latency_ms")

        lr = float(self._hyperparams["learning_rate"])
        explore = float(self._hyperparams["exploration_rate"])
        reg = float(self._hyperparams["regularization"])

        if avg_reward is not None and avg_loss is not None:
            if avg_reward < 0 and (avg_loss is not None and avg_loss > 0.5):
                lr = min(0.2, lr + 0.01)
                reg = min(0.5, reg + 0.01)
            elif avg_reward > 0.5 and avg_loss < 0.2:
                explore = max(0.01, explore - 0.01)
                reg = max(0.0, reg - 0.005)

        if avg_latency is not None and avg_latency > 1000:
            reg = min(0.5, reg + 0.02)

        self._hyperparams.update(
            learning_rate=round(lr, 5),
            exploration_rate=round(explore, 5),
            regularization=round(reg, 5),
        )
        self._persist()
        return dict(self._hyperparams)

    def summarize(self) -> Dict[str, Any]:
        """Compute summary statistics for the current experience buffer."""
        if not self._buffer:
            return {
                "count": 0,
                "avg_reward": None,
                "avg_loss": None,
                "avg_latency_ms": None,
            }
        rewards: List[float] = []
        losses: List[float] = []
        latencies: List[float] = []
        for exp in self._buffer:
            m = exp.metrics
            if "reward" in m and isinstance(m["reward"], (int, float)):
                rewards.append(float(m["reward"]))
            if "loss" in m and isinstance(m["loss"], (int, float)):
                losses.append(float(m["loss"]))
            if "latency_ms" in m and isinstance(m["latency_ms"], (int, float)):
                latencies.append(float(m["latency_ms"]))

        def _avg(vs: List[float]) -> Optional[float]:
            return round(mean(vs), 6) if vs else None

        return {
            "count": len(self._buffer),
            "avg_reward": _avg(rewards),
            "avg_loss": _avg(losses),
            "avg_latency_ms": _avg(latencies),
        }

    # Persistence helpers
    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            payload = {
                "hyperparams": self._hyperparams,
                "experiences": [{"metrics": e.metrics, "ts": e.ts} for e in self._buffer],
            }
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning("Failed to persist meta-learning state: %s", e)

    def _load(self) -> None:
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            h = data.get("hyperparams")
            if isinstance(h, dict):
                self._hyperparams.update(h)
            exps = data.get("experiences", [])
            for item in exps:
                if not isinstance(item, dict):
                    continue
                ts_val = item.get("ts", 0.0)
                try:
                    ts_float = float(ts_val)
                except Exception:
                    ts_float = 0.0
                self._buffer.append(Experience(metrics=dict(item.get("metrics") or {}), ts=ts_float))
        except Exception as e:
            logger.warning("Failed to load meta-learning state: %s", e)