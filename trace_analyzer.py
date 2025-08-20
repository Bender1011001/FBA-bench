from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


Record = Mapping[str, Any]


@dataclass
class LatencyStats:
    count: int = 0
    min_ms: float = math.inf
    max_ms: float = 0.0
    avg_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p99_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "min_ms": (self.min_ms if self.count else 0.0),
            "max_ms": (self.max_ms if self.count else 0.0),
            "avg_ms": (self.avg_ms if self.count else 0.0),
            "p50_ms": (self.p50_ms if self.count else 0.0),
            "p90_ms": (self.p90_ms if self.count else 0.0),
            "p99_ms": (self.p99_ms if self.count else 0.0),
        }


@dataclass
class AnalysisReport:
    total_records: int
    by_level: Dict[str, int]
    errors: int
    warnings: int
    by_event_name: Dict[str, int]
    by_component: Dict[str, int]
    latency: LatencyStats
    error_samples: List[Dict[str, Any]] = field(default_factory=list)
    warning_samples: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    time_range: Optional[Tuple[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "by_level": dict(self.by_level),
            "errors": self.errors,
            "warnings": self.warnings,
            "by_event_name": dict(self.by_event_name),
            "by_component": dict(self.by_component),
            "latency": self.latency.to_dict(),
            "error_samples": self.error_samples,
            "warning_samples": self.warning_samples,
            "anomalies": self.anomalies,
            "time_range": self.time_range,
        }


class TraceAnalyzer:
    """Analyze heterogeneous trace/log/span records and produce a deterministic summary.

    Input records can be dict-like with any of the following common keys:
    - level: log level (e.g., INFO, WARNING, ERROR)
    - name or event or event_name: event identifier
    - component or source or module: component that emitted the record
    - latency_ms or duration_ms or duration: numeric latency in milliseconds
    - ts or timestamp or time: ISO8601 or epoch seconds
    - error or exception: presence indicates error record

    Methods:
    - analyze(records): produce an AnalysisReport
    - summarize(report): return a concise human-readable summary
    - detect_anomalies(report, thresholds): return anomaly strings (also included in report)
    """

    DEFAULT_ERROR_LEVELS = {"ERROR", "CRITICAL"}
    DEFAULT_WARNING_LEVELS = {"WARNING", "WARN"}

    def analyze(self, records: Iterable[Record], *, sample_limit: int = 10) -> AnalysisReport:
        by_level: Counter[str] = Counter()
        by_event: Counter[str] = Counter()
        by_component: Counter[str] = Counter()
        latencies: List[float] = []
        error_samples: List[Dict[str, Any]] = []
        warning_samples: List[Dict[str, Any]] = []
        timestamps: List[float] = []

        total = 0
        for rec in records:
            if rec is None:
                continue
            total += 1
            level = self._extract_level(rec)
            event = self._extract_event(rec)
            component = self._extract_component(rec)
            latency = self._extract_latency_ms(rec)
            ts = self._extract_timestamp(rec)

            if level:
                by_level[level] += 1
            if event:
                by_event[event] += 1
            if component:
                by_component[component] += 1
            if latency is not None and latency >= 0:
                latencies.append(float(latency))
            if ts is not None:
                timestamps.append(ts)

            # Sample errors/warnings
            if self._is_error(level, rec):
                if len(error_samples) < sample_limit:
                    error_samples.append(self._compact_record(rec))
            elif self._is_warning(level, rec):
                if len(warning_samples) < sample_limit:
                    warning_samples.append(self._compact_record(rec))

        latency_stats = self._latency_summary(latencies)
        time_range = self._time_range_summary(timestamps)

        report = AnalysisReport(
            total_records=total,
            by_level=dict(by_level),
            errors=sum(by_level[l] for l in self.DEFAULT_ERROR_LEVELS if l in by_level),
            warnings=sum(by_level[l] for l in self.DEFAULT_WARNING_LEVELS if l in by_level),
            by_event_name=dict(by_event),
            by_component=dict(by_component),
            latency=latency_stats,
            error_samples=error_samples,
            warning_samples=warning_samples,
            time_range=time_range,
        )
        # Populate anomalies with default thresholds
        report.anomalies = self.detect_anomalies(
            report,
            thresholds={
                "error_rate": 0.05,       # 5%
                "warning_rate": 0.15,     # 15%
                "p99_latency_ms": 5000.0, # 5s
            },
        )
        return report

    def summarize(self, report: AnalysisReport) -> str:
        tr = f"{report.time_range[0]} .. {report.time_range[1]}" if report.time_range else "n/a"
        parts = [
            f"Records={report.total_records}",
            f"Errors={report.errors}",
            f"Warnings={report.warnings}",
            f"Levels={report.by_level}",
            f"TopEvents={self._topk_to_str(report.by_event_name)}",
            f"TopComponents={self._topk_to_str(report.by_component)}",
            f"Latency(ms)={report.latency.to_dict()}",
            f"TimeRange={tr}",
        ]
        if report.anomalies:
            parts.append(f"Anomalies={report.anomalies}")
        return " | ".join(parts)

    def detect_anomalies(
        self,
        report: AnalysisReport,
        thresholds: Optional[Mapping[str, float]] = None,
    ) -> List[str]:
        thresholds = dict(thresholds or {})
        error_rate_thr = float(thresholds.get("error_rate", 0.1))
        warning_rate_thr = float(thresholds.get("warning_rate", 0.25))
        p99_latency_thr = float(thresholds.get("p99_latency_ms", 10000.0))

        anomalies: List[str] = []
        denom = max(1, report.total_records)
        error_rate = report.errors / denom
        warning_rate = report.warnings / denom

        if error_rate > error_rate_thr:
            anomalies.append(f"HighErrorRate({error_rate:.2%} > {error_rate_thr:.2%})")
        if warning_rate > warning_rate_thr:
            anomalies.append(f"HighWarningRate({warning_rate:.2%} > {warning_rate_thr:.2%})")
        if report.latency.p99_ms and report.latency.p99_ms > p99_latency_thr:
            anomalies.append(f"HighP99Latency({report.latency.p99_ms:.1f}ms > {p99_latency_thr:.1f}ms)")

        # Check for bursty components/events (simple heuristic: dominant share > 70%)
        top_event, top_event_cnt = self._top1(report.by_event_name)
        if top_event_cnt > 0 and top_event_cnt / denom > 0.7:
            anomalies.append(f"EventConcentration({top_event}:{top_event_cnt}/{denom})")
        top_comp, top_comp_cnt = self._top1(report.by_component)
        if top_comp_cnt > 0 and top_comp_cnt / denom > 0.7:
            anomalies.append(f"ComponentConcentration({top_comp}:{top_comp_cnt}/{denom})")

        return anomalies

    # -------------------- Extraction helpers --------------------

    @staticmethod
    def _extract_level(rec: Record) -> Optional[str]:
        v = rec.get("level") or rec.get("levelname") or rec.get("severity") or rec.get("severity_text")
        if isinstance(v, str):
            return v.upper()
        return None

    @staticmethod
    def _extract_event(rec: Record) -> Optional[str]:
        v = rec.get("event_name") or rec.get("event") or rec.get("name") or rec.get("message")
        if v is None:
            # Try OpenTelemetry-style attributes
            attrs = rec.get("attributes") or {}
            if isinstance(attrs, Mapping):
                v = attrs.get("event_name") or attrs.get("http.route") or attrs.get("db.statement")
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return None

    @staticmethod
    def _extract_component(rec: Record) -> Optional[str]:
        v = rec.get("component") or rec.get("source") or rec.get("module") or rec.get("logger")
        if v is None:
            attrs = rec.get("attributes") or {}
            if isinstance(attrs, Mapping):
                v = attrs.get("service.name") or attrs.get("component")
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return None

    @staticmethod
    def _extract_latency_ms(rec: Record) -> Optional[float]:
        v = rec.get("latency_ms")
        if v is None:
            v = rec.get("duration_ms")
        if v is None:
            v = rec.get("duration")
        if v is None:
            # OTel span kind: end_time - start_time
            start = rec.get("start_time") or rec.get("startTime")
            end = rec.get("end_time") or rec.get("endTime")
            try:
                if start is not None and end is not None:
                    start_s = TraceAnalyzer._to_epoch_seconds(start)
                    end_s = TraceAnalyzer._to_epoch_seconds(end)
                    v = (end_s - start_s) * 1000.0
            except Exception:
                v = None
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_timestamp(rec: Record) -> Optional[float]:
        v = rec.get("ts") or rec.get("timestamp") or rec.get("time")
        if v is None:
            # try OTel times
            v = rec.get("start_time") or rec.get("end_time")
        try:
            return TraceAnalyzer._to_epoch_seconds(v) if v is not None else None
        except Exception:
            return None

    @staticmethod
    def _to_epoch_seconds(value: Any) -> float:
        # Accept epoch seconds, epoch ms, or ISO8601 strings
        if isinstance(value, (int, float)):
            # Heuristic: treat large numbers as ms
            if value > 1e12:  # nanoseconds
                return value / 1e9
            if value > 1e10:  # milliseconds
                return value / 1e3
            return float(value)
        if isinstance(value, str):
            # Try ISO8601
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.timestamp()
            except Exception:
                # Try float-like
                return float(value)
        # Unknown
        raise ValueError(f"Unsupported timestamp format: {value!r}")

    @staticmethod
    def _is_error(level: Optional[str], rec: Record) -> bool:
        if rec.get("error") or rec.get("exception"):
            return True
        return level in TraceAnalyzer.DEFAULT_ERROR_LEVELS if level else False

    @staticmethod
    def _is_warning(level: Optional[str], rec: Record) -> bool:
        return level in TraceAnalyzer.DEFAULT_WARNING_LEVELS if level else False

    @staticmethod
    def _compact_record(rec: Record) -> Dict[str, Any]:
        keys = ("time", "timestamp", "ts", "level", "message", "event", "event_name", "component", "source", "module", "logger", "error")
        compact: Dict[str, Any] = {}
        for k in keys:
            if k in rec:
                compact[k] = rec[k]
        # Include id/span ids if present
        for k in ("trace_id", "span_id", "traceId", "spanId"):
            if k in rec:
                compact[k] = rec[k]
        return compact

    # -------------------- Math helpers --------------------

    @staticmethod
    def _latency_summary(latencies_ms: Sequence[float]) -> LatencyStats:
        stats = LatencyStats()
        n = len(latencies_ms)
        if n == 0:
            return stats
        arr = sorted(latencies_ms)
        stats.count = n
        stats.min_ms = arr[0]
        stats.max_ms = arr[-1]
        stats.avg_ms = sum(arr) / n
        stats.p50_ms = TraceAnalyzer._percentile(arr, 0.50)
        stats.p90_ms = TraceAnalyzer._percentile(arr, 0.90)
        stats.p99_ms = TraceAnalyzer._percentile(arr, 0.99)
        return stats

    @staticmethod
    def _percentile(arr_sorted: Sequence[float], p: float) -> float:
        if not arr_sorted:
            return 0.0
        k = (len(arr_sorted) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return arr_sorted[int(k)]
        d0 = arr_sorted[f] * (c - k)
        d1 = arr_sorted[c] * (k - f)
        return d0 + d1

    @staticmethod
    def _time_range_summary(timestamps: Sequence[float]) -> Optional[Tuple[str, str]]:
        if not timestamps:
            return None
        start = min(timestamps)
        end = max(timestamps)
        return datetime.utcfromtimestamp(start).isoformat() + "Z", datetime.utcfromtimestamp(end).isoformat() + "Z"

    @staticmethod
    def _topk_to_str(d: Mapping[str, int], k: int = 3) -> str:
        if not d:
            return "[]"
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return "[" + ", ".join(f"{name}:{cnt}" for name, cnt in items) + "]"

    @staticmethod
    def _top1(d: Mapping[str, int]) -> Tuple[str, int]:
        if not d:
            return "", 0
        name, cnt = max(d.items(), key=lambda kv: kv[1])
        return name, cnt