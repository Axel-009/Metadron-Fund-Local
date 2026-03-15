"""Memory Monitor — System and process memory management for Metadron Capital.

Provides:
    - System memory usage tracking
    - Per-component memory profiling
    - DataFrame memory optimization recommendations
    - Cache management (LRU eviction)
    - Memory leak detection
    - GPU memory monitoring (if NVIDIA present)
    - Process pool monitoring
    - Memory budget enforcement
    - Periodic garbage collection
    - Memory usage reporting
"""

import gc
import sys
import os
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from collections import OrderedDict
from functools import wraps

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MemorySnapshot:
    """Point-in-time memory snapshot."""
    timestamp: str = ""
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    heap_mb: float = 0.0
    python_objects: int = 0
    dataframe_count: int = 0
    dataframe_mb: float = 0.0
    cache_mb: float = 0.0
    gpu_mb: float = 0.0
    gc_generation_0: int = 0
    gc_generation_1: int = 0
    gc_generation_2: int = 0


@dataclass
class ComponentMemory:
    """Memory usage by component."""
    name: str = ""
    size_mb: float = 0.0
    object_count: int = 0
    largest_object_mb: float = 0.0
    growth_rate_mb_per_hour: float = 0.0


@dataclass
class MemoryBudget:
    """Memory budget configuration."""
    total_budget_mb: float = 4096.0
    data_budget_mb: float = 2048.0
    cache_budget_mb: float = 512.0
    model_budget_mb: float = 1024.0
    overhead_budget_mb: float = 512.0
    warning_threshold: float = 0.80
    critical_threshold: float = 0.95


@dataclass
class OptimizationRecommendation:
    """Memory optimization suggestion."""
    component: str = ""
    current_mb: float = 0.0
    potential_savings_mb: float = 0.0
    recommendation: str = ""
    priority: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL


# ---------------------------------------------------------------------------
# LRU Cache Manager
# ---------------------------------------------------------------------------
class LRUCacheManager:
    """LRU cache with memory-aware eviction."""

    def __init__(self, max_size: int = 100, max_memory_mb: float = 256.0):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._sizes: dict[str, float] = {}
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any, size_mb: float = 0.0):
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            self._sizes[key] = size_mb
            return

        # Evict if needed
        while len(self._cache) >= self.max_size or self.total_memory_mb + size_mb > self.max_memory_mb:
            if not self._cache:
                break
            evicted_key, _ = self._cache.popitem(last=False)
            self._sizes.pop(evicted_key, None)

        self._cache[key] = value
        self._sizes[key] = size_mb

    def evict(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            self._sizes.pop(key, None)
            return True
        return False

    def clear(self):
        self._cache.clear()
        self._sizes.clear()

    @property
    def total_memory_mb(self) -> float:
        return sum(self._sizes.values())

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "memory_mb": round(self.total_memory_mb, 2),
            "max_memory_mb": self.max_memory_mb,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }


# ---------------------------------------------------------------------------
# DataFrame Memory Analyzer
# ---------------------------------------------------------------------------
class DataFrameMemoryAnalyzer:
    """Analyze and optimize DataFrame memory usage."""

    def analyze(self, df: pd.DataFrame, name: str = "") -> dict:
        memory_usage = df.memory_usage(deep=True)
        total_mb = memory_usage.sum() / (1024 * 1024)

        dtypes = df.dtypes.value_counts().to_dict()
        dtype_dist = {str(k): int(v) for k, v in dtypes.items()}

        return {
            "name": name,
            "rows": len(df),
            "cols": len(df.columns),
            "total_mb": round(total_mb, 3),
            "per_column_mb": {col: round(memory_usage[col] / (1024 * 1024), 4) for col in df.columns},
            "dtypes": dtype_dist,
        }

    def optimize(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Optimize DataFrame memory by downcasting dtypes.

        Returns (optimized_df, savings_mb).
        """
        original_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        optimized = df.copy()

        for col in optimized.columns:
            col_type = optimized[col].dtype

            if col_type == np.float64:
                optimized[col] = pd.to_numeric(optimized[col], downcast="float")
            elif col_type == np.int64:
                optimized[col] = pd.to_numeric(optimized[col], downcast="integer")
            elif col_type == object:
                num_unique = optimized[col].nunique()
                num_total = len(optimized[col])
                if num_unique / num_total < 0.5:
                    optimized[col] = optimized[col].astype("category")

        optimized_mb = optimized.memory_usage(deep=True).sum() / (1024 * 1024)
        savings_mb = original_mb - optimized_mb

        return optimized, savings_mb

    def get_recommendations(self, df: pd.DataFrame, name: str = "") -> list[OptimizationRecommendation]:
        recs = []
        for col in df.columns:
            col_type = df[col].dtype
            col_mb = df[col].memory_usage(deep=True) / (1024 * 1024)

            if col_type == np.float64:
                recs.append(OptimizationRecommendation(
                    component=f"{name}.{col}",
                    current_mb=col_mb,
                    potential_savings_mb=col_mb * 0.5,
                    recommendation="Downcast float64 to float32",
                    priority="MEDIUM" if col_mb > 1 else "LOW",
                ))
            elif col_type == object:
                num_unique = df[col].nunique()
                if num_unique < 100:
                    recs.append(OptimizationRecommendation(
                        component=f"{name}.{col}",
                        current_mb=col_mb,
                        potential_savings_mb=col_mb * 0.7,
                        recommendation=f"Convert to category ({num_unique} unique values)",
                        priority="HIGH" if col_mb > 5 else "MEDIUM",
                    ))
        return recs


# ---------------------------------------------------------------------------
# Process Memory Tracker
# ---------------------------------------------------------------------------
class ProcessMemoryTracker:
    """Track process memory over time."""

    def __init__(self, max_history: int = 1000):
        self._history: list[MemorySnapshot] = []
        self.max_history = max_history

    def take_snapshot(self) -> MemorySnapshot:
        snap = MemorySnapshot(timestamp=datetime.now().isoformat())

        # RSS and VMS
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            snap.rss_mb = usage.ru_maxrss / 1024  # Linux reports in KB
        except (ImportError, AttributeError):
            pass

        # Python objects
        snap.python_objects = len(gc.get_objects())

        # GC generations
        gc_stats = gc.get_stats()
        if len(gc_stats) >= 3:
            snap.gc_generation_0 = gc_stats[0].get("collections", 0)
            snap.gc_generation_1 = gc_stats[1].get("collections", 0)
            snap.gc_generation_2 = gc_stats[2].get("collections", 0)

        # DataFrame count
        df_count = 0
        df_total_bytes = 0
        for obj in gc.get_objects():
            if isinstance(obj, pd.DataFrame):
                df_count += 1
                try:
                    df_total_bytes += obj.memory_usage(deep=True).sum()
                except Exception:
                    pass
        snap.dataframe_count = df_count
        snap.dataframe_mb = df_total_bytes / (1024 * 1024)

        # GPU memory
        snap.gpu_mb = self._get_gpu_memory()

        self._history.append(snap)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

        return snap

    def _get_gpu_memory(self) -> float:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0])
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return 0.0

    def get_growth_rate(self, window: int = 10) -> float:
        """Compute memory growth rate in MB/hour."""
        if len(self._history) < window:
            return 0.0

        recent = self._history[-window:]
        if len(recent) < 2:
            return 0.0

        try:
            t0 = datetime.fromisoformat(recent[0].timestamp)
            t1 = datetime.fromisoformat(recent[-1].timestamp)
            hours = (t1 - t0).total_seconds() / 3600
            if hours <= 0:
                return 0.0
            delta_mb = recent[-1].rss_mb - recent[0].rss_mb
            return delta_mb / hours
        except (ValueError, TypeError):
            return 0.0

    def detect_leak(self, threshold_mb_per_hour: float = 100.0) -> bool:
        rate = self.get_growth_rate()
        return rate > threshold_mb_per_hour

    def get_history(self) -> list[MemorySnapshot]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Garbage Collection Manager
# ---------------------------------------------------------------------------
class GCManager:
    """Managed garbage collection with scheduling."""

    def __init__(self, generation_thresholds: tuple = (700, 10, 10)):
        self._last_gc_time: float = time.time()
        self._gc_count: int = 0
        self._original_thresholds = gc.get_threshold()
        gc.set_threshold(*generation_thresholds)

    def force_collect(self) -> dict:
        """Force full garbage collection."""
        before = len(gc.get_objects())
        collected = gc.collect()
        after = len(gc.get_objects())
        self._gc_count += 1
        self._last_gc_time = time.time()

        return {
            "collected": collected,
            "objects_before": before,
            "objects_after": after,
            "freed": before - after,
            "timestamp": datetime.now().isoformat(),
        }

    def periodic_collect(self, interval_seconds: float = 300.0) -> Optional[dict]:
        """Collect if enough time has passed."""
        if time.time() - self._last_gc_time >= interval_seconds:
            return self.force_collect()
        return None

    def get_stats(self) -> dict:
        return {
            "gc_count": self._gc_count,
            "last_gc": datetime.fromtimestamp(self._last_gc_time).isoformat(),
            "thresholds": gc.get_threshold(),
            "is_enabled": gc.isenabled(),
            "objects": len(gc.get_objects()),
        }


# ---------------------------------------------------------------------------
# Memory Budget Enforcer
# ---------------------------------------------------------------------------
class MemoryBudgetEnforcer:
    """Enforce memory budgets across components."""

    def __init__(self, budget: Optional[MemoryBudget] = None):
        self.budget = budget or MemoryBudget()
        self._allocations: dict[str, float] = {}

    def register(self, component: str, allocated_mb: float):
        self._allocations[component] = allocated_mb

    def check_budget(self) -> dict:
        total_used = sum(self._allocations.values())
        utilization = total_used / self.budget.total_budget_mb if self.budget.total_budget_mb > 0 else 0

        status = "OK"
        if utilization > self.budget.critical_threshold:
            status = "CRITICAL"
        elif utilization > self.budget.warning_threshold:
            status = "WARNING"

        return {
            "total_budget_mb": self.budget.total_budget_mb,
            "total_used_mb": round(total_used, 2),
            "utilization": round(utilization, 3),
            "status": status,
            "components": dict(self._allocations),
            "remaining_mb": round(self.budget.total_budget_mb - total_used, 2),
        }

    def can_allocate(self, mb: float) -> bool:
        total_used = sum(self._allocations.values())
        return (total_used + mb) <= self.budget.total_budget_mb * self.budget.critical_threshold


# ---------------------------------------------------------------------------
# Memory Monitor
# ---------------------------------------------------------------------------
class MemoryMonitor:
    """Master memory monitoring system."""

    def __init__(self, budget: Optional[MemoryBudget] = None):
        self._tracker = ProcessMemoryTracker()
        self._gc = GCManager()
        self._cache = LRUCacheManager()
        self._df_analyzer = DataFrameMemoryAnalyzer()
        self._enforcer = MemoryBudgetEnforcer(budget)
        self._component_memory: dict[str, ComponentMemory] = {}

    def take_snapshot(self) -> MemorySnapshot:
        return self._tracker.take_snapshot()

    def register_component(self, name: str, size_mb: float, object_count: int = 0):
        self._component_memory[name] = ComponentMemory(
            name=name, size_mb=size_mb, object_count=object_count,
        )
        self._enforcer.register(name, size_mb)

    def check_health(self) -> dict:
        snapshot = self.take_snapshot()
        budget = self._enforcer.check_budget()
        leak_detected = self._tracker.detect_leak()
        growth_rate = self._tracker.get_growth_rate()

        status = "HEALTHY"
        issues = []

        if budget["status"] == "CRITICAL":
            status = "CRITICAL"
            issues.append("Memory budget exceeded critical threshold")
        elif budget["status"] == "WARNING":
            status = "WARNING"
            issues.append("Memory usage approaching budget limit")

        if leak_detected:
            status = "WARNING" if status == "HEALTHY" else status
            issues.append(f"Potential memory leak detected ({growth_rate:.1f} MB/hr)")

        return {
            "status": status,
            "snapshot": {
                "rss_mb": snapshot.rss_mb,
                "python_objects": snapshot.python_objects,
                "dataframe_count": snapshot.dataframe_count,
                "dataframe_mb": round(snapshot.dataframe_mb, 2),
                "gpu_mb": snapshot.gpu_mb,
            },
            "budget": budget,
            "growth_rate_mb_hr": round(growth_rate, 2),
            "leak_detected": leak_detected,
            "issues": issues,
            "cache": self._cache.get_stats(),
            "gc": self._gc.get_stats(),
        }

    def optimize(self) -> dict:
        """Run optimization cycle."""
        gc_result = self._gc.force_collect()

        # Trim cache if over budget
        cache_stats = self._cache.get_stats()
        trimmed = 0
        while self._cache.total_memory_mb > self._cache.max_memory_mb * 0.9:
            if self._cache.size == 0:
                break
            key = next(iter(self._cache._cache))
            self._cache.evict(key)
            trimmed += 1

        return {
            "gc": gc_result,
            "cache_trimmed": trimmed,
            "cache_after": self._cache.get_stats(),
        }

    def get_cache(self) -> LRUCacheManager:
        return self._cache

    def print_report(self) -> str:
        health = self.check_health()
        lines = [
            "=" * 60,
            "MEMORY MONITOR REPORT",
            "=" * 60,
            f"  Status: {health['status']}",
            f"  RSS: {health['snapshot']['rss_mb']:.1f} MB",
            f"  Python Objects: {health['snapshot']['python_objects']:,}",
            f"  DataFrames: {health['snapshot']['dataframe_count']} ({health['snapshot']['dataframe_mb']:.1f} MB)",
            f"  GPU: {health['snapshot']['gpu_mb']:.1f} MB",
            "",
            "  Budget:",
            f"    Used: {health['budget']['total_used_mb']:.1f} / {health['budget']['total_budget_mb']:.1f} MB",
            f"    Utilization: {health['budget']['utilization']:.1%}",
            f"    Status: {health['budget']['status']}",
            "",
            f"  Growth Rate: {health['growth_rate_mb_hr']:.1f} MB/hr",
            f"  Leak Detected: {health['leak_detected']}",
            "",
            "  Cache:",
            f"    Size: {health['cache']['size']} / {health['cache']['max_size']}",
            f"    Memory: {health['cache']['memory_mb']:.1f} MB",
            f"    Hit Rate: {health['cache']['hit_rate']:.1%}",
        ]

        if health["issues"]:
            lines.append("")
            lines.append("  Issues:")
            for issue in health["issues"]:
                lines.append(f"    ! {issue}")

        lines.append("=" * 60)
        return "\n".join(lines)
