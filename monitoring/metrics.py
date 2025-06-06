## monitoring/metrics.py
"""
Comprehensive monitoring and metrics for MLX Vector Database
Provides Prometheus-compatible metrics and health monitoring
"""
import time
import threading
import psutil
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
from pathlib import Path

logger = logging.getLogger("mlx_vector_db.monitoring")

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

class TimeSeriesMetric:
    """Time series metric storage"""
    
    def __init__(self, name: str, help_text: str, max_points: int = 1000):
        self.name = name
        self.help_text = help_text
        self.max_points = max_points
        self.points: deque = deque(maxlen=max_points)
        self.lock = threading.Lock()
    
    def record(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        with self.lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            self.points.append(point)
    
    def get_current_value(self) -> Optional[float]:
        """Get the most recent value"""
        with self.lock:
            return self.points[-1].value if self.points else None
    
    def get_points(self, since: Optional[float] = None) -> List[MetricPoint]:
        """Get metric points since timestamp"""
        with self.lock:
            if since is None:
                return list(self.points)
            return [p for p in self.points if p.timestamp >= since]

class Counter(TimeSeriesMetric):
    """Counter metric (monotonically increasing)"""
    
    def __init__(self, name: str, help_text: str):
        super().__init__(name, help_text)
        self.value = 0.0
        self.value_lock = threading.Lock()
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter"""
        with self.value_lock:
            self.value += amount
            self.record(self.value, labels)

class Gauge(TimeSeriesMetric):
    """Gauge metric (can go up and down)"""
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value"""
        self.record(value, labels)

class Histogram(TimeSeriesMetric):
    """Histogram metric for measuring distributions"""
    
    def __init__(self, name: str, help_text: str, buckets: List[float] = None):
        super().__init__(name, help_text)
        self.buckets = buckets or [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.bucket_counts = defaultdict(int)
        self.sum = 0.0
        self.count = 0
        self.lock = threading.Lock()
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value"""
        with self.lock:
            self.sum += value
            self.count += 1
            
            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[bucket] += 1
            
            self.record(value, labels)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get histogram statistics"""
        with self.lock:
            return {
                "count": self.count,
                "sum": self.sum,
                "mean": self.sum / self.count if self.count > 0 else 0,
                "buckets": dict(self.bucket_counts)
            }

class MetricsRegistry:
    """Central registry for all metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, TimeSeriesMetric] = {}
        self.lock = threading.Lock()
        
        # Initialize core metrics
        self._init_core_metrics()
        
        # Start system metrics collection
        self._system_metrics_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self._system_metrics_thread.start()
    
    def _init_core_metrics(self):
        """Initialize core application metrics"""
        # Request metrics
        self.request_total = self.counter("http_requests_total", "Total HTTP requests")
        self.request_duration = self.histogram("http_request_duration_seconds", "HTTP request duration")
        
        # Vector operations metrics
        self.vector_queries_total = self.counter("vector_queries_total", "Total vector queries")
        self.vector_query_duration = self.histogram("vector_query_duration_seconds", "Vector query duration")
        self.vector_additions_total = self.counter("vector_additions_total", "Total vector additions")
        self.vector_store_size = self.gauge("vector_store_size", "Number of vectors in stores")
        
        # Cache metrics
        self.cache_hits_total = self.counter("cache_hits_total", "Total cache hits")
        self.cache_misses_total = self.counter("cache_misses_total", "Total cache misses")
        self.cache_memory_usage = self.gauge("cache_memory_usage_bytes", "Cache memory usage")
        
        # HNSW index metrics
        self.hnsw_build_duration = self.histogram("hnsw_build_duration_seconds", "HNSW index build time")
        self.hnsw_search_duration = self.histogram("hnsw_search_duration_seconds", "HNSW search time")
        self.hnsw_index_size = self.gauge("hnsw_index_size", "HNSW index node count")
        
        # System metrics
        self.system_cpu_usage = self.gauge("system_cpu_usage_percent", "System CPU usage")
        self.system_memory_usage = self.gauge("system_memory_usage_bytes", "System memory usage")
        self.system_disk_usage = self.gauge("system_disk_usage_bytes", "System disk usage")
        
        # Error metrics
        self.errors_total = self.counter("errors_total", "Total errors by type")
    
    def counter(self, name: str, help_text: str) -> Counter:
        """Create or get a counter metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Counter(name, help_text)
            return self.metrics[name]
    
    def gauge(self, name: str, help_text: str) -> Gauge:
        """Create or get a gauge metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Gauge(name, help_text)
            return self.metrics[name]
    
    def histogram(self, name: str, help_text: str, buckets: List[float] = None) -> Histogram:
        """Create or get a histogram metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Histogram(name, help_text, buckets)
            return self.metrics[name]
    
    def _collect_system_metrics(self):
        """Collect system metrics in background thread"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.system_memory_usage.set(memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.system_disk_usage.set(disk.used)
                
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(30)  # Wait longer on error
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values"""
        with self.lock:
            result = {}
            for name, metric in self.metrics.items():
                current_value = metric.get_current_value()
                result[name] = {
                    "value": current_value,
                    "help": metric.help_text,
                    "type": type(metric).__name__.lower()
                }
                
                # Add histogram stats if applicable
                if isinstance(metric, Histogram):
                    result[name]["stats"] = metric.get_stats()
            
            return result
    
    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format"""
        lines = []
        
        with self.lock:
            for name, metric in self.metrics.items():
                # Add help text
                lines.append(f"# HELP {name} {metric.help_text}")
                lines.append(f"# TYPE {name} {type(metric).__name__.lower()}")
                
                current_value = metric.get_current_value()
                if current_value is not None:
                    lines.append(f"{name} {current_value}")
                
                # Add histogram buckets if applicable
                if isinstance(metric, Histogram):
                    stats = metric.get_stats()
                    for bucket, count in stats["buckets"].items():
                        lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
                    lines.append(f"{name}_sum {stats['sum']}")
                    lines.append(f"{name}_count {stats['count']}")
                
                lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)

# Global metrics registry
metrics_registry = MetricsRegistry()

# Convenience functions for common metrics
def record_request(duration: float, status_code: int, endpoint: str):
    """Record HTTP request metrics"""
    labels = {"status_code": str(status_code), "endpoint": endpoint}
    metrics_registry.request_total.inc(labels=labels)
    metrics_registry.request_duration.observe(duration, labels)

def record_vector_query(duration: float, vector_count: int, k: int, user_id: str):
    """Record vector query metrics"""
    labels = {"user_id": user_id, "k": str(k)}
    metrics_registry.vector_queries_total.inc(labels=labels)
    metrics_registry.vector_query_duration.observe(duration, labels)

def record_vector_addition(vector_count: int, user_id: str):
    """Record vector addition metrics"""
    labels = {"user_id": user_id}
    metrics_registry.vector_additions_total.inc(vector_count, labels)

def record_cache_hit(user_id: str, model_id: str):
    """Record cache hit"""
    labels = {"user_id": user_id, "model_id": model_id}
    metrics_registry.cache_hits_total.inc(labels=labels)

def record_cache_miss(user_id: str, model_id: str):
    """Record cache miss"""
    labels = {"user_id": user_id, "model_id": model_id}
    metrics_registry.cache_misses_total.inc(labels=labels)

def record_hnsw_build(duration: float, node_count: int, user_id: str):
    """Record HNSW index build metrics"""
    labels = {"user_id": user_id}
    metrics_registry.hnsw_build_duration.observe(duration, labels)
    metrics_registry.hnsw_index_size.set(node_count, labels)

def record_error(error_type: str, endpoint: str):
    """Record error metrics"""
    labels = {"error_type": error_type, "endpoint": endpoint}
    metrics_registry.errors_total.inc(labels=labels)

# Health check functionality
class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.checks = {}
        self.register_default_checks()
    
    def register_check(self, name: str, check_func, critical: bool = False):
        """Register a health check function"""
        self.checks[name] = {
            "func": check_func,
            "critical": critical,
            "last_result": None,
            "last_check": None
        }
    
    def register_default_checks(self):
        """Register default health checks"""
        self.register_check("mlx_available", self._check_mlx, critical=True)
        self.register_check("disk_space", self._check_disk_space, critical=True)
        self.register_check("memory_usage", self._check_memory_usage, critical=False)
        self.register_check("cache_health", self._check_cache_health, critical=False)
    
    def _check_mlx(self) -> Dict[str, Any]:
        """Check MLX functionality"""
        try:
            import mlx.core as mx
            test_array = mx.array([1, 2, 3])
            mx.eval(test_array)
            return {"status": "healthy", "message": "MLX operations working"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"MLX error: {e}"}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            disk = psutil.disk_usage('/')
            free_percent = (disk.free / disk.total) * 100
            
            if free_percent < 5:
                return {"status": "unhealthy", "message": f"Low disk space: {free_percent:.1f}% free"}
            elif free_percent < 10:
                return {"status": "warning", "message": f"Disk space low: {free_percent:.1f}% free"}
            else:
                return {"status": "healthy", "message": f"Disk space OK: {free_percent:.1f}% free"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Disk check error: {e}"}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            
            if used_percent > 95:
                return {"status": "unhealthy", "message": f"High memory usage: {used_percent:.1f}%"}
            elif used_percent > 85:
                return {"status": "warning", "message": f"Memory usage high: {used_percent:.1f}%"}
            else:
                return {"status": "healthy", "message": f"Memory usage OK: {used_percent:.1f}%"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Memory check error: {e}"}
    
    def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache health"""
        try:
            from performance.vector_cache import get_global_cache
            cache = get_global_cache()
            stats = cache.get_stats()
            
            hit_rate = stats.get("hit_rate_percent", 0)
            if hit_rate < 20:
                return {"status": "warning", "message": f"Low cache hit rate: {hit_rate:.1f}%"}
            else:
                return {"status": "healthy", "message": f"Cache hit rate: {hit_rate:.1f}%"}
        except Exception as e:
            return {"status": "warning", "message": f"Cache check error: {e}"}
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_status = "healthy"
        
        for name, check_info in self.checks.items():
            try:
                result = check_info["func"]()
                check_info["last_result"] = result
                check_info["last_check"] = time.time()
                results[name] = result
                
                # Update overall status
                if result["status"] == "unhealthy" and check_info["critical"]:
                    overall_status = "unhealthy"
                elif result["status"] in ["unhealthy", "warning"] and overall_status == "healthy":
                    overall_status = "warning"
                    
            except Exception as e:
                error_result = {"status": "error", "message": f"Check failed: {e}"}
                results[name] = error_result
                if check_info["critical"]:
                    overall_status = "unhealthy"
        
        return {
            "overall_status": overall_status,
            "checks": results,
            "timestamp": time.time()
        }

# Global health checker
health_checker = HealthChecker()