# service/service_handling.py

"""
Production-Grade Error Handling & Recovery System
Für MLX Vector Database auf Apple Silicon

Features:
- Graceful Degradation bei Memory-Problemen
- Automatic Recovery von korrupten Stores  
- Circuit Breaker Pattern für Robustheit
- Detailliertes Error Logging mit Context
- Health Check Integration
- Retry Logic mit Exponential Backoff
"""

import mlx.core as mx
import numpy as np
import logging
import time
import traceback
import threading
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import functools
import inspect
from contextlib import contextmanager
import psutil
import signal
import os

# =================== ERROR CLASSIFICATION ===================

class ErrorSeverity(Enum):
    """Error Severity Levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error Categories for Better Handling"""
    MEMORY = "memory"
    STORAGE = "storage"
    COMPUTATION = "computation"
    VALIDATION = "validation"
    NETWORK = "network"
    TIMEOUT = "timeout"
    CORRUPTION = "corruption"
    RESOURCE = "resource"


@dataclass
class ErrorContext:
    """Rich Error Context for Better Debugging"""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    user_id: Optional[str] = None
    model_id: Optional[str] = None
    vector_count: Optional[int] = None
    memory_usage_gb: Optional[float] = None
    stack_trace: Optional[str] = None
    system_info: Optional[Dict] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    additional_context: Dict[str, Any] = field(default_factory=dict)


class MLXVectorDBError(Exception):
    """Base Exception for MLX Vector Database"""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, 
                 cause: Optional[Exception] = None):
        super().__init__(message)
        self.context = context
        self.cause = cause
        self.timestamp = time.time()


class MemoryPressureError(MLXVectorDBError):
    """Memory pressure too high for operation"""
    pass


class StorageCorruptionError(MLXVectorDBError):
    """Storage corruption detected"""
    pass


class ComputationError(MLXVectorDBError):
    """MLX computation failed"""
    pass


class ValidationError(MLXVectorDBError):
    """Data validation failed"""
    pass


# =================== CIRCUIT BREAKER PATTERN ===================

class CircuitBreakerState(Enum):
    """Circuit Breaker States"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit Breaker Configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3
    timeout_duration: float = 10.0


class CircuitBreaker:
    """Circuit Breaker for Protecting System Components"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for protecting functions"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise MLXVectorDBError(
                        f"Circuit breaker {self.name} is OPEN",
                        context=ErrorContext(
                            timestamp=time.time(),
                            error_type="CircuitBreakerOpen",
                            error_message=f"Circuit breaker {self.name} is protecting system",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.RESOURCE,
                            operation=func.__name__
                        )
                    )
        
        start_time = time.time()
        try:
            result = self._execute_with_timeout(func, *args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._on_failure(e, execution_time)
            raise
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout protection"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.config.timeout_duration)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise MLXVectorDBError(
                    f"Operation {func.__name__} timed out after {self.config.timeout_duration}s",
                    context=ErrorContext(
                        timestamp=time.time(),
                        error_type="OperationTimeout",
                        error_message=f"Function {func.__name__} exceeded timeout",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.TIMEOUT,
                        operation=func.__name__
                    )
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        return time.time() - self._last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful execution"""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self.logger.info(f"Circuit breaker {self.name} reset to CLOSED state")
            elif self.state == CircuitBreakerState.CLOSED:
                self._failure_count = 0
    
    def _on_failure(self, error: Exception, execution_time: float) -> None:
        """Handle failed execution"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} failed in HALF_OPEN, returning to OPEN")
            
            elif (self.state == CircuitBreakerState.CLOSED and 
                  self._failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                self.logger.error(f"Circuit breaker {self.name} opened after {self._failure_count} failures")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'last_failure_time': self._last_failure_time
            }

# =================== RETRY LOGIC WITH EXPONENTIAL BACKOFF ===================

@dataclass
class RetryConfig:
    """Retry Configuration"""
    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True


class RetryHandler:
    """Intelligent Retry Handler with Exponential Backoff"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger("retry_handler")
    
    def __call__(self, retryable_exceptions: tuple = (Exception,)):
        """Decorator for adding retry logic"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_retry(func, retryable_exceptions, *args, **kwargs)
            return wrapper
        return decorator
    
    def execute_with_retry(self, func: Callable, retryable_exceptions: tuple, 
                          *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
                
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}, retrying in {delay:.2f}s: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Function {func.__name__} failed after {self.config.max_attempts} attempts")
            
            except Exception as e:
                self.logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                raise
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(self.config.base_delay * (self.config.exponential_base ** attempt), self.config.max_delay)
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        return delay

# =================== GRACEFUL DEGRADATION MANAGER ===================

class DegradationLevel(Enum):
    """System Degradation Levels"""
    NORMAL = "normal"
    REDUCED = "reduced"
    ESSENTIAL = "essential"
    EMERGENCY = "emergency"


@dataclass
class DegradationPolicy:
    """Degradation Policy Configuration"""
    memory_thresholds: Dict[DegradationLevel, float] = field(default_factory=lambda: {
        DegradationLevel.NORMAL: 0.7, DegradationLevel.REDUCED: 0.8,
        DegradationLevel.ESSENTIAL: 0.9, DegradationLevel.EMERGENCY: 0.95
    })
    error_rate_thresholds: Dict[DegradationLevel, float] = field(default_factory=lambda: {
        DegradationLevel.NORMAL: 0.01, DegradationLevel.REDUCED: 0.05,
        DegradationLevel.ESSENTIAL: 0.15, DegradationLevel.EMERGENCY: 0.3
    })


class GracefulDegradationManager:
    """Manages System Degradation Based on Health Metrics"""
    
    def __init__(self, policy: DegradationPolicy):
        self.policy = policy
        self.current_level = DegradationLevel.NORMAL
        self._lock = threading.RLock()
        self._degradation_history: List[Dict] = []
        self.logger = logging.getLogger("degradation_manager")
    
    def assess_system_health(self, memory_usage: float, error_rate: float, 
                           additional_metrics: Optional[Dict] = None) -> DegradationLevel:
        """Assess system health and determine degradation level"""
        memory_level = DegradationLevel.EMERGENCY
        for level, threshold in self.policy.memory_thresholds.items():
            if memory_usage <= threshold:
                memory_level = level
                break
        
        error_level = DegradationLevel.EMERGENCY
        for level, threshold in self.policy.error_rate_thresholds.items():
            if error_rate <= threshold:
                error_level = level
                break
        
        new_level = max(memory_level, error_level, key=lambda x: list(DegradationLevel).index(x))
        
        with self._lock:
            if new_level != self.current_level:
                self._record_degradation_change(self.current_level, new_level, {
                    'memory_usage': memory_usage, 'error_rate': error_rate,
                    'additional_metrics': additional_metrics or {}
                })
                self.current_level = new_level
        
        return new_level
    
    def _record_degradation_change(self, old_level: DegradationLevel, new_level: DegradationLevel, context: Dict) -> None:
        """Record degradation level change"""
        change_record = {'timestamp': time.time(), 'old_level': old_level.value, 'new_level': new_level.value, 'context': context}
        self._degradation_history.append(change_record)
        if len(self._degradation_history) > 100:
            self._degradation_history.pop(0)
        
        if list(DegradationLevel).index(new_level) > list(DegradationLevel).index(old_level):
            self.logger.warning(f"System degradation increased: {old_level.value} -> {new_level.value}")
        else:
            self.logger.info(f"System degradation decreased: {old_level.value} -> {new_level.value}")
    
    def get_available_features(self) -> Dict[str, bool]:
        """Get list of available features at current degradation level"""
        feature_map = {
            DegradationLevel.NORMAL: {'vector_caching': True, 'batch_operations': True, 'hnsw_indexing': True, 'advanced_metrics': True},
            DegradationLevel.REDUCED: {'vector_caching': True, 'batch_operations': True, 'hnsw_indexing': False, 'advanced_metrics': False},
            DegradationLevel.ESSENTIAL: {'vector_caching': False, 'batch_operations': False, 'hnsw_indexing': False, 'advanced_metrics': False},
            DegradationLevel.EMERGENCY: {'vector_caching': False, 'batch_operations': False, 'hnsw_indexing': False, 'advanced_metrics': False}
        }
        return feature_map.get(self.current_level, feature_map[DegradationLevel.EMERGENCY])
    
    def get_degradation_stats(self) -> Dict[str, Any]:
        """Get degradation statistics"""
        with self._lock:
            return {
                'current_level': self.current_level.value,
                'available_features': self.get_available_features(),
                'degradation_history': self._degradation_history[-10:]
            }

# =================== COMPREHENSIVE ERROR HANDLER ===================

class MLXErrorHandler:
    """Comprehensive Error Handler for MLX Vector Database"""
    
    def __init__(self):
        self.logger = logging.getLogger("mlx_error_handler")
        self._error_history: List[ErrorContext] = []
        self._error_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
        
        self.circuit_breakers = {
            'vector_add': CircuitBreaker('vector_add', CircuitBreakerConfig()),
            'vector_query': CircuitBreaker('vector_query', CircuitBreakerConfig()),
            'storage_io': CircuitBreaker('storage_io', CircuitBreakerConfig(failure_threshold=3)),
            'mlx_computation': CircuitBreaker('mlx_computation', CircuitBreakerConfig())
        }
        
        self.retry_handlers = {
            'storage': RetryHandler(RetryConfig(max_attempts=3, base_delay=0.1)),
            'computation': RetryHandler(RetryConfig(max_attempts=2, base_delay=0.05)),
            'network': RetryHandler(RetryConfig(max_attempts=5, base_delay=0.2))
        }
        
        self.degradation_manager = GracefulDegradationManager(DegradationPolicy())
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        # KORREKTUR: Prüfen, ob wir im Haupt-Thread sind, bevor Signal-Handler registriert werden.
        if threading.current_thread() is threading.main_thread():
            def signal_handler(signum, frame):
                self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
                self._graceful_shutdown()
            
            try:
                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)
                self.logger.info("Signal handlers for graceful shutdown registered.")
            except (AttributeError, ValueError, OSError) as e:
                self.logger.warning(f"Could not set signal handlers: {e}")
        else:
            self.logger.warning("Skipping signal handler setup in non-main thread.")

    def _graceful_shutdown(self) -> None:
        """Perform graceful shutdown"""
        self.logger.info("Performing graceful shutdown...")
        try:
            self._save_error_history()
        except Exception as e:
            self.logger.error(f"Failed to save error history during shutdown: {e}")
        self.logger.info("Graceful shutdown completed")
    
    @contextmanager
    def error_context(self, operation: str, user_id: Optional[str] = None, 
                     model_id: Optional[str] = None, **kwargs):
        """Context manager for error handling with rich context"""
        try:
            yield
        except Exception as e:
            context = ErrorContext(
                timestamp=time.time(), error_type=type(e).__name__, error_message=str(e),
                severity=self._classify_error_severity(e), category=self._classify_error_category(e),
                operation=operation, user_id=user_id, model_id=model_id,
                stack_trace=traceback.format_exc(), system_info=self._get_system_info(),
                additional_context=kwargs
            )
            self._record_error(context)
            recovered = self._attempt_recovery(context, e)
            context.recovery_attempted = True
            context.recovery_successful = recovered
            if not recovered:
                self._update_degradation_level()
                raise MLXVectorDBError(f"Operation '{operation}' failed: {str(e)}", context=context, cause=e)
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and context"""
        if isinstance(error, (MemoryError, MemoryPressureError)): return ErrorSeverity.CRITICAL
        elif isinstance(error, (StorageCorruptionError, ComputationError)): return ErrorSeverity.HIGH
        elif isinstance(error, (ValidationError, FileNotFoundError, TimeoutError, ConnectionError)): return ErrorSeverity.MEDIUM
        else: return ErrorSeverity.LOW
    
    def _classify_error_category(self, error: Exception) -> ErrorCategory:
        """Classify error category based on type"""
        if isinstance(error, (MemoryError, MemoryPressureError)): return ErrorCategory.MEMORY
        elif isinstance(error, (IOError, OSError, FileNotFoundError)): return ErrorCategory.STORAGE
        elif isinstance(error, StorageCorruptionError): return ErrorCategory.CORRUPTION
        elif isinstance(error, (ValidationError, ValueError, TypeError)): return ErrorCategory.VALIDATION
        elif isinstance(error, TimeoutError): return ErrorCategory.TIMEOUT
        elif 'mlx' in str(type(error)).lower(): return ErrorCategory.COMPUTATION
        else: return ErrorCategory.RESOURCE
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            return {
                'memory_percent': memory.percent, 'memory_available_gb': memory.available / (1024**3),
                'process_memory_gb': process.memory_info().rss / (1024**3), 'cpu_percent': psutil.cpu_percent(),
                'open_files': len(process.open_files()), 'thread_count': process.num_threads(),
                'mlx_device': str(mx.default_device()) if 'mx' in globals() else 'unknown'
            }
        except Exception:
            return {'error': 'Failed to collect system info'}
    
    def _record_error(self, context: ErrorContext) -> None:
        """Record error for analysis and monitoring"""
        with self._lock:
            self._error_history.append(context)
            error_key = f"{context.category.value}:{context.error_type}"
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
            if len(self._error_history) > 1000:
                self._error_history.pop(0)
            self.logger.error(f"Error recorded: {context.error_type} in {context.operation} (Severity: {context.severity.value}, Category: {context.category.value})")
    
    def _attempt_recovery(self, context: ErrorContext, error: Exception) -> bool:
        """Attempt automatic recovery based on error type"""
        try:
            if context.category == ErrorCategory.MEMORY: return self._recover_from_memory_error(context, error)
            elif context.category == ErrorCategory.STORAGE: return self._recover_from_storage_error(context, error)
            elif context.category == ErrorCategory.CORRUPTION: return self._recover_from_corruption_error(context, error)
            else: return False
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            return False
    
    def _recover_from_memory_error(self, context: ErrorContext, error: Exception) -> bool:
        """Attempt recovery from memory errors"""
        self.logger.info("Attempting memory error recovery by clearing caches and running GC...")
        import gc; gc.collect()
        return True
    
    def _recover_from_storage_error(self, context: ErrorContext, error: Exception) -> bool:
        """Attempt recovery from storage errors"""
        self.logger.info("Attempting storage error recovery by checking disk space...")
        if psutil.disk_usage('/').free / psutil.disk_usage('/').total < 0.05:
            self.logger.error("Insufficient disk space for recovery")
            return False
        return True
        
    def _recover_from_corruption_error(self, context: ErrorContext, error: Exception) -> bool:
        """Attempt recovery from corruption errors"""
        self.logger.warning("Corruption detected - backup restoration would be triggered. Marking as non-recoverable for now.")
        return False
        
    def _update_degradation_level(self) -> None:
        """Update system degradation level based on recent errors"""
        with self._lock:
            recent_errors = [e for e in self._error_history if time.time() - e.timestamp < 300]
            error_rate = len(recent_errors) / 300
            memory_usage = psutil.virtual_memory().percent / 100
            self.degradation_manager.assess_system_health(memory_usage, error_rate)
    
    def _save_error_history(self) -> None:
        """Save error history to disk for analysis"""
        try:
            error_file = Path("./logs/error_history.json")
            error_file.parent.mkdir(exist_ok=True)
            serializable_errors = [vars(e) for e in self._error_history[-100:]] # Simplified serialization
            with open(error_file, 'w') as f:
                json.dump(serializable_errors, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save error history: {e}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        with self._lock:
            recent_errors = [e for e in self._error_history if time.time() - e.timestamp < 3600]
            recovery_attempts = [e for e in recent_errors if e.recovery_attempted]
            recovery_successes = [e for e in recovery_attempts if e.recovery_successful]
            return {
                'total_errors': len(self._error_history),
                'recent_errors_count': len(recent_errors),
                'error_rate_per_second': len(recent_errors) / 3600 if recent_errors else 0,
                'recovery_rate': len(recovery_successes) / len(recovery_attempts) if recovery_attempts else 0,
                'circuit_breaker_stats': {name: cb.get_stats() for name, cb in self.circuit_breakers.items()},
                'degradation_stats': self.degradation_manager.get_degradation_stats(),
                'most_common_errors': sorted(self._error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check including error analysis"""
        stats = self.get_error_statistics()
        health_score = 100
        issues = []
        if stats['error_rate_per_second'] > 0.1: health_score -= 30; issues.append('High error rate')
        if self.degradation_manager.current_level != DegradationLevel.NORMAL: health_score -= 20; issues.append('System degraded')
        for name, cb in self.circuit_breakers.items():
            if cb.state != CircuitBreakerState.CLOSED: health_score -= 15; issues.append(f'Circuit breaker {name} is {cb.state.value}')
        
        health_level = 'good' if health_score >= 70 else 'warning' if health_score >= 50 else 'critical'
        return {'healthy': health_score >= 70, 'health_score': max(0, health_score), 'health_level': health_level, 'issues': issues, 'error_statistics': stats}

# =================== DECORATORS FOR EASY INTEGRATION ===================

def with_error_handling(operation: str, retryable: bool = True):
    """Decorator for adding comprehensive error handling to functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            error_handler = getattr(self, 'error_handler', None)
            if not error_handler: return func(self, *args, **kwargs)
            with error_handler.error_context(operation):
                if retryable:
                    retry_handler = error_handler.retry_handlers['storage']
                    return retry_handler.execute_with_retry(func, (IOError, ComputationError), self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)
        return wrapper
    return decorator

def with_circuit_breaker(breaker_name: str):
    """Decorator for adding circuit breaker protection"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            error_handler = getattr(self, 'error_handler', None)
            if not error_handler: return func(self, *args, **kwargs)
            circuit_breaker = error_handler.circuit_breakers.get(breaker_name)
            if circuit_breaker: return circuit_breaker.execute(func, self, *args, **kwargs)
            else: return func(self, *args, **kwargs)
        return wrapper