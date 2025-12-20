"""
Observability and monitoring for production RAG system.

Provides:
- LangSmith tracing integration
- Structured logging
- Metrics collection
- Performance monitoring
"""

import logging
import time
from typing import Optional, Any, Dict
from functools import wraps
from contextlib import contextmanager
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MetricsCollector:
    """Simple metrics collector for monitoring."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {
            "requests_total": 0,
            "requests_by_tenant": {},
            "errors_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time_ms": 0.0,
            "llm_calls": 0,
            "embeddings_generated": 0,
            "documents_indexed": 0,
            "retrieval_calls": 0,
        }
        self.request_times = []
    
    def increment(self, metric: str, value: int = 1, tenant_id: Optional[str] = None):
        """Increment a counter metric."""
        if metric in self.metrics:
            if isinstance(self.metrics[metric], dict) and tenant_id:
                self.metrics[metric][tenant_id] = self.metrics[metric].get(tenant_id, 0) + value
            else:
                self.metrics[metric] += value
    
    def record_response_time(self, duration_ms: float):
        """Record request response time."""
        self.request_times.append(duration_ms)
        
        # Keep only last 1000 measurements
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
        
        # Update average
        if self.request_times:
            self.metrics["avg_response_time_ms"] = sum(self.request_times) / len(self.request_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def reset(self):
        """Reset all metrics."""
        self.__init__()


class StructuredLogger:
    """Structured logger for production logging."""
    
    def __init__(self, name: str):
        """Initialize logger."""
        self.logger = logging.getLogger(name)
    
    def log(
        self,
        level: str,
        message: str,
        **kwargs
    ):
        """Log with structured data."""
        log_data = {
            "message": message,
            **kwargs
        }
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(json.dumps(log_data))
    
    def info(self, message: str, **kwargs):
        """Log info level."""
        self.log("info", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level."""
        self.log("error", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level."""
        self.log("warning", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level."""
        self.log("debug", message, **kwargs)


class LangSmithTracer:
    """
    LangSmith tracing integration.
    
    In production, configure with actual LangSmith credentials.
    """
    
    def __init__(self, enabled: bool = False):
        """
        Initialize tracer.
        
        Args:
            enabled: Whether tracing is enabled
        """
        self.enabled = enabled
        self.logger = StructuredLogger("langsmith")
    
    @contextmanager
    def trace_chain(self, name: str, inputs: Dict[str, Any]):
        """
        Trace a chain execution.
        
        Args:
            name: Chain name
            inputs: Chain inputs
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        
        self.logger.info(
            f"Chain started: {name}",
            chain_name=name,
            inputs=inputs,
        )
        
        try:
            yield
            duration = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"Chain completed: {name}",
                chain_name=name,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            self.logger.error(
                f"Chain failed: {name}",
                chain_name=name,
                duration_ms=duration,
                error=str(e),
            )
            raise
    
    @contextmanager
    def trace_llm(self, model: str, prompt: str):
        """
        Trace an LLM call.
        
        Args:
            model: Model name
            prompt: Prompt text
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        
        self.logger.debug(
            "LLM call started",
            model=model,
            prompt_length=len(prompt),
        )
        
        try:
            yield
            duration = (time.time() - start_time) * 1000
            
            self.logger.info(
                "LLM call completed",
                model=model,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            self.logger.error(
                "LLM call failed",
                model=model,
                duration_ms=duration,
                error=str(e),
            )
            raise


def trace_function(name: Optional[str] = None):
    """
    Decorator to trace function execution.
    
    Args:
        name: Optional custom name for trace
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace_name = name or func.__name__
            logger = StructuredLogger(func.__module__)
            
            start_time = time.time()
            logger.debug(f"Function started: {trace_name}")
            
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                logger.debug(
                    f"Function completed: {trace_name}",
                    duration_ms=duration,
                )
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(
                    f"Function failed: {trace_name}",
                    duration_ms=duration,
                    error=str(e),
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            trace_name = name or func.__name__
            logger = StructuredLogger(func.__module__)
            
            start_time = time.time()
            logger.debug(f"Function started: {trace_name}")
            
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                logger.debug(
                    f"Function completed: {trace_name}",
                    duration_ms=duration,
                )
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(
                    f"Function failed: {trace_name}",
                    duration_ms=duration,
                    error=str(e),
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_langsmith_tracer: Optional[LangSmithTracer] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_tracer(enabled: bool = False) -> LangSmithTracer:
    """
    Get global LangSmith tracer.
    
    Args:
        enabled: Whether to enable tracing
    """
    global _langsmith_tracer
    if _langsmith_tracer is None:
        _langsmith_tracer = LangSmithTracer(enabled=enabled)
    return _langsmith_tracer
