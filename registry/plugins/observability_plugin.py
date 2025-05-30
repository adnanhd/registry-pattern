# registry/plugins/observability_plugin.py
"""Observability plugin for enhanced logging and monitoring."""

try:
    import structlog
    from opentelemetry import trace, metrics
    from prometheus_client import Counter, Histogram, Gauge
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    structlog = None

import time
import logging
from typing import Type, Any, Dict, Optional, List
from .base import BaseRegistryPlugin

class ObservabilityPlugin(BaseRegistryPlugin):
    """Plugin for enhanced observability."""
    
    def _on_cleanup(self) -> None:
        """Cleanup observability resources."""
        # Clear metrics registry
        if hasattr(self, 'metrics_registry'):
            self.metrics_registry._collector_to_names.clear()
            self.metrics_registry._names_to_collectors.clear()
    """Plugin for enhanced observability."""
    
    @property
    def name(self) -> str:
        return "observability"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def dependencies(self) -> List[str]:
        return ["structlog", "opentelemetry", "prometheus_client"]
    
    def _on_initialize(self) -> None:
        if not OBSERVABILITY_AVAILABLE:
            raise ImportError("Observability dependencies required. Install with: pip install registry[observability-plugin]")
        
        self._setup_logging()
        self._setup_metrics()
        if self.registry_class:
            self._add_observability_methods()
    
    def _setup_logging(self) -> None:
        """Setup structured logging."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger("registry")
    
    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics."""
        from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge
        
        # Create a custom registry to avoid conflicts
        self.metrics_registry = CollectorRegistry()
        
        # Create metrics with unique names based on plugin instance
        plugin_id = id(self)
        
        self.registration_counter = Counter(
            f'registry_registrations_total_{plugin_id}',
            'Total number of registry registrations',
            ['registry_name', 'status'],
            registry=self.metrics_registry
        )
        
        self.validation_duration = Histogram(
            f'registry_validation_duration_seconds_{plugin_id}',
            'Time spent on validation',
            ['registry_name', 'validation_type'],
            registry=self.metrics_registry
        )
        
        self.registry_size = Gauge(
            f'registry_size_{plugin_id}',
            'Current size of registry',
            ['registry_name'],
            registry=self.metrics_registry
        )
    
    def _add_observability_methods(self) -> None:
        """Add observability methods to the registry."""
        
        original_register = self.registry_class.register_artifact
        original_unregister = self.registry_class.unregister_artifact
        
        @classmethod
        def register_artifact_with_observability(cls, *args, **kwargs):
            """Wrapped register_artifact with observability."""
            start_time = time.time()
            registry_name = cls.__name__
            
            try:
                result = original_register(*args, **kwargs)
                self.registration_counter.labels(
                    registry_name=registry_name, 
                    status='success'
                ).inc()
                self.logger.info(
                    "Artifact registered successfully",
                    registry_name=registry_name,
                    artifact_count=cls._len_mapping()
                )
                return result
            except Exception as e:
                self.registration_counter.labels(
                    registry_name=registry_name, 
                    status='error'
                ).inc()
                self.logger.error(
                    "Artifact registration failed",
                    registry_name=registry_name,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
            finally:
                duration = time.time() - start_time
                self.validation_duration.labels(
                    registry_name=registry_name,
                    validation_type='registration'
                ).observe(duration)
                self.registry_size.labels(registry_name=registry_name).set(cls._len_mapping())
        
        @classmethod
        def unregister_artifact_with_observability(cls, *args, **kwargs):
            """Wrapped unregister_artifact with observability."""
            registry_name = cls.__name__
            
            try:
                result = original_unregister(*args, **kwargs)
                self.logger.info(
                    "Artifact unregistered successfully",
                    registry_name=registry_name,
                    artifact_count=cls._len_mapping()
                )
                return result
            except Exception as e:
                self.logger.error(
                    "Artifact unregistration failed",
                    registry_name=registry_name,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
            finally:
                self.registry_size.labels(registry_name=registry_name).set(cls._len_mapping())
        
        # Replace methods
        setattr(self.registry_class, 'register_artifact', register_artifact_with_observability)
        setattr(self.registry_class, 'unregister_artifact', unregister_artifact_with_observability)
