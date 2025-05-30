# registry/core/memory_optimized.py
"""Memory optimizations for registry pattern."""

import gc
import sys
import weakref
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar
from collections import defaultdict

from ..core._validator import ValidationError, get_mem_addr

T = TypeVar("T")

class MemoryOptimizedRegistry:
    """Memory optimization utilities for registries."""
    
    def __init__(self):
        self._object_pool: Dict[str, Any] = {}
        self._weak_refs: Set[weakref.ref] = set()
        self._memory_stats: Dict[str, int] = defaultdict(int)
    
    def intern_object(self, obj: Any) -> Any:
        """
        Intern objects to reduce memory usage for duplicate objects.
        Similar to string interning but for any hashable object.
        """
        try:
            obj_hash = hash(obj)
            obj_key = f"{type(obj).__name__}:{obj_hash}"
            
            if obj_key in self._object_pool:
                # Return existing object
                existing = self._object_pool[obj_key]
                if existing is not None:
                    self._memory_stats['hits'] += 1
                    return existing
            
            # Store new object
            self._object_pool[obj_key] = obj
            self._memory_stats['misses'] += 1
            return obj
            
        except TypeError:
            # Object is not hashable, return as-is
            return obj
    
    def create_weak_ref_with_cleanup(self, obj: Any, callback: Optional[callable] = None) -> weakref.ref:
        """Create a weak reference with automatic cleanup."""
        def cleanup_callback(ref):
            self._weak_refs.discard(ref)
            if callback:
                callback(ref)
        
        ref = weakref.ref(obj, cleanup_callback)
        self._weak_refs.add(ref)
        return ref
    
    def cleanup_dead_refs(self) -> int:
        """Clean up dead weak references."""
        dead_refs = {ref for ref in self._weak_refs if ref() is None}
        self._weak_refs -= dead_refs
        return len(dead_refs)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'object_pool_size': len(self._object_pool),
            'weak_refs_count': len(self._weak_refs),
            'dead_refs_count': sum(1 for ref in self._weak_refs if ref() is None),
            'cache_hits': self._memory_stats['hits'],
            'cache_misses': self._memory_stats['misses'],
            'hit_ratio': (
                self._memory_stats['hits'] / 
                (self._memory_stats['hits'] + self._memory_stats['misses'])
                if (self._memory_stats['hits'] + self._memory_stats['misses']) > 0 else 0
            )
        }
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        # Clean up object pool
        dead_keys = []
        for key, obj in self._object_pool.items():
            if obj is None or (hasattr(obj, '__weakref__') and sys.getrefcount(obj) <= 2):
                dead_keys.append(key)
        
        for key in dead_keys:
            del self._object_pool[key]
        
        # Clean up weak references
        dead_refs = self.cleanup_dead_refs()
        
        # Force garbage collection
        collected = gc.collect()
        
        return {
            'objects_collected': collected,
            'dead_refs_cleaned': dead_refs,
            'pool_objects_removed': len(dead_keys),
        }

class SlotOptimizedMixin:
    """
    Mixin to add __slots__ optimization to registry classes.
    This reduces memory usage by preventing creation of __dict__ for instances.
    """
    __slots__ = ()
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Ensure __slots__ is defined for memory optimization
        if not hasattr(cls, '__slots__'):
            cls.__slots__ = ()

class CompactRegistryMixin:
    """Mixin for compact registry storage."""
    
    @classmethod
    def _compact_storage(cls) -> Dict[str, Any]:
        """
        Compact storage by removing unnecessary metadata and using efficient data structures.
        """
        if not hasattr(cls, '_repository'):
            return {}
        
        # Create compact representation
        compact_data = {}
        for key, value in cls._repository.items():
            # Use shorter keys where possible
            compact_key = get_mem_addr(key, with_prefix=False) if hasattr(key, '__hash__') else str(key)
            compact_data[compact_key] = value
        
        return compact_data
    
    @classmethod
    def _estimate_memory_usage(cls) -> Dict[str, int]:
        """Estimate memory usage of the registry."""
        if not hasattr(cls, '_repository'):
            return {'total_bytes': 0, 'key_bytes': 0, 'value_bytes': 0}
        
        key_bytes = sum(sys.getsizeof(k) for k in cls._repository.keys())
        value_bytes = sum(sys.getsizeof(v) for v in cls._repository.values())
        
        return {
            'total_bytes': key_bytes + value_bytes + sys.getsizeof(cls._repository),
            'key_bytes': key_bytes,
            'value_bytes': value_bytes,
            'container_bytes': sys.getsizeof(cls._repository),
            'item_count': len(cls._repository),
            'avg_key_size': key_bytes / len(cls._repository) if cls._repository else 0,
            'avg_value_size': value_bytes / len(cls._repository) if cls._repository else 0,
        }

class LazyLoadingMixin:
    """Mixin for lazy loading of registry data."""
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._lazy_data = {}
        cls._loaded_keys = set()
    
    @classmethod
    def _lazy_load_value(cls, key: Any) -> Any:
        """Lazy load a value when first accessed."""
        if key in cls._loaded_keys:
            return cls._repository.get(key)
        
        # Load the value (this could be from disk, network, etc.)
        value = cls._load_deferred_value(key)
        if value is not None:
            cls._repository[key] = value
            cls._loaded_keys.add(key)
        
        return value
    
    @classmethod
    def _load_deferred_value(cls, key: Any) -> Any:
        """Override this method to implement actual lazy loading."""
        return cls._lazy_data.get(key)
    
    @classmethod
    def _defer_value_loading(cls, key: Any, loader_func: callable) -> None:
        """Defer loading of a value until first access."""
        cls._lazy_data[key] = loader_func

class MemoryEfficientRegistry(SlotOptimizedMixin, CompactRegistryMixin, LazyLoadingMixin):
    """
    Memory-efficient registry that combines multiple optimization techniques.
    """
    __slots__ = ()
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Initialize class-level memory optimizer
        cls._memory_optimizer = MemoryOptimizedRegistry()
    
    @classmethod
    def _intern_artifact(cls, value: Any) -> Any:
        """Override to add object interning."""
        # Ensure memory optimizer exists
        if not hasattr(cls, '_memory_optimizer') or cls._memory_optimizer is None:
            cls._memory_optimizer = MemoryOptimizedRegistry()
        
        value = cls._memory_optimizer.intern_object(value)
        return super()._intern_artifact(value)
    
    @classmethod
    def _create_optimized_weak_ref(cls, obj: Any) -> weakref.ref:
        """Create memory-optimized weak reference."""
        if hasattr(cls, '_memory_optimizer'):
            return cls._memory_optimizer.create_weak_ref_with_cleanup(obj)
        return weakref.ref(obj)
    
    @classmethod
    def optimize_memory(cls) -> Dict[str, Any]:
        """Perform memory optimization."""
        stats = {}
        
        # Force garbage collection
        if hasattr(cls, '_memory_optimizer'):
            gc_stats = cls._memory_optimizer.force_garbage_collection()
            stats['garbage_collection'] = gc_stats
        
        # Compact storage
        original_size = sys.getsizeof(cls._repository) if hasattr(cls, '_repository') else 0
        compact_data = cls._compact_storage()
        
        if compact_data:
            cls._repository = compact_data
            new_size = sys.getsizeof(cls._repository)
            stats['storage_optimization'] = {
                'original_size': original_size,
                'new_size': new_size,
                'saved_bytes': original_size - new_size,
                'compression_ratio': new_size / original_size if original_size > 0 else 1.0
            }
        
        # Get memory statistics
        stats['memory_usage'] = cls._estimate_memory_usage()
        if hasattr(cls, '_memory_optimizer'):
            stats['optimizer_stats'] = cls._memory_optimizer.get_memory_stats()
        
        return stats
    
    @classmethod
    def get_memory_report(cls) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        report = {
            'registry_name': cls.__name__,
            'registry_type': cls.__class__.__name__,
            'memory_usage': cls._estimate_memory_usage(),
        }
        
        if hasattr(cls, '_memory_optimizer'):
            report['optimizer_stats'] = cls._memory_optimizer.get_memory_stats()
        
        # Add Python memory info
        report['python_memory'] = {
            'total_objects': len(gc.get_objects()),
            'garbage_collected': gc.collect(),
            'reference_cycles': len(gc.garbage),
        }
        
        return report

# registry/core/memory_monitor.py
"""Memory monitoring utilities."""

import psutil
import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory percentage
    registry_size: int  # Number of items in registry
    
class MemoryMonitor:
    """Monitor memory usage of registry operations."""
    
    def __init__(self, registry_class: type, sample_interval: float = 1.0):
        self.registry_class = registry_class
        self.sample_interval = sample_interval
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable[[MemorySnapshot], None]] = []
    
    def add_callback(self, callback: Callable[[MemorySnapshot], None]) -> None:
        """Add callback to be called on each memory snapshot."""
        self.callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                registry_size = (
                    len(self.registry_class._repository) 
                    if hasattr(self.registry_class, '_repository') 
                    else 0
                )
                
                snapshot = MemorySnapshot(
                    timestamp=time.time(),
                    rss_mb=memory_info.rss / 1024 / 1024,
                    vms_mb=memory_info.vms / 1024 / 1024,
                    percent=memory_percent,
                    registry_size=registry_size
                )
                
                self.snapshots.append(snapshot)
                
                # Call callbacks
                for callback in self.callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        print(f"Memory monitor callback error: {e}")
                
                # Keep only last 1000 snapshots
                if len(self.snapshots) > 1000:
                    self.snapshots = self.snapshots[-1000:]
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def get_memory_trend(self, minutes: int = 5) -> Dict[str, float]:
        """Get memory trend over specified minutes."""
        if not self.snapshots:
            return {}
        
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [
            s for s in self.snapshots 
            if s.timestamp >= cutoff_time
        ]
        
        if len(recent_snapshots) < 2:
            return {}
        
        start_snapshot = recent_snapshots[0]
        end_snapshot = recent_snapshots[-1]
        
        return {
            'rss_change_mb': end_snapshot.rss_mb - start_snapshot.rss_mb,
            'vms_change_mb': end_snapshot.vms_mb - start_snapshot.vms_mb,
            'percent_change': end_snapshot.percent - start_snapshot.percent,
            'registry_size_change': end_snapshot.registry_size - start_snapshot.registry_size,
            'duration_minutes': (end_snapshot.timestamp - start_snapshot.timestamp) / 60,
            'average_rss_mb': sum(s.rss_mb for s in recent_snapshots) / len(recent_snapshots),
            'peak_rss_mb': max(s.rss_mb for s in recent_snapshots),
            'min_rss_mb': min(s.rss_mb for s in recent_snapshots),
        }
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0, window_minutes: int = 10) -> Optional[str]:
        """Detect potential memory leaks."""
        trend = self.get_memory_trend(window_minutes)
        
        if trend and trend.get('rss_change_mb', 0) > threshold_mb:
            return f"Potential memory leak detected: {trend['rss_change_mb']:.2f}MB increase over {window_minutes} minutes"
        
        return None

# Context manager for memory profiling
class MemoryProfiler:
    """Context manager for profiling memory usage of registry operations."""
    
    def __init__(self, operation_name: str, registry_class: type):
        self.operation_name = operation_name
        self.registry_class = registry_class
        self.start_memory = None
        self.end_memory = None
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        process = psutil.Process()
        self.start_memory = process.memory_info()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        process = psutil.Process()
        self.end_memory = process.memory_info()
    
    def get_profile_report(self) -> Dict[str, Any]:
        """Get memory profile report."""
        if not self.start_memory or not self.end_memory:
            return {}
        
        return {
            'operation': self.operation_name,
            'registry': self.registry_class.__name__,
            'duration_seconds': self.end_time - self.start_time,
            'memory_change_mb': (self.end_memory.rss - self.start_memory.rss) / 1024 / 1024,
            'start_rss_mb': self.start_memory.rss / 1024 / 1024,
            'end_rss_mb': self.end_memory.rss / 1024 / 1024,
            'start_vms_mb': self.start_memory.vms / 1024 / 1024,
            'end_vms_mb': self.end_memory.vms / 1024 / 1024,
        }

# Decorator for memory profiling
def profile_memory(operation_name: str = None):
    """Decorator to profile memory usage of registry methods."""
    def decorator(func):
        def wrapper(cls, *args, **kwargs):
            op_name = operation_name or f"{cls.__name__}.{func.__name__}"
            
            with MemoryProfiler(op_name, cls) as profiler:
                result = func(cls, *args, **kwargs)
            
            # Optionally log or store profile report
            report = profiler.get_profile_report()
            if report.get('memory_change_mb', 0) > 1.0:  # Log if > 1MB change
                print(f"Memory profile: {op_name} used {report['memory_change_mb']:.2f}MB")
            
            return result
        return wrapper
    return decorator
