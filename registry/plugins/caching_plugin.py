# registry/plugins/caching_plugin.py
"""Caching plugin for advanced caching strategies."""

try:
    from cachetools import TTLCache, LRUCache, LFUCache
    import diskcache
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    TTLCache = LRUCache = LFUCache = diskcache = None

import time
from typing import Type, Any, Dict, Optional, Union, List
from .base import BaseRegistryPlugin
from ..core._validator import ValidationError

class CachingPlugin(BaseRegistryPlugin):
    """Plugin for advanced caching strategies."""
    
    @property
    def name(self) -> str:
        return "caching"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def dependencies(self) -> List[str]:
        return ["cachetools", "diskcache"]
    
    def _on_initialize(self) -> None:
        if not CACHING_AVAILABLE:
            raise ImportError("Caching dependencies required. Install with: pip install registry[caching-plugin]")
        
        self.memory_cache = None
        self.disk_cache = None
        
        if self.registry_class:
            self._add_caching_methods()
    
    def _add_caching_methods(self) -> None:
        """Add caching methods to the registry."""
        
        def setup_memory_cache(cache_type: str = "lru", max_size: int = 1000, ttl: Optional[int] = None) -> None:
            """Setup memory cache."""
            if cache_type == "lru":
                self.memory_cache = LRUCache(maxsize=max_size)
            elif cache_type == "lfu":
                self.memory_cache = LFUCache(maxsize=max_size)
            elif cache_type == "ttl" and ttl:
                self.memory_cache = TTLCache(maxsize=max_size, ttl=ttl)
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")
        
        def setup_disk_cache(cache_dir: str, size_limit: int = 1024*1024*100) -> None:
            """Setup disk cache."""
            self.disk_cache = diskcache.Cache(cache_dir, size_limit=size_limit)
        
        def get_cached_artifact(key: Any) -> Optional[Any]:
            """Get artifact from cache."""
            # Try memory cache first
            if self.memory_cache and key in self.memory_cache:
                return self.memory_cache[key]
            
            # Try disk cache
            if self.disk_cache and key in self.disk_cache:
                value = self.disk_cache[key]
                # Promote to memory cache
                if self.memory_cache:
                    self.memory_cache[key] = value
                return value
            
            return None
        
        def cache_artifact(key: Any, value: Any) -> None:
            """Cache an artifact."""
            if self.memory_cache:
                self.memory_cache[key] = value
            if self.disk_cache:
                self.disk_cache[key] = value
        
        def clear_cache() -> None:
            """Clear all caches."""
            if self.memory_cache:
                self.memory_cache.clear()
            if self.disk_cache:
                self.disk_cache.clear()
        
        def get_cache_stats() -> Dict[str, Any]:
            """Get cache statistics."""
            stats = {}
            if self.memory_cache:
                stats["memory"] = {
                    "size": len(self.memory_cache),
                    "maxsize": getattr(self.memory_cache, 'maxsize', 'unlimited'),
                }
            if self.disk_cache:
                stats["disk"] = {
                    "size": len(self.disk_cache),
                    "volume": self.disk_cache.volume(),
                }
            return stats
        
        # Override get_artifact to use cache
        original_get_artifact = getattr(self.registry_class, 'get_artifact', None)
        
        def get_artifact_with_cache(key: Any):
            """Get artifact with caching."""
            cached_value = get_cached_artifact(key)
            if cached_value is not None:
                return cached_value
            
            value = original_get_artifact(key)
            cache_artifact(key, value)
            return value
        
        # Add methods to registry class as unbound functions
        setattr(self.registry_class, 'setup_memory_cache', setup_memory_cache)
        setattr(self.registry_class, 'setup_disk_cache', setup_disk_cache)
        setattr(self.registry_class, 'get_cached_artifact', get_cached_artifact)
        setattr(self.registry_class, 'cache_artifact', cache_artifact)
        setattr(self.registry_class, 'clear_cache', clear_cache)
        setattr(self.registry_class, 'get_cache_stats', get_cache_stats)
        
        # Only override get_artifact if original exists
        if original_get_artifact:
            setattr(self.registry_class, 'get_artifact', get_artifact_with_cache)
