# registry/plugins/remote_plugin.py
"""Remote registry plugin for distributed registries."""

try:
    import redis
    import httpx
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    REMOTE_AVAILABLE = True
except ImportError:
    REMOTE_AVAILABLE = False
    redis = httpx = FastAPI = HTTPException = JSONResponse = None

import json
import asyncio
from typing import Type, Any, Dict, Optional, Union, List
from .base import BaseRegistryPlugin
from ..core._validator import ValidationError

class RemotePlugin(BaseRegistryPlugin):
    """Plugin for remote registry operations."""
    
    @property
    def name(self) -> str:
        return "remote"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def dependencies(self) -> List[str]:
        return ["redis", "httpx", "fastapi", "uvicorn"]
    
    def _on_initialize(self) -> None:
        if not REMOTE_AVAILABLE:
            raise ImportError("Remote dependencies required. Install with: pip install registry[remote-plugin]")
        
        self.redis_client = None
        self.http_client = None
        self.app = None
        
        if self.registry_class:
            self._add_remote_methods()
    
    def _add_remote_methods(self) -> None:
        """Add remote registry methods."""
        
        def setup_redis_backend(host: str = "localhost", port: int = 6379, db: int = 0, **kwargs) -> None:
            """Setup Redis backend for distributed registry."""
            self.redis_client = redis.Redis(host=host, port=port, db=db, **kwargs)
            
            # Test connection
            try:
                self.redis_client.ping()
            except Exception as e:
                raise ValidationError(
                    f"Failed to connect to Redis: {e}",
                    suggestions=[
                        "Check Redis server is running",
                        "Verify connection parameters",
                        "Check network connectivity"
                    ],
                    context={"host": host, "port": port, "db": db}
                )
        
        def sync_to_redis() -> None:
            """Sync local registry to Redis."""
            if not self.redis_client:
                raise ValidationError("Redis backend not configured")
            
            registry_key = f"registry:{self.registry_class.__name__}"
            
            # Get serializable data
            try:
                data = self.registry_class._serialize_registry() if hasattr(self.registry_class, '_serialize_registry') else {}
            except Exception:
                data = {"error": "Could not serialize registry"}
            
            try:
                self.redis_client.set(registry_key, json.dumps(data))
            except Exception as e:
                raise ValidationError(
                    f"Failed to sync to Redis: {e}",
                    suggestions=["Check Redis connection", "Verify data is serializable"],
                    context={"registry_name": self.registry_class.__name__}
                )
        
        def sync_from_redis() -> None:
            """Sync registry from Redis."""
            if not self.redis_client:
                raise ValidationError("Redis backend not configured")
            
            registry_key = f"registry:{self.registry_class.__name__}"
            
            try:
                data = self.redis_client.get(registry_key)
                if data:
                    parsed_data = json.loads(data)
                    if hasattr(self.registry_class, '_deserialize_registry'):
                        self.registry_class._deserialize_registry(parsed_data)
            except Exception as e:
                raise ValidationError(
                    f"Failed to sync from Redis: {e}",
                    suggestions=["Check Redis connection", "Verify data format"],
                    context={"registry_name": self.registry_class.__name__}
                )
        
        def setup_http_client(base_url: str, timeout: int = 30) -> None:
            """Setup HTTP client for remote registry operations."""
            self.http_client = httpx.Client(base_url=base_url, timeout=timeout)
        
        async def register_remote(key: Any, value: Any) -> None:
            """Register artifact on remote registry."""
            if not self.http_client:
                raise ValidationError("HTTP client not configured")
            
            try:
                response = self.http_client.post(
                    "/register",
                    json={"key": str(key), "value": str(value), "registry": self.registry_class.__name__}
                )
                response.raise_for_status()
            except Exception as e:
                raise ValidationError(
                    f"Failed to register remotely: {e}",
                    suggestions=["Check network connection", "Verify remote server is running"],
                    context={"key": str(key), "registry": self.registry_class.__name__}
                )
        
        def create_http_server(host: str = "0.0.0.0", port: int = 8000) -> FastAPI:
            """Create HTTP server for remote registry access."""
            app = FastAPI(title=f"{self.registry_class.__name__} Registry API")
            
            @app.get("/")
            async def root():
                return {"registry": self.registry_class.__name__, "size": self.registry_class._len_mapping()}
            
            @app.get("/artifacts")
            async def list_artifacts():
                try:
                    mapping = self.registry_class._get_mapping()
                    return {
                        "artifacts": {str(k)[:100]: str(v)[:100] for k, v in mapping.items()},
                        "size": self.registry_class._len_mapping()
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.get("/artifacts/{key}")
            async def get_artifact(key: str):
                try:
                    value = self.registry_class.get_artifact(key)
                    return {"key": key, "value": str(value)}
                except Exception as e:
                    raise HTTPException(status_code=404, detail=str(e))
            
            @app.post("/register")
            async def register_artifact(data: Dict[str, Any]):
                try:
                    key = data.get("key")
                    value = data.get("value")
                    self.registry_class.register_artifact(key, value)
                    return {"status": "registered", "key": key}
                except Exception as e:
                    raise HTTPException(status_code=400, detail=str(e))
            
            @app.delete("/artifacts/{key}")
            async def unregister_artifact(key: str):
                try:
                    self.registry_class.unregister_artifact(key)
                    return {"status": "unregistered", "key": key}
                except Exception as e:
                    raise HTTPException(status_code=404, detail=str(e))
            
            return app
        
        # Add methods to registry class as unbound functions
        setattr(self.registry_class, 'setup_redis_backend', setup_redis_backend)
        setattr(self.registry_class, 'sync_to_redis', sync_to_redis)
        setattr(self.registry_class, 'sync_from_redis', sync_from_redis)
        setattr(self.registry_class, 'setup_http_client', setup_http_client)
        setattr(self.registry_class, 'register_remote', register_remote)
        setattr(self.registry_class, 'create_http_server', create_http_server)
