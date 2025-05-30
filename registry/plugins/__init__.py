# registry/plugins/__init__.py
"""Plugin system for registry pattern extensions."""

import importlib
import logging
from typing import Dict, List, Optional, Type, Any, Set

from .base import BaseRegistryPlugin, RegistryPluginError, PluginNotFoundError

logger = logging.getLogger(__name__)

class PluginManager:
    """Manages registry plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, BaseRegistryPlugin] = {}
        self._loaded_plugins: Set[str] = set()
        self._plugin_registry: Dict[str, str] = {
            # Map plugin names to their module paths
            'pydantic': 'registry.plugins.pydantic_plugin',
            'observability': 'registry.plugins.observability_plugin',
            'serialization': 'registry.plugins.serialization_plugin',
            'caching': 'registry.plugins.caching_plugin',
            'remote': 'registry.plugins.remote_plugin',
            'cli': 'registry.plugins.cli_plugin',
        }
    
    def register_plugin_module(self, plugin_name: str, module_path: str) -> None:
        """Register a plugin module path."""
        self._plugin_registry[plugin_name] = module_path
    
    def load_plugin(self, plugin_name: str, force_new_instance: bool = False) -> Optional[BaseRegistryPlugin]:
        """Load a plugin by name."""
        # Always create new instances to avoid reinitialization issues
        if plugin_name in self._loaded_plugins and not force_new_instance:
            # Create a new instance instead of reusing
            return self._create_plugin_instance(plugin_name)
        
        if plugin_name not in self._plugin_registry:
            logger.warning(f"Unknown plugin: {plugin_name}")
            return None
        
        return self._create_plugin_instance(plugin_name)
    
    def _create_plugin_instance(self, plugin_name: str) -> Optional[BaseRegistryPlugin]:
        """Create a new plugin instance."""
        try:
            module_path = self._plugin_registry[plugin_name]
            module = importlib.import_module(module_path)
            
            # Get the plugin class (assumes naming convention: PluginNamePlugin)
            plugin_class_name = self._get_plugin_class_name(plugin_name)
            plugin_class = getattr(module, plugin_class_name)
            plugin_instance = plugin_class()
            
            # Store reference but allow multiple instances
            if plugin_name not in self._loaded_plugins:
                self._plugins[plugin_name] = plugin_instance
                self._loaded_plugins.add(plugin_name)
                logger.info(f"Loaded plugin: {plugin_name} v{plugin_instance.version}")
            
            return plugin_instance
            
        except ImportError as e:
            logger.warning(f"Could not load plugin {plugin_name}: {e}")
            return None
        except AttributeError as e:
            logger.error(f"Plugin {plugin_name} does not have expected class {plugin_class_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return None
    
    def _get_plugin_class_name(self, plugin_name: str) -> str:
        """Generate expected plugin class name from plugin name."""
        # Convert plugin_name to PascalCase + "Plugin"
        # e.g., "pydantic" -> "PydanticPlugin", "custom_logger" -> "CustomLoggerPlugin"
        words = plugin_name.split('_')
        class_name = ''.join(word.capitalize() for word in words) + 'Plugin'
        return class_name
    
    def get_plugin(self, name: str) -> Optional[BaseRegistryPlugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all loaded plugins."""
        return list(self._plugins.keys())
    
    def list_available_plugins(self) -> List[str]:
        """List all available plugins (loaded and unloaded)."""
        return list(self._plugin_registry.keys())
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a plugin."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.get_status()
        elif plugin_name in self._plugin_registry:
            return {
                "name": plugin_name,
                "status": "available_not_loaded",
                "module_path": self._plugin_registry[plugin_name]
            }
        return None
    
    def cleanup_all(self) -> None:
        """Cleanup all loaded plugins."""
        for plugin in self._plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin.name}: {e}")
    
    def reload_plugin(self, plugin_name: str) -> Optional[BaseRegistryPlugin]:
        """Reload a plugin (cleanup and load again)."""
        if plugin_name in self._plugins:
            plugin = self._plugins[plugin_name]
            plugin.cleanup()
            del self._plugins[plugin_name]
            self._loaded_plugins.discard(plugin_name)
        
        return self.load_plugin(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[BaseRegistryPlugin]:
        """Get all loaded plugins of a specific type."""
        # This could be extended to support plugin categories
        return [p for p in self._plugins.values() if plugin_type in p.name.lower()]

# Global plugin manager instance
_plugin_manager = PluginManager()

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager."""
    return _plugin_manager

def load_plugin(name: str) -> Optional[BaseRegistryPlugin]:
    """Convenience function to load a plugin."""
    return _plugin_manager.load_plugin(name)

def get_plugin(name: str) -> Optional[BaseRegistryPlugin]:
    """Convenience function to get a loaded plugin."""
    return _plugin_manager.get_plugin(name)

def list_plugins() -> List[str]:
    """Convenience function to list loaded plugins."""
    return _plugin_manager.list_plugins()

def list_available_plugins() -> List[str]:
    """Convenience function to list all available plugins."""
    return _plugin_manager.list_available_plugins()

def register_plugin(plugin_name: str, module_path: str) -> None:
    """Register a custom plugin."""
    _plugin_manager.register_plugin_module(plugin_name, module_path)

# Auto-discovery of plugins
def discover_plugins() -> List[str]:
    """Discover available plugins automatically."""
    discovered = []
    for plugin_name in _plugin_manager.list_available_plugins():
        try:
            plugin = _plugin_manager.load_plugin(plugin_name)
            if plugin:
                discovered.append(plugin_name)
        except Exception:
            pass  # Plugin not available
    return discovered

# Plugin decorator for easy registration
def plugin(name: str):
    """Decorator to register a plugin class."""
    def decorator(cls):
        # Register the plugin class in the global registry
        module_path = f"{cls.__module__}"
        register_plugin(name, module_path)
        return cls
    return decorator

__all__ = [
    'BaseRegistryPlugin',
    'PluginManager',
    'RegistryPluginError',
    'PluginNotFoundError',
    'get_plugin_manager',
    'load_plugin',
    'get_plugin',
    'list_plugins',
    'list_available_plugins',
    'register_plugin',
    'discover_plugins',
    'plugin',
]
