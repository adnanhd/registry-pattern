# registry/plugins/base.py
"""Base classes for registry plugins."""

from abc import ABC, abstractmethod
from ..mixin import MutableRegistryValidatorMixin
from typing import Type, Any, Dict, Optional, List

class BaseRegistryPlugin(ABC):
    """Base class for registry plugins."""

    def __init__(self):
        self.registry_class: Optional[Type[MutableRegistryValidatorMixin]] = None
        self.config: Dict[str, Any] = {}
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @property
    def description(self) -> str:
        """Plugin description. Override in subclasses for custom description."""
        return f"{self.name} plugin v{self.version}"

    @property
    def dependencies(self) -> List[str]:
        """List of required dependencies. Override in subclasses."""
        return []

    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized

    def initialize(self, registry_class: Type) -> None:
        """Initialize the plugin with a registry class."""
        # Allow reinitialization with different registry classes
        if self._initialized and self.registry_class == registry_class:
            raise RuntimeError(f"Plugin {self.name} is already initialized with {registry_class.__name__}")

        # If already initialized with a different registry, allow reinitialization
        if self._initialized and self.registry_class != registry_class:
            self.cleanup()

        self.registry_class = registry_class
        self._check_dependencies()
        self._on_initialize()
        self._initialized = True

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        if not self._initialized:
            return

        self._on_cleanup()
        self._initialized = False
        self.registry_class = None

    def configure(self, **config) -> None:
        """Configure the plugin."""
        self.config.update(config)
        if self._initialized:
            self._on_configure()

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def _check_dependencies(self) -> None:
        """Check if all required dependencies are available."""
        missing_deps = []
        for dep in self.dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            deps_str = ", ".join(missing_deps)
            raise ImportError(
                f"Plugin {self.name} requires missing dependencies: {deps_str}. "
                f"Install with: pip install registry[{self.name}-plugin]"
            )

    def _on_initialize(self) -> None:
        """Override in subclasses for initialization logic."""
        pass

    def _on_cleanup(self) -> None:
        """Override in subclasses for cleanup logic."""
        pass

    def _on_configure(self) -> None:
        """Override in subclasses for configuration logic."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "initialized": self._initialized,
            "registry_class": self.registry_class.__name__ if self.registry_class else None,
            "config": dict(self.config),
            "dependencies": self.dependencies,
        }

    def validate_registry_compatibility(self, registry_class: Type) -> bool:
        """
        Validate if the plugin is compatible with the registry class.
        Override in subclasses for custom compatibility checks.
        """
        return True

    def __repr__(self) -> str:
        """String representation of the plugin."""
        status = "initialized" if self._initialized else "not initialized"
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}', {status})"

class RegistryPluginError(Exception):
    """Base exception for plugin-related errors."""
    pass

class PluginNotFoundError(RegistryPluginError):
    """Raised when a plugin is not found."""
    pass

class PluginInitializationError(RegistryPluginError):
    """Raised when plugin initialization fails."""
    pass

class PluginCompatibilityError(RegistryPluginError):
    """Raised when plugin is not compatible with registry."""
    pass

# Decorator for plugin methods
def plugin_method(func):
    """Decorator to mark methods that should only be called on initialized plugins."""
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            raise RuntimeError(f"Plugin {self.name} must be initialized before calling {func.__name__}")
        return func(self, *args, **kwargs)
    return wrapper

# Registry method decorator for plugins
def add_registry_method(method_name: str):
    """Decorator to add a method to the registry class."""
    def decorator(func):
        def wrapper(plugin_self):
            if not plugin_self.registry_class:
                raise RuntimeError("Plugin not initialized with registry class")

            setattr(plugin_self.registry_class, method_name, func)
            return func

        return wrapper
    return decorator
