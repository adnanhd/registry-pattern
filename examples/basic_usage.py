# Fixed Plugin Example - Uses Correct ObjectRegistry Patterns

from registry import ObjectRegistry
from registry.plugins.base import BaseRegistryPlugin
from typing import Type, Any, Dict

class CustomLoggingPlugin(BaseRegistryPlugin):
    """Custom plugin for enhanced logging."""

    @property
    def name(self) -> str:
        return "custom_logging"

    @property
    def version(self) -> str:
        return "1.0.0"

    def _on_initialize(self) -> None:
        """Initialize custom logging."""
        self.log_file = self.config.get('log_file', 'registry.log')
        self.log_level = self.config.get('log_level', 'INFO')

        if self.registry_class:
            self._add_logging_methods()

    def _add_logging_methods(self) -> None:
        """Add custom logging methods."""
        import logging

        # Setup logger
        logger = logging.getLogger(f"registry.{self.registry_class.__name__}")
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, self.log_level))

        # Wrap methods with logging
        original_register = self.registry_class.register_artifact
        original_unregister_id = self.registry_class.unregister_identifier
        original_unregister_artifact = self.registry_class.unregister_artifact

        @classmethod
        def register_with_logging(cls, *args, **kwargs):
            """Register artifact with logging."""
            try:
                result = original_register(*args, **kwargs)
                if len(args) == 2:
                    logger.info(f"Registered artifact with key: {args[0]}")
                else:
                    logger.info(f"Registered artifact with auto-key: {args[0]}")
                return result
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                raise

        @classmethod
        def unregister_identifier_with_logging(cls, *args, **kwargs):
            """Unregister by identifier with logging."""
            try:
                result = original_unregister_id(*args, **kwargs)
                logger.info(f"Unregistered by identifier: {args[0] if args else 'unknown'}")
                return result
            except Exception as e:
                logger.error(f"Unregistration by identifier failed: {e}")
                raise

        @classmethod
        def unregister_artifact_with_logging(cls, *args, **kwargs):
            """Unregister by artifact with logging."""
            try:
                result = original_unregister_artifact(*args, **kwargs)
                logger.info(f"Unregistered artifact: {args[0] if args else 'unknown'}")
                return result
            except Exception as e:
                logger.error(f"Unregistration by artifact failed: {e}")
                raise

        # Replace methods
        setattr(self.registry_class, 'register_artifact', register_with_logging)
        setattr(self.registry_class, 'unregister_identifier', unregister_identifier_with_logging)
        setattr(self.registry_class, 'unregister_artifact', unregister_artifact_with_logging)

# Usage of custom plugin
class LoggedRegistry(ObjectRegistry[object]):
    pass

    @classmethod
    def _identifier_of(cls, item):
        reversed_dict = {v: k for k, v in cls._repository.items()}
        return reversed_dict.get(item)

# Create and configure custom plugin
custom_plugin = CustomLoggingPlugin()
custom_plugin.configure(log_file='custom_registry.log', log_level='DEBUG')
custom_plugin.initialize(LoggedRegistry)

print("=== Fixed Plugin Example - Correct ObjectRegistry Usage ===\n")

# =====================================
# Correct Approach 1: Explicit Keys
# =====================================
print("Approach 1: Explicit Key Management")
LoggedRegistry.register_artifact("logged_key_1", "logged_value_1")
print(f"Registry has 'logged_key_1': {LoggedRegistry.has_identifier('logged_key_1')}")

# CORRECT: Use unregister_identifier() with the same key
LoggedRegistry.unregister_identifier("logged_key_1")
print(f"After unregister_identifier: {LoggedRegistry.has_identifier('logged_key_1')}")
print()

# =====================================
# Correct Approach 2: Auto Keys
# =====================================
print("Approach 2: Auto-Key Management")
logged_value_2 = "logged_value_2"
LoggedRegistry.register_artifact(logged_value_2)  # Single argument!
print(f"Registry has artifact: {LoggedRegistry.has_artifact(logged_value_2)}")

# CORRECT: Use unregister_artifact() when registered with single argument
LoggedRegistry.unregister_artifact(logged_value_2)
print(f"After unregister_artifact: {LoggedRegistry.has_artifact(logged_value_2)}")
print()

# =====================================
# Correct Approach 3: Mixed Pattern
# =====================================
print("Approach 3: Mixed Pattern (Get-Then-Unregister)")
LoggedRegistry.register_artifact("logged_key_3", "logged_value_3")
retrieved_artifact = LoggedRegistry.get_artifact("logged_key_3")
LoggedRegistry.unregister_artifact(retrieved_artifact)
print(f"After mixed approach: {LoggedRegistry.has_identifier('logged_key_3')}")
print()

# =====================================
# Correct Approach 4: Objects
# =====================================
print("Approach 4: Object Registration")

class LoggedObject:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"LoggedObject({self.name})"

# Register object with auto-key (recommended for objects)
obj = LoggedObject("test_object")
LoggedRegistry.register_artifact(obj)
print(f"Registered object: {obj}")
print(f"Has object: {LoggedRegistry.has_artifact(obj)}")

# Unregister using the object itself
LoggedRegistry.unregister_artifact(obj)
print(f"After unregister: {LoggedRegistry.has_artifact(obj)}")

print("\n=== All approaches completed successfully! ===")
print("Check custom_registry.log for detailed log entries")

# Cleanup
import os
if os.path.exists('custom_registry.log'):
    print(f"\nLog file contents:")
    with open('custom_registry.log', 'r') as f:
        print(f.read())
