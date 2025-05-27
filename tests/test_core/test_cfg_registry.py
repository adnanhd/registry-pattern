import pytest
from types import new_class
from typing import Protocol, Dict, Any
import weakref
import gc

from registry import RegistryError, ConfigRegistry, ConformanceError


# -------------------------------------------------------------------
# Define test classes
# -------------------------------------------------------------------


class TestObject:
    """Simple class for testing ConfigRegistry."""

    __test__ = False

    def __init__(self, name: str):
        self.name = name


# -------------------------------------------------------------------
# Base test class for ConfigRegistry
# -------------------------------------------------------------------


class BaseConfigRegistryTest:
    """Base class for ConfigRegistry tests."""

    @pytest.fixture
    def registry_class(self):
        """Return a registry class for testing. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement this fixture")

    def test_register_artifact(self, registry_class):
        """Test registering an object with configuration."""
        obj = TestObject("test")
        config = {"param1": 42, "param2": "value"}

        registry_class.register_artifact(obj, config)

        # Verify registration
        assert registry_class.has_identifier(obj)
        assert registry_class.get_artifact(obj) == config

    def test_unregister_artifact(self, registry_class):
        """Test unregistering a configuration."""
        obj = TestObject("test")
        config = {"param1": 42, "param2": "value"}

        registry_class.register_artifact(obj, config)
        registry_class.unregister_artifact(obj)

        # After unregistration, the object should not be in the registry
        assert not registry_class.has_identifier(obj)
        with pytest.raises(RegistryError):
            registry_class.get_artifact(obj)

    def test_register_duplicate_config(self, registry_class):
        """Test registering the same object twice raises an error."""
        obj = TestObject("test")
        config1 = {"param1": 42}
        config2 = {"param1": 43}

        registry_class.register_artifact(obj, config1)
        with pytest.raises(RegistryError):
            registry_class.register_artifact(obj, config2)

        # Original config should remain
        assert registry_class.has_identifier(obj)
        assert registry_class.get_artifact(obj) == config1

    def test_validate_artifact(self, registry_class):
        """Test validating configuration dictionaries."""
        config = {"param1": 42}
        validated = registry_class.validate_artifact(config)
        assert validated == config

        # Non-dictionary should raise TypeError
        with pytest.raises(TypeError):
            registry_class.validate_artifact("not a dictionary")


# -------------------------------------------------------------------
# Test class for standard ConfigRegistry
# -------------------------------------------------------------------


class TestConfigRegistry(BaseConfigRegistryTest):
    """Tests for basic ConfigRegistry functionality."""

    @pytest.fixture
    def registry_class(self):
        """Create a standard ConfigRegistry for TestObject."""
        return new_class("CfgRegistry", (ConfigRegistry[TestObject, Dict[str, Any]],))

    def test_register_class_configs(self, registry_class):
        """Test automatic registration of configurations via register_class_configs."""
        # Create a tracked version of TestObject
        TrackedClass = registry_class.register_class_configs(TestObject)

        # Create an instance with configuration
        instance = TrackedClass(name="auto_registered")

        # Verify registration with the correct config
        assert registry_class.has_identifier(instance)
        assert registry_class.get_artifact(instance) == {"name": "auto_registered"}

    def test_register_builder(self, registry_class):
        """Test the register_builder decorator if available."""
        # Check if register_builder is available
        if hasattr(registry_class, "register_builder"):

            @registry_class.register_builder
            def create_object(name: str, value: int):
                return TestObject(name)

            # Create an object using the builder
            obj = create_object(name="built", value=42)

            # Verify registration with the correct config
            assert registry_class.has_identifier(obj)
            assert registry_class.get_artifact(obj) == {"name": "built", "value": 42}

    def test_weak_references(self, registry_class):
        """Test that weak references are used properly."""
        # Register an object
        obj = TestObject("test_weak_ref")
        config = {"some": "data"}
        registry_class.register_artifact(obj, config)

        # Get the repository and check that it contains a weakref
        repo = registry_class._repository
        # Find the key for our object
        weak_key = None
        for key in repo:
            if isinstance(key, weakref.ref):
                ref_obj = key()
                if ref_obj is obj:
                    weak_key = key
                    break

        assert weak_key is not None, "Object should be stored as a weak reference"
        assert isinstance(weak_key, weakref.ref), "Key should be a weak reference"

        # Access the config through the regular interface
        assert registry_class.get_artifact(obj) == config


# -------------------------------------------------------------------
# Test class for ConfigRegistry with custom validation
# -------------------------------------------------------------------


class CustomConfigProtocol(Protocol):
    """Protocol for objects with custom configuration requirements."""

    name: str

    def get_info(self) -> str: ...


class TestCustomConfigRegistry:
    """Tests for ConfigRegistry with custom validation."""

    @pytest.fixture
    def registry_class(self):
        """Create a strict ConfigRegistry for custom objects."""

        class CustomObject:
            def __init__(self, name: str):
                self.name = name

            def get_info(self) -> str:
                return f"Object: {self.name}"

        class CustomCfgRegistry(
            ConfigRegistry[CustomObject, Dict[str, Any]], strict=True
        ):
            @classmethod
            def _intern_artifact(cls, value: Dict[str, Any]) -> Dict[str, Any]:
                # Custom validation to ensure config has required fields
                if not isinstance(value, dict):
                    raise TypeError("Configuration must be a dictionary")
                if "name" not in value:
                    raise ValueError("Configuration must contain 'name' key")
                return value

        return CustomCfgRegistry

    def test_register_with_valid_config(self, registry_class):
        """Test registration with a valid configuration."""

        class CustomObject:
            def __init__(self, name: str):
                self.name = name

            def get_info(self) -> str:
                return f"Object: {self.name}"

        obj = CustomObject("valid")
        config = {"name": "valid", "extra": "data"}

        registry_class.register_artifact(obj, config)
        assert registry_class.has_identifier(obj)
        assert registry_class.get_artifact(obj) == config

    def test_register_with_invalid_config(self, registry_class):
        """Test registration with an invalid configuration raises an error."""

        class CustomObject:
            def __init__(self, name: str):
                self.name = name

            def get_info(self) -> str:
                return f"Object: {self.name}"

        obj = CustomObject("invalid")
        invalid_config = {"missing_name": "value"}

        with pytest.raises(ValueError):
            registry_class.register_artifact(obj, invalid_config)

        # Object should not be registered
        assert not registry_class.has_identifier(obj)

    def test_register_non_conforming_object(self, registry_class):
        """Test that non-conforming objects cannot be registered in strict mode."""

        class NonConformingObject:
            # Missing get_info method and name attribute
            def __init__(self, value: int):
                self.value = value

        obj = NonConformingObject(42)
        config = {"name": "test"}

        with pytest.raises(ConformanceError):
            registry_class.register_artifact(obj, config)

        # Object should not be registered
        assert not registry_class.has_identifier(obj)


# -------------------------------------------------------------------
# Test class for weak reference behavior
# -------------------------------------------------------------------


class TestWeakReferenceBehavior:
    """Tests for weak reference behavior in ConfigRegistry."""

    @pytest.fixture
    def registry_class(self):
        """Create a standard ConfigRegistry."""
        return new_class("WeakRefMap", (ConfigRegistry[TestObject, Dict[str, Any]],))

    def test_instance_garbage_collection(self, registry_class):
        """Test that objects are removed from registry when garbage collected."""
        # Register an object
        obj = TestObject("temporary")
        config = {"temporary": True}
        registry_class.register_artifact(obj, config)

        # Verify registration
        assert registry_class.has_identifier(obj)

        # Create a weak reference to the object to check later
        obj_ref = weakref.ref(obj)

        # Remove the only strong reference to the object
        del obj

        # Force garbage collection
        gc.collect()

        # Verify that our object has been garbage collected
        assert obj_ref() is None, "Object should have been garbage collected"

        # Cleanup should remove the dead reference and return 1
        cleanup_count = registry_class.cleanup()
        assert (
            cleanup_count >= 1
        ), "Cleanup should have removed at least one dead reference"

        # Repository should now be empty
        assert (
            len(registry_class._repository) == 0
        ), "Repository should be empty after cleanup"

    def test_cleanup_method(self, registry_class):
        """Test the cleanup method for removing dead references."""
        # Register multiple objects
        objs = [TestObject(f"obj{i}") for i in range(3)]
        for i, obj in enumerate(objs):
            registry_class.register_artifact(obj, {"index": i})

        # Verify all are registered
        for obj in objs:
            assert registry_class.has_identifier(obj)

        # Remove references to the first two objects
        del objs[0]
        del objs[0]  # Now deletes the original second object

        # Force garbage collection
        gc.collect()

        # Cleanup should remove two dead references
        cleanup_count = registry_class.cleanup()
        assert (
            cleanup_count == 2
        ), "Cleanup should have removed exactly two dead references"

        # Only one object should remain in the repository
        assert (
            len(registry_class._repository) == 1
        ), "Repository should have only one entry left"
