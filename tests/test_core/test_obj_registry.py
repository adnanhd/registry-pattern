import gc
import pytest
import weakref
from types import new_class
from typing import Protocol

from registry import (
    ObjectRegistry,
    RegistryError,
    ConformanceError,
    InheritanceError,
)
from registry.core._dev_utils import P


# -------------------------------------------------------------------
# Define a protocol for executable objects
# -------------------------------------------------------------------


class Executable(Protocol):
    def execute(self) -> str: ...


# -------------------------------------------------------------------
# Base test class for ObjectRegistry
# -------------------------------------------------------------------


class BaseObjectRegistryTest:
    """Base class for ObjectRegistry tests."""

    @pytest.fixture
    def registry_class(self):
        """Return a registry class for testing. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement this fixture")

    def test_register_artifact(self, registry_class):
        """Test that an instance can be registered and retrieved."""

        class DummyObject:
            pass

        obj = DummyObject()
        obj_id = hex(id(obj))

        registry_class.register_artifact(obj_id, obj)
        assert registry_class.has_identifier(obj_id)
        assert registry_class.has_artifact(obj)
        assert registry_class.get_artifact(obj_id) is obj

    def test_unregister_artifact(self, registry_class):
        """Test that an instance can be unregistered."""

        class DummyObject:
            pass

        obj = DummyObject()
        obj_id = hex(id(obj))

        registry_class.register_artifact(obj_id, obj)
        assert registry_class.has_artifact(obj)

        registry_class.unregister_artifact(obj_id)
        assert not registry_class.has_artifact(obj)
        with pytest.raises(RegistryError):
            registry_class.get_artifact(obj_id)

    def test_duplicate_registration(self, registry_class):
        """Test that registering the same instance twice raises an error."""

        class DummyObject:
            pass

        obj = DummyObject()
        obj_id = hex(id(obj))

        registry_class.register_artifact(obj_id, obj)
        with pytest.raises(RegistryError):
            registry_class.register_artifact(obj_id, obj)

    def test_register_class_instances(self, registry_class):
        """Test automatic registration of instances via register_class_instances."""

        class DummyObject:
            def __init__(self, value):
                self.value = value

        # Create a tracked version of the class
        TrackedClass = registry_class.register_class_instances(DummyObject)

        # Creating an instance should automatically register it
        instance = TrackedClass(42)

        # Verify it's registered
        assert registry_class.has_artifact(instance)
        # The instance should be retrievable using itself as the key
        assert registry_class.get_artifact(instance) is instance


# -------------------------------------------------------------------
# Test class for non-strict ObjectRegistry
# -------------------------------------------------------------------


class TestNonStrictObjectRegistry(BaseObjectRegistryTest):
    """Tests for ObjectRegistry in non-strict mode."""

    @pytest.fixture
    def registry_class(self):
        """Create a non-strict registry for generic objects."""
        return new_class("ObjectReg", (ObjectRegistry[object],))

    def test_register_any_instance(self, registry_class):
        """Test that any type of instance can be registered in non-strict mode."""
        # Register an integer
        registry_class.register_artifact("int_key", 42)
        assert registry_class.get_artifact("int_key") == 42

        # Register a string
        registry_class.register_artifact("str_key", "hello")
        assert registry_class.get_artifact("str_key") == "hello"

        # Register a list
        my_list = [1, 2, 3]
        registry_class.register_artifact("list_key", my_list)
        assert registry_class.get_artifact("list_key") is my_list


# -------------------------------------------------------------------
# Test class for strict ObjectRegistry
# -------------------------------------------------------------------


class TestStrictObjectRegistry:
    """Tests for ObjectRegistry in strict mode."""

    @pytest.fixture
    def registry_class(self):
        """Create a strict registry for Executable objects."""
        return new_class(
            "StrictObjectReg", (ObjectRegistry[Executable],), {"strict": True}
        )

    def test_register_valid_instance(self, registry_class):
        """Test that a conforming instance can be registered in strict mode."""

        class ValidExecutable:
            def execute(self) -> str:
                return "executed"

        obj = ValidExecutable()
        obj_id = hex(id(obj))

        registry_class.register_artifact(obj_id, obj)
        assert registry_class.has_artifact(obj)
        assert registry_class.get_artifact(obj_id) is obj

    def test_register_invalid_instance(self, registry_class):
        """Test that a non-conforming instance cannot be registered in strict mode."""

        class InvalidObject:
            def wrong_method(self) -> str:
                return "wrong"

        obj = InvalidObject()
        obj_id = hex(id(obj))

        with pytest.raises(ConformanceError):
            registry_class.register_artifact(obj_id, obj)

        # Should not be registered
        assert not registry_class.has_artifact(obj)
        with pytest.raises(RegistryError):
            registry_class.get_artifact(obj_id)

    def test_validate_instance(self, registry_class):
        """Test that validate_artifact correctly validates instances in strict mode."""

        class ValidExecutable:
            def execute(self) -> str:
                return "executed"

        class InvalidObject:
            def wrong_method(self) -> str:
                return "wrong"

        # Valid instance should pass validation
        valid_obj = ValidExecutable()
        inner_artifact = registry_class._intern_artifact(valid_obj)
        assert registry_class._extern_artifact(inner_artifact) is valid_obj

        # Invalid instance should fail validation
        invalid_obj = InvalidObject()
        with pytest.raises(ConformanceError):
            registry_class._intern_artifact(invalid_obj)


# -------------------------------------------------------------------
# Test class for abstract ObjectRegistry
# -------------------------------------------------------------------


class TestAbstractObjectRegistry:
    """Tests for ObjectRegistry in abstract mode."""

    @pytest.fixture
    def registry_class(self):
        """Create an abstract registry for objects."""

        class AbstractObjectReg(ObjectRegistry[object], abstract=True):
            pass

        return AbstractObjectReg

    def test_register_inheriting_instance(self, registry_class):
        """Test that an instance inheriting from the registry class can be registered."""

        class InheritingClass(registry_class):
            pass

        obj = InheritingClass()
        obj_id = hex(id(obj))

        registry_class.register_artifact(obj_id, obj)
        assert registry_class.has_artifact(obj)
        assert registry_class.get_artifact(obj_id) is obj

    def test_register_non_inheriting_instance(self, registry_class):
        """Test that an instance not inheriting from the registry cannot be registered."""

        class NonInheritingClass:
            pass

        obj = NonInheritingClass()
        obj_id = hex(id(obj))

        with pytest.raises(InheritanceError):
            registry_class.register_artifact(obj_id, obj)

        # Should not be registered
        assert not registry_class.has_artifact(obj)
        with pytest.raises(RegistryError):
            registry_class.get_artifact(obj_id)


# -------------------------------------------------------------------
# Test class for WeakRef behavior
# -------------------------------------------------------------------


class TestWeakRefBehavior:
    """Tests for weak reference behavior in ObjectRegistry."""

    @pytest.fixture
    def registry_class(self):
        """Create a registry for generic objects."""
        return new_class("WeakRefReg", (ObjectRegistry[object],))

    def test_weakref_garbage_collection(self, registry_class):
        """Test that objects are garbage collected when no longer referenced."""

        class GarbageObject:
            pass

        # Create an object and get its ID
        obj = GarbageObject()
        obj_id = hex(id(obj))

        # Register the object
        registry_class.register_artifact(obj_id, obj)

        # Verify it's registered
        assert registry_class.has_artifact(obj)
        assert registry_class.get_artifact(obj_id) is obj

        # Create a weak reference to check if it's collected
        obj_ref = weakref.ref(obj)

        # Remove the strong reference
        del obj

        # Force garbage collection
        gc.collect()

        # The weak reference should now be None
        assert obj_ref() is None

        # The registry should no longer contain the object
        with pytest.raises(RegistryError):
            registry_class.get_artifact(obj_id)

    def test_multiple_references(self, registry_class):
        """Test that objects with multiple references are not collected."""

        class RefObject:
            pass

        # Create an object and register it
        obj1 = RefObject()
        obj_id = hex(id(obj1))
        registry_class.register_artifact(obj_id, obj1)

        # Create another reference to the same object
        obj2 = obj1

        # Create a weak reference
        obj_ref = weakref.ref(obj1)

        # Remove one reference
        del obj1

        # Force garbage collection
        gc.collect()

        # The weak reference should still be valid
        assert obj_ref() is obj2

        # The registry should still contain the object
        assert registry_class.get_artifact(obj_id) is obj2
