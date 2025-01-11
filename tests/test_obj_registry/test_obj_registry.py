import pytest
from types import new_class
from registry import ObjectRegistry, RegistryError


class MyObject:
    """A simple class for testing the ObjectRegistry."""

    def __init__(self, name: str):
        self.name = name


@pytest.fixture
def ObjRegistry():
    """Fixture for ObjectRegistry."""
    return new_class("ObjRegistry", (ObjectRegistry[MyObject],), {})


def test_register_instance(ObjRegistry: ObjectRegistry):
    """Test registering an instance."""
    obj = MyObject("test")
    ObjRegistry.register_instance(obj)
    assert ObjRegistry.has_registry_item(obj)
    assert ObjRegistry.get_registry_item(obj) == obj


def test_unregister_instance(ObjRegistry: ObjectRegistry):
    """Test unregistering an instance."""
    obj = MyObject("test")
    ObjRegistry.register_instance(obj)
    ObjRegistry.unregister_instance(obj)
    assert not ObjRegistry.has_registry_item(obj)


def test_register_duplicate_instance(ObjRegistry: ObjectRegistry):
    """Test that duplicate instances are handled correctly."""
    obj = MyObject("test")
    ObjRegistry.register_instance(obj)
    with pytest.raises(RegistryError):
        ObjRegistry.register_instance(obj)  # Registering the same instance again
    assert ObjRegistry.has_registry_item(obj)
    assert ObjRegistry.get_registry_item(obj) == obj


def test_validate_key(ObjRegistry: ObjectRegistry):
    """Test key validation."""
    obj = MyObject("test")
    key = ObjRegistry.validate_key(obj)
    assert key == hex(id(obj))


def test_register_class_instances(ObjRegistry: ObjectRegistry):
    """Test registering all instances of a class."""

    @ObjRegistry.register_class_instances
    class MySubObject(MyObject):
        pass

    instance = MySubObject("example")
    assert ObjRegistry.has_registry_item(instance)
    assert ObjRegistry.get_registry_item(instance) == instance
