import pytest
from types import new_class
from registry import ObjectRegistry, RegistryError

# -----------------------------------------------------------------------------
# Test Class Definition
# -----------------------------------------------------------------------------


class MyObject:
    """A simple class for testing the ObjectRegistry."""

    def __init__(self, name: str):
        self.name = name


# -----------------------------------------------------------------------------
# Fixture for ObjectRegistry
# -----------------------------------------------------------------------------


@pytest.fixture
def ObjRegistry():
    """
    Fixture for creating an ObjectRegistry specialized for MyObject.

    The registry is dynamically created using new_class and is based on
    ObjectRegistry[MyObject].
    """
    return new_class("ObjRegistry", (ObjectRegistry[MyObject],), {})


# -----------------------------------------------------------------------------
# Test Cases for ObjectRegistry
# -----------------------------------------------------------------------------


def test_register_instance(ObjRegistry: ObjectRegistry):
    """
    Test that an instance is registered correctly.

    The instance should be stored as both the key and value in the registry.
    """
    obj = MyObject("test")
    ObjRegistry.register_instance(obj)

    # Verify that the object is present in the registry.
    assert ObjRegistry.artifact_exists(obj)
    # Retrieve the registered item; it should be the same as the object.
    assert ObjRegistry.get_artifact(obj) == obj


def test_unregister_instance(ObjRegistry: ObjectRegistry):
    """
    Test that an instance can be unregistered.

    After unregistration, the object should no longer be in the registry.
    """
    obj = MyObject("test")
    ObjRegistry.register_instance(obj)
    ObjRegistry.unregister_instance(obj)

    # Verify that the object is no longer registered.
    assert not ObjRegistry.artifact_exists(obj)


def test_register_duplicate_instance(ObjRegistry: ObjectRegistry):
    """
    Test that attempting to register the same instance twice raises a RegistryError.

    The registry should not allow duplicate registrations; the original configuration
    should remain unchanged.
    """
    obj = MyObject("test")
    ObjRegistry.register_instance(obj)

    with pytest.raises(RegistryError):
        ObjRegistry.register_instance(obj)  # Attempt duplicate registration.

    # Confirm that the object is still registered.
    assert ObjRegistry.artifact_exists(obj)
    assert ObjRegistry.get_artifact(obj) == obj


def test_validate_artifact_id(ObjRegistry: ObjectRegistry):
    """
    Test that the validate_artifact_id method returns the correct key.

    The default key generation returns the hexadecimal representation of the
    object's id.
    """
    obj = MyObject("test")
    key = ObjRegistry.validate_artifact_id(obj)
    assert key == hex(id(obj))


def test_register_class_instances(ObjRegistry: ObjectRegistry):
    """
    Test that using register_class_instances correctly wraps a class for automatic
    instance registration.

    Instances created from the wrapped class should be automatically tracked in
    the registry.
    """

    @ObjRegistry.register_class_instances
    class MySubObject(MyObject):
        pass

    instance = MySubObject("example")

    # Verify that the instance is automatically registered.
    assert ObjRegistry.artifact_exists(instance)
    assert ObjRegistry.get_artifact(instance) == instance


def test_lookup_unregistered_instance(ObjRegistry: ObjectRegistry):
    """
    Test that attempting to retrieve an unregistered instance raises RegistryError.

    This ensures that get_artifact does not return a value for keys that are
    not present in the registry.
    """
    obj = MyObject("unregistered")
    with pytest.raises(RegistryError):
        ObjRegistry.get_artifact(obj)
