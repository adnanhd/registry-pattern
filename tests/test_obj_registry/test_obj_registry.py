import pytest
from types import new_class
from registry import ObjectConfigMap, RegistryError


class TestObject:
    """A simple class for testing ObjectConfigMap."""

    __test__ = False  # Prevent pytest from treating this as a test class.

    def __init__(self, name: str):
        self.name = name


@pytest.fixture
def ObjConfigMap():
    """
    Fixture for ObjectConfigMap.
    Creates a new registry class (named 'ConfMap') specialized for TestObject.
    """
    return new_class("ConfMap", (ObjectConfigMap[TestObject],))


def test_register_instance(ObjConfigMap: ObjectConfigMap):
    """
    Test registering an object with configuration.

    The object is registered with a configuration dictionary.
    We verify that the object (not the config) is present in the registry,
    and that the retrieved configuration matches what was registered.
    """
    obj = TestObject("test")
    config = {"param1": 42, "param2": "value"}
    ObjConfigMap.register_instance(obj, config)

    # Verify that the object is used as the key in the registry.
    assert ObjConfigMap.artifact_id_exists(obj)
    # Verify that the configuration associated with the object is correct.
    assert ObjConfigMap.get_artifact(obj) == config


def test_unregister_instance(ObjConfigMap: ObjectConfigMap):
    """
    Test unregistering an object.

    After registration, the object is unregistered, so it should no longer
    be present in the registry.
    """
    obj = TestObject("test")
    config = {"param1": 42, "param2": "value"}
    ObjConfigMap.register_instance(obj, config)
    ObjConfigMap.unregister_instance(obj)

    # After unregistration, the object should not be found in the registry.
    assert not ObjConfigMap.artifact_id_exists(obj)


def test_register_duplicate_instance(ObjConfigMap: ObjectConfigMap):
    """
    Test registering the same object multiple times.

    Registering the same object with a different configuration should raise
    a RegistryError, and the original configuration should remain unchanged.
    """
    obj = TestObject("test")
    config1 = {"param1": 42}
    config2 = {"param1": 43}

    # Register the object with the initial configuration.
    ObjConfigMap.register_instance(obj, config1)
    with pytest.raises(RegistryError):
        # Attempting to register the same object with a new configuration should fail.
        ObjConfigMap.register_instance(obj, config2)

    # Verify that the object is registered with the original configuration.
    assert ObjConfigMap.artifact_id_exists(obj)
    assert ObjConfigMap.get_artifact(obj) == config1


def test_validate_artifact(ObjConfigMap: ObjectConfigMap):
    """
    Test validating an item (configuration dictionary).

    A valid configuration (a dictionary) should pass validation,
    whereas a non-dictionary value should raise a TypeError.
    """
    config = {"param1": 42}
    valid_config = ObjConfigMap.validate_artifact(config)
    assert valid_config == config

    with pytest.raises(TypeError):
        ObjConfigMap.validate_artifact("invalid config")
