import pytest
from types import new_class
from registry import ObjectConfigMap
from registry.base import RegistryError


class TestObject:
    """A simple class for testing ObjectConfigMap."""

    __test__ = False

    def __init__(self, name: str):
        self.name = name


@pytest.fixture
def ObjConfigMap():
    """Fixture for ObjectConfigMap."""
    return new_class("ConfMap", (ObjectConfigMap[TestObject],))


def test_register_instance(ObjConfigMap: ObjectConfigMap):
    """Test registering an object with configuration."""
    obj = TestObject("test")
    config = {"param1": 42, "param2": "value"}
    ObjConfigMap.register_instance(obj, config)

    assert ObjConfigMap.has_registry_item(config)
    assert ObjConfigMap.get_registry_item(obj) == config


def test_unregister_instance(ObjConfigMap: ObjectConfigMap):
    """Test unregistering an object."""
    obj = TestObject("test")
    config = {"param1": 42, "param2": "value"}
    ObjConfigMap.register_instance(obj, config)
    ObjConfigMap.unregister_instance(obj)

    assert not ObjConfigMap.has_registry_item(config)


def test_register_duplicate_instance(ObjConfigMap: ObjectConfigMap):
    """Test registering the same object multiple times."""
    obj = TestObject("test")
    config1 = {"param1": 42}
    config2 = {"param1": 43}

    ObjConfigMap.register_instance(obj, config1)
    with pytest.raises(RegistryError):
        ObjConfigMap.register_instance(obj, config2)  # Overwrite with new config

    assert ObjConfigMap.has_registry_item(config1)
    assert not ObjConfigMap.has_registry_item(config2)
    assert ObjConfigMap.get_registry_item(obj) == config1


def test_validate_item(ObjConfigMap: ObjectConfigMap):
    """Test validating an item."""
    config = {"param1": 42}
    valid_config = ObjConfigMap.validate_item(config)

    assert valid_config == config

    with pytest.raises(TypeError):
        ObjConfigMap.validate_item("invalid config")
