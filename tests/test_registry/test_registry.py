import pytest
from functools import lru_cache
from typing import Any, Dict

# Import the classes and exceptions from your registry module.
# Adjust the import paths according to your project structure.
from registry import RegistryError, RegistryLookupError, ValidationError, Registry, MutableRegistry

# -------------------------------------------------------------------
# Define concrete registry classes for testing.
# -------------------------------------------------------------------


class TestMutableRegistry(MutableRegistry[str, int]):
    """
    A simple mutable registry for testing that stores integer values
    keyed by strings.
    """

    # Initialize the repository as an empty dict.
    _repository = {}


class TestValidationRegistry(MutableRegistry[str, int]):
    """
    A registry that validates items such that negative integers
    are rejected.
    """

    _repository = {}

    @classmethod
    def validate_item(cls, value: int) -> int:
        if value < 0:
            raise ValidationError("Negative value not allowed")
        return value


# -------------------------------------------------------------------
# Fixtures for test isolation.
# -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_registries():
    """
    This fixture clears the registries (and the lru_cache of get_registry_item)
    before each test to ensure tests run independently.
    """
    TestMutableRegistry._repository.clear()
    TestValidationRegistry._repository.clear()
    TestMutableRegistry.get_registry_item.cache_clear()
    TestValidationRegistry.get_registry_item.cache_clear()


# -------------------------------------------------------------------
# Tests for basic registry functionality.
# -------------------------------------------------------------------


def test_add_and_lookup():
    """Test adding an item and looking it up by key."""
    # Add an item using the class method.
    TestMutableRegistry.add_registry_item("a", 1)
    # The key should be in the registry.
    assert TestMutableRegistry.has_registry_key("a")
    # Using the class method lookup.
    assert TestMutableRegistry.get_registry_item("a") == 1

    # Also test the instance __getitem__ (which calls get_registry_item).
    instance = TestMutableRegistry()
    assert instance["a"] == 1


def test_duplicate_registration():
    """Test that adding an item with a duplicate key raises RegistryError."""
    TestMutableRegistry.add_registry_item("a", 1)
    with pytest.raises(RegistryError) as excinfo:
        TestMutableRegistry.add_registry_item("a", 2)
    # Check that the error message contains an appropriate message.
    assert "already registered" in str(excinfo.value)


def test_lookup_nonexistent():
    """Test that looking up a key that has not been registered raises RegistryLookupError."""
    with pytest.raises(RegistryLookupError) as excinfo:
        TestMutableRegistry.get_registry_item("nonexistent")
    assert "not registered" in str(excinfo.value)


def test_delete_registry_item():
    """Test that deleting an item removes it from the registry."""
    TestMutableRegistry.add_registry_item("a", 1)
    TestMutableRegistry.del_registry_item("a")
    # After deletion, the lookup should fail.
    with pytest.raises(RegistryLookupError):
        TestMutableRegistry.get_registry_item("a")
    # Also, __contains__ should report that the key is absent.
    assert not TestMutableRegistry.has_registry_key("a")


def test_keys_and_values():
    """Test that keys() and values() return the correct lists."""
    items = {"a": 1, "b": 2, "c": 3}
    for key, value in items.items():
        TestMutableRegistry.add_registry_item(key, value)
    keys = TestMutableRegistry.keys()
    values = TestMutableRegistry.values()
    # The keys and values should match what was added.
    assert set(keys) == set(items.keys())
    assert set(values) == set(items.values())
    # Also test the len() method.
    assert TestMutableRegistry.len() == len(items)


def test_clear_registry():
    """Test that clear_registry() empties the registry."""
    TestMutableRegistry.add_registry_item("a", 1)
    TestMutableRegistry.add_registry_item("b", 2)
    TestMutableRegistry.clear_registry()
    assert TestMutableRegistry.len() == 0
    assert TestMutableRegistry.keys() == []


def test_setitem_and_delitem():
    """
    Test the instance methods __setitem__ and __delitem__ (which call
    add_registry_item and del_registry_item, respectively).
    """
    instance = TestMutableRegistry()
    # Set an item using the [] assignment.
    instance["x"] = 42
    assert instance["x"] == 42
    # Delete the item using del.
    del instance["x"]
    with pytest.raises(RegistryLookupError):
        _ = instance["x"]


def test_has_registry_key_and_item():
    """Test the has_registry_key() and has_registry_item() methods."""
    TestMutableRegistry.add_registry_item("key1", 100)
    assert TestMutableRegistry.has_registry_key("key1")
    assert TestMutableRegistry.has_registry_item(100)
    # Keys or values that were not registered should return False.
    assert not TestMutableRegistry.has_registry_key("nonexistent")
    assert not TestMutableRegistry.has_registry_item(200)


# -------------------------------------------------------------------
# Tests for custom item validation.
# -------------------------------------------------------------------


def test_validate_item_error():
    """
    Test that the custom validation in TestValidationRegistry causes
    negative values to be rejected.
    """
    with pytest.raises(ValidationError):
        TestValidationRegistry.add_registry_item("neg", -10)
    # Ensure that the key "neg" was not added.
    assert not TestValidationRegistry.has_registry_key("neg")


def test_validate_item_accept():
    """
    Test that a positive value passes the validation in TestValidationRegistry.
    """
    TestValidationRegistry.add_registry_item("pos", 10)
    assert TestValidationRegistry.get_registry_item("pos") == 10


# -------------------------------------------------------------------
# Additional tests.
# -------------------------------------------------------------------


def test_contains_operator():
    """
    Test that the __contains__ operator works on the registry instance.
    """
    TestMutableRegistry.add_registry_item("a", 1)
    instance = TestMutableRegistry()
    assert "a" in instance
    assert "b" not in instance


def test_cache_behavior():
    """
    Test the caching behavior of get_registry_item.

    After the initial lookup, modify the underlying repository directly
    and verify that the cached value remains until the cache is cleared.
    """
    TestMutableRegistry.add_registry_item("a", 1)
    # Populate the cache.
    value1 = TestMutableRegistry.get_registry_item("a")
    # Modify the repository directly.
    TestMutableRegistry._repository["a"] = 42
    # The cached value should still be returned.
    value2 = TestMutableRegistry.get_registry_item("a")
    assert value1 == value2 == 1
    # Now clear the cache.
    TestMutableRegistry.get_registry_item.cache_clear()
    value3 = TestMutableRegistry.get_registry_item("a")
    assert value3 == 42
