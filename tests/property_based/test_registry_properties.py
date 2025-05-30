# tests/property_based/test_registry_properties.py
"""Property-based tests for registry pattern using Hypothesis."""

import gc
import uuid
import weakref
import hypothesis.strategies as st
from hypothesis import given, assume, example, settings, Verbosity
import pytest
from typing import Any, Dict, List, Type, Union

from registry import ObjectRegistry, FunctionalRegistry, ConfigRegistry
from registry.core._validator import ValidationError
from registry.mixin import RegistryError

# Custom test objects that support weak references
class TestObject:
    """A simple test object that supports weak references."""
    def __init__(self, value):
        self.value = value
    
    def __eq__(self, other):
        return isinstance(other, TestObject) and self.value == other.value
    
    def __hash__(self):
        return hash(self.value)
    
    def __repr__(self):
        return f"TestObject({self.value!r})"

class TestKey:
    """A simple test key that supports weak references."""
    def __init__(self, key):
        self.key = key
    
    def __eq__(self, other):
        return isinstance(other, TestKey) and self.key == other.key
    
    def __hash__(self):
        return hash(self.key)
    
    def __repr__(self):
        return f"TestKey({self.key!r})"

# Strategies for generating test data
@st.composite
def test_keys(draw):
    """Generate test keys that work with weak references."""
    key_value = draw(st.one_of(
        st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
        st.integers(min_value=1, max_value=1000),  # Avoid 0 to prevent confusion
    ))
    return TestKey(key_value)

@st.composite
def test_objects(draw):
    """Generate test objects that work with weak references."""
    obj_value = draw(st.one_of(
        st.text(max_size=50),
        st.integers(min_value=-100, max_value=100),
        st.booleans(),
    ))
    return TestObject(obj_value)

@st.composite
def simple_hashable_keys(draw):
    """Generate simple hashable keys for basic tests."""
    return draw(st.one_of(
        st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isalnum()),
        st.integers(min_value=1, max_value=1000),
        st.tuples(st.text(min_size=1, max_size=10).filter(lambda x: x.isalnum()), 
                  st.integers(min_value=1, max_value=100)),
    ))

def create_isolated_registry():
    """Create a completely isolated ObjectRegistry."""
    # Use a unique module name to ensure complete isolation
    module_id = str(uuid.uuid4()).replace('-', '')
    
    # Create a completely fresh class with no shared state
    class IsolatedObjectRegistry(ObjectRegistry[object]):
        _repository = {}  # Fresh repository
        _strict = False
        _abstract = False
        _strict_weakref = False
        
        @classmethod
        def _get_mapping(cls):
            # Ensure we always return the instance repository
            return cls._repository
    
    # Give it a unique name
    IsolatedObjectRegistry.__name__ = f"IsolatedRegistry_{module_id}"
    IsolatedObjectRegistry.__qualname__ = f"IsolatedRegistry_{module_id}"
    
    return IsolatedObjectRegistry

class TestRegistryProperties:
    """Property-based tests for registry core functionality."""
    
    @given(key=test_keys(), obj=test_objects())
    @settings(max_examples=30, deadline=None)
    def test_register_get_roundtrip(self, key, obj):
        """Property: Whatever we register, we should be able to get back."""
        registry = create_isolated_registry()
        
        registry.register_artifact(key, obj)
        retrieved = registry.get_artifact(key)
        assert retrieved == obj
    
    @given(key=test_keys(), obj=test_objects())
    @settings(max_examples=30)
    def test_register_has_identifier(self, key, obj):
        """Property: After registration, has_identifier should return True."""
        registry = create_isolated_registry()
        
        registry.register_artifact(key, obj)
        assert registry.has_identifier(key)
    
    @given(key=test_keys(), obj=test_objects())
    @settings(max_examples=30)
    def test_register_has_artifact(self, key, obj):
        """Property: After registration, has_artifact should return True."""
        registry = create_isolated_registry()
        
        registry.register_artifact(key, obj)
        assert registry.has_artifact(obj)
    
    @given(key=test_keys(), obj=test_objects())
    @settings(max_examples=20)
    def test_unregister_removes_artifact(self, key, obj):
        """Property: After unregistration, artifact should not be retrievable."""
        registry = create_isolated_registry()
        
        registry.register_artifact(key, obj)
        registry.unregister_identifier(key)
        
        assert not registry.has_identifier(key)
        with pytest.raises(RegistryError):
            registry.get_artifact(key)
    
    @given(key_obj_pairs=st.lists(
        st.tuples(test_keys(), test_objects()), 
        min_size=1, max_size=5, 
        unique_by=lambda x: x[0]  # Ensure unique keys
    ))
    @settings(max_examples=10)
    def test_multiple_registrations(self, key_obj_pairs):
        """Property: Multiple registrations should all be retrievable."""
        registry = create_isolated_registry()
        
        for key, obj in key_obj_pairs:
            registry.register_artifact(key, obj)
        
        for key, expected_obj in key_obj_pairs:
            retrieved = registry.get_artifact(key)
            assert retrieved == expected_obj
    
    @given(key=test_keys(), obj=test_objects())
    @settings(max_examples=20)
    def test_duplicate_registration_fails(self, key, obj):
        """Property: Registering the same key twice should fail."""
        registry = create_isolated_registry()
        
        registry.register_artifact(key, obj)
        
        with pytest.raises(RegistryError):
            registry.register_artifact(key, obj)
    
    def test_invalid_key_registration_fails(self):
        """Test that registering with invalid key should fail."""
        registry = create_isolated_registry()
        
        invalid_keys = [
            [1, 2, 3],      # list
            {'a': 1},       # dict
            {1, 2, 3},      # set
        ]
        
        for invalid_key in invalid_keys:
            with pytest.raises((ValidationError, TypeError)):
                registry.register_artifact(invalid_key, TestObject("value"))
    
    @given(key=test_keys())
    @settings(max_examples=20)
    def test_get_nonexistent_key_fails(self, key):
        """Property: Getting non-existent key should fail."""
        registry = create_isolated_registry()
        
        with pytest.raises(RegistryError):
            registry.get_artifact(key)


class TestSimpleRegistry:
    """Simple non-property tests to verify basic functionality."""
    
    def test_basic_registration(self):
        """Test basic registration with simple values."""
        registry = create_isolated_registry()
        
        key = TestKey("simple_key")
        obj = TestObject("simple_value")
        
        registry.register_artifact(key, obj)
        
        assert registry.has_identifier(key)
        assert registry.has_artifact(obj)
        assert registry.get_artifact(key) == obj
    
    def test_string_key_registration(self):
        """Test registration with string keys (which can't be weakly referenced)."""
        registry = create_isolated_registry()
        
        # Test with simple string key and object value
        registry.register_artifact("string_key", TestObject("value"))
        assert registry.get_artifact("string_key") == TestObject("value")
    
    def test_integer_key_registration(self):
        """Test registration with integer keys."""
        registry = create_isolated_registry()
        
        # Test with integer key and object value
        registry.register_artifact(42, TestObject("value"))
        assert registry.get_artifact(42) == TestObject("value")
    
    def test_unregistration(self):
        """Test unregistration works correctly."""
        registry = create_isolated_registry()
        
        key = TestKey("temp_key")
        obj = TestObject("temp_value")
        
        registry.register_artifact(key, obj)
        assert registry.has_identifier(key)
        
        registry.unregister_identifier(key)
        assert not registry.has_identifier(key)
        
        with pytest.raises(RegistryError):
            registry.get_artifact(key)
    
    def test_clear_registry(self):
        """Test clearing registry."""
        registry = create_isolated_registry()
        
        # Register several items
        for i in range(3):
            registry.register_artifact(TestKey(f"key_{i}"), TestObject(f"value_{i}"))
        
        assert len(registry._repository) == 3
        
        registry.clear_artifacts()
        
        assert len(registry._repository) == 0


class TestWeakReferenceHandling:
    """Test weak reference handling specifically."""
    
    def test_weak_reference_compatible_objects(self):
        """Test that weak reference compatible objects work correctly."""
        registry = create_isolated_registry()
        
        key = TestKey("weak_key")
        obj = TestObject("weak_value")
        
        # These should work with weak references
        registry.register_artifact(key, obj)
        
        assert registry.has_identifier(key)
        assert registry.has_artifact(obj)
        assert registry.get_artifact(key) == obj
    
    def test_non_weak_reference_objects(self):
        """Test that objects that can't be weakly referenced are handled."""
        registry = create_isolated_registry()
        
        # Simple values that can't be weakly referenced
        test_cases = [
            ("str_key", "string_value"),
            ("int_key", 42),
            ("bool_key", True),
            ("none_key", None),
        ]
        
        for key, value in test_cases:
            # These should work even though they can't be weakly referenced
            registry.register_artifact(key, value)
            assert registry.get_artifact(key) == value


class TestPerformanceBasic:
    """Basic performance tests without property-based testing."""
    
    def test_registration_performance(self):
        """Test basic registration performance."""
        registry = create_isolated_registry()
        
        import time
        
        # Register 100 items
        num_items = 100
        
        start_time = time.time()
        for i in range(num_items):
            key = f"perf_key_{i}_{uuid.uuid4()}"  # Unique keys
            obj = TestObject(f"perf_value_{i}")
            registry.register_artifact(key, obj)
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_item = total_time / num_items
        
        # Should be very fast
        assert time_per_item < 0.01, f"Registration too slow: {time_per_item:.6f}s per item"
    
    def test_retrieval_performance(self):
        """Test basic retrieval performance."""
        registry = create_isolated_registry()
        
        import time
        
        # Register items first
        keys = []
        for i in range(50):
            key = f"retr_key_{i}_{uuid.uuid4()}"
            obj = TestObject(f"retr_value_{i}")
            registry.register_artifact(key, obj)
            keys.append(key)
        
        # Test retrieval performance
        test_key = keys[25]  # Middle key
        
        start_time = time.time()
        for _ in range(100):
            registry.get_artifact(test_key)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Should be very fast
        assert avg_time < 0.001, f"Retrieval too slow: {avg_time:.6f}s per retrieval"


class TestRegistryInvariants:
    """Test registry invariants with simple scenarios."""
    
    def test_size_invariant(self):
        """Test that registry size matches expectations."""
        registry = create_isolated_registry()
        
        # Start empty
        assert len(registry._repository) == 0
        
        # Add items
        items = []
        for i in range(5):
            key = TestKey(f"inv_key_{i}")
            obj = TestObject(f"inv_value_{i}")
            registry.register_artifact(key, obj)
            items.append((key, obj))
        
        assert len(registry._repository) == 5
        
        # Remove some items
        for i in range(0, 5, 2):  # Remove even indices
            registry.unregister_identifier(items[i][0])
        
        assert len(registry._repository) == 3
        
        # Clear all
        registry.clear_artifacts()
        assert len(registry._repository) == 0
    
    def test_consistency_invariant(self):
        """Test that registry state remains consistent."""
        registry = create_isolated_registry()
        
        items = []
        for i in range(3):
            key = TestKey(f"cons_key_{i}")
            obj = TestObject(f"cons_value_{i}")
            registry.register_artifact(key, obj)
            items.append((key, obj))
        
        # All items should be retrievable
        for key, expected_obj in items:
            assert registry.has_identifier(key)
            assert registry.has_artifact(expected_obj)
            assert registry.get_artifact(key) == expected_obj


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_duplicate_key_error(self):
        """Test that duplicate keys are properly rejected."""
        registry = create_isolated_registry()
        
        key = TestKey("dup_key")
        obj1 = TestObject("value1")
        obj2 = TestObject("value2")
        
        # First registration should succeed
        registry.register_artifact(key, obj1)
        
        # Second registration should fail
        with pytest.raises(RegistryError):
            registry.register_artifact(key, obj2)
        
        # First value should still be there
        assert registry.get_artifact(key) == obj1
    
    def test_nonexistent_key_error(self):
        """Test that accessing non-existent keys fails properly."""
        registry = create_isolated_registry()
        
        nonexistent_key = TestKey("nonexistent")
        
        # Should fail for has_identifier
        assert not registry.has_identifier(nonexistent_key)
        
        # Should fail for get_artifact
        with pytest.raises(RegistryError):
            registry.get_artifact(nonexistent_key)
        
        # Should fail for unregister
        with pytest.raises(RegistryError):
            registry.unregister_identifier(nonexistent_key)


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--tb=short"])
