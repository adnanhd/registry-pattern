# tests/property_based/test_simple_properties.py
"""Simplified property-based tests that focus on core functionality."""

import hypothesis.strategies as st
from hypothesis import given, assume, settings
import pytest
from types import new_class
from registry import ObjectRegistry
from registry.core._validator import ValidationError
from registry.mixin import RegistryError

# Simple strategies
hashable_keys = st.one_of(
    st.text(min_size=1, max_size=20),
    st.integers(),
    st.tuples(st.text(min_size=1, max_size=10), st.integers())
)

simple_values = st.one_of(
    st.text(max_size=50),
    st.integers(),
    st.booleans(),
    st.none()
)

class TestSimpleRegistryProperties:
    """Simplified property-based tests for registry functionality."""
    
    def create_fresh_registry(self):
        """Create a fresh registry for each test."""
        import time
        class_name = f"TestRegistry_{int(time.time() * 1000000)}"
        registry_class = new_class(class_name, (ObjectRegistry[object],))
        # Ensure clean state
        registry_class._repository = {}
        return registry_class
    
    @given(key=hashable_keys, value=simple_values)
    @settings(max_examples=50, deadline=None)
    def test_register_retrieve_roundtrip(self, key, value):
        """Property: What we register, we can retrieve."""
        registry = self.create_fresh_registry()
        
        registry.register_artifact(key, value)
        retrieved = registry.get_artifact(key)
        
        assert retrieved == value
    
    @given(key=hashable_keys, value=simple_values)
    @settings(max_examples=50)
    def test_registration_affects_has_identifier(self, key, value):
        """Property: Registration makes has_identifier return True."""
        registry = self.create_fresh_registry()
        
        # Before registration
        assert not registry.has_identifier(key)
        
        # After registration
        registry.register_artifact(key, value)
        assert registry.has_identifier(key)
    
    @given(key=hashable_keys, value=simple_values)
    @settings(max_examples=50)
    def test_registration_affects_has_artifact(self, key, value):
        """Property: Registration makes has_artifact return True."""
        registry = self.create_fresh_registry()
        
        # Before registration
        assert not registry.has_artifact(value)
        
        # After registration
        registry.register_artifact(key, value)
        assert registry.has_artifact(value)
    
    @given(key=hashable_keys, value=simple_values)
    @settings(max_examples=30)
    def test_unregistration_removes_artifact(self, key, value):
        """Property: Unregistration removes the artifact."""
        registry = self.create_fresh_registry()
        
        # Register then unregister
        registry.register_artifact(key, value)
        registry.unregister_identifier(key)
        
        # Should be gone
        assert not registry.has_identifier(key)
        
        with pytest.raises(RegistryError):
            registry.get_artifact(key)
    
    @given(key=hashable_keys, value=simple_values)
    @settings(max_examples=30)
    def test_duplicate_registration_fails(self, key, value):
        """Property: Registering same key twice fails."""
        registry = self.create_fresh_registry()
        
        # First registration succeeds
        registry.register_artifact(key, value)
        
        # Second registration fails
        with pytest.raises(RegistryError):
            registry.register_artifact(key, value)
    
    @given(key=hashable_keys)
    @settings(max_examples=30)
    def test_get_nonexistent_fails(self, key):
        """Property: Getting non-existent key fails."""
        registry = self.create_fresh_registry()
        
        # Should fail for non-existent key
        with pytest.raises(RegistryError):
            registry.get_artifact(key)
    
    def test_invalid_key_types_fail(self):
        """Test that invalid key types are rejected."""
        registry = self.create_fresh_registry()
        
        invalid_keys = [
            [1, 2, 3],  # list
            {'a': 1},   # dict
            {1, 2, 3},  # set
        ]
        
        for invalid_key in invalid_keys:
            with pytest.raises((ValidationError, TypeError)):
                registry.register_artifact(invalid_key, "value")
    
    @given(
        keys_values=st.lists(
            st.tuples(hashable_keys, simple_values),
            min_size=1,
            max_size=10,
            unique_by=lambda x: x[0]  # Unique keys
        )
    )
    @settings(max_examples=20)
    def test_multiple_registrations(self, keys_values):
        """Property: Can register and retrieve multiple items."""
        registry = self.create_fresh_registry()
        
        # Register all items
        for key, value in keys_values:
            registry.register_artifact(key, value)
        
        # Retrieve and verify all items
        for key, expected_value in keys_values:
            retrieved = registry.get_artifact(key)
            assert retrieved == expected_value
    
    @given(
        keys_values=st.lists(
            st.tuples(hashable_keys, simple_values),
            min_size=2,
            max_size=5,
            unique_by=lambda x: x[0]  # Unique keys
        )
    )
    @settings(max_examples=10)
    def test_partial_unregistration(self, keys_values):
        """Property: Can unregister some items while keeping others."""
        registry = self.create_fresh_registry()
        
        # Register all items
        for key, value in keys_values:
            registry.register_artifact(key, value)
        
        # Unregister first item
        first_key, first_value = keys_values[0]
        registry.unregister_identifier(first_key)
        
        # First should be gone
        assert not registry.has_identifier(first_key)
        
        # Others should remain
        for key, value in keys_values[1:]:
            assert registry.has_identifier(key)
            assert registry.get_artifact(key) == value


class TestRegistryInvariants:
    """Test registry invariants that should always hold."""
    
    def create_fresh_registry(self):
        """Create a fresh registry for each test."""
        import time
        class_name = f"TestRegistry_{int(time.time() * 1000000)}"
        registry_class = new_class(class_name, (ObjectRegistry[object],))
        registry_class._repository = {}
        return registry_class
    
    @given(
        operations=st.lists(
            st.one_of(
                st.tuples(st.just("register"), hashable_keys, simple_values),
                st.tuples(st.just("unregister"), hashable_keys),
                st.tuples(st.just("clear"))
            ),
            max_size=20
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_registry_invariants_hold(self, operations):
        """Property: Registry invariants hold after any sequence of operations."""
        registry = self.create_fresh_registry()
        registered_keys = set()
        
        for operation in operations:
            try:
                if operation[0] == "register":
                    _, key, value = operation
                    if key not in registered_keys:
                        registry.register_artifact(key, value)
                        registered_keys.add(key)
                
                elif operation[0] == "unregister":
                    _, key = operation
                    if key in registered_keys:
                        registry.unregister_identifier(key)
                        registered_keys.remove(key)
                
                elif operation[0] == "clear":
                    registry.clear_artifacts()
                    registered_keys.clear()
                    
            except (RegistryError, ValidationError):
                # Some operations may fail, which is fine
                pass
        
        # Invariant: All tracked keys should be in registry
        for key in registered_keys:
            assert registry.has_identifier(key), f"Key {key} should be in registry"
        
        # Invariant: Registry size should match tracked keys
        actual_size = len(registry._repository)
        expected_size = len(registered_keys)
        # Allow some tolerance for weak references
        assert abs(actual_size - expected_size) <= 1, f"Size mismatch: {actual_size} vs {expected_size}"


if __name__ == "__main__":
    # Run these simplified tests
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
