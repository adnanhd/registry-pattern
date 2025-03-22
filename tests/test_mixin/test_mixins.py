import pytest
from typing import Dict, Any

from registry.mixin import (
    RegistryError,
    RegistryAccessorMixin,
    RegistryMutatorMixin,
    ImmutableRegistryValidatorMixin,
    MutableRegistryValidatorMixin,
)


# -------------------------------------------------------------------
# Test classes for RegistryAccessorMixin
# -------------------------------------------------------------------


class TestRegistryAccessorMixin:
    """Tests for RegistryAccessorMixin."""

    class DummyAccessor(RegistryAccessorMixin[str, int]):
        """Concrete implementation of RegistryAccessorMixin for testing."""

        _repository: Dict[str, int] = {"a": 1, "b": 2, "c": 3}

        @classmethod
        def _get_mapping(cls) -> Dict[str, int]:
            return cls._repository

    def test_get_mapping(self):
        """Test that _get_mapping returns the correct repository."""
        assert self.DummyAccessor._get_mapping() == {"a": 1, "b": 2, "c": 3}

    def test_len_mapping(self):
        """Test that _len_mapping returns the correct repository size."""
        assert self.DummyAccessor._len_mapping() == 3

    def test_iter_mapping(self):
        """Test that _iter_mapping returns an iterator over keys."""
        keys = list(self.DummyAccessor._iter_mapping())
        assert set(keys) == {"a", "b", "c"}

    def test_get_artifact(self):
        """Test that _get_artifact retrieves the correct value."""
        assert self.DummyAccessor._get_artifact("b") == 2

    def test_get_artifact_missing(self):
        """Test that _get_artifact raises for missing keys."""
        with pytest.raises(RegistryError):
            self.DummyAccessor._get_artifact("z")

    def test_has_identifier(self):
        """Test that _has_identifier correctly checks key presence."""
        assert self.DummyAccessor._has_identifier("a")
        assert not self.DummyAccessor._has_identifier("z")

    def test_has_artifact(self):
        """Test that _has_artifact correctly checks value presence."""
        assert self.DummyAccessor._has_artifact(3)
        assert not self.DummyAccessor._has_artifact(4)

    def test_assert_presence_success(self):
        """Test that _assert_presence succeeds for present keys."""
        mapping = self.DummyAccessor._assert_presence("a")
        assert mapping["a"] == 1

    def test_assert_presence_failure(self):
        """Test that _assert_presence raises for missing keys."""
        with pytest.raises(RegistryError):
            self.DummyAccessor._assert_presence("z")


# -------------------------------------------------------------------
# Test classes for RegistryMutatorMixin
# -------------------------------------------------------------------


class TestRegistryMutatorMixin:
    """Tests for RegistryMutatorMixin."""

    class DummyMutator(RegistryMutatorMixin[str, int]):
        """Concrete implementation of RegistryMutatorMixin for testing."""

        _repository: Dict[str, int] = {"x": 10, "y": 20}

        @classmethod
        def _get_mapping(cls) -> Dict[str, int]:
            return cls._repository

    def test_set_mapping(self):
        """Test that _set_mapping replaces the entire repository."""
        new_mapping = {"new1": 100, "new2": 200}
        self.DummyMutator._set_mapping(new_mapping)
        assert self.DummyMutator._get_mapping() == new_mapping

    def test_update_mapping(self):
        """Test that _update_mapping correctly updates the repository."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        update = {"z": 30}
        self.DummyMutator._update_mapping(update)
        assert self.DummyMutator._get_mapping() == {"x": 10, "y": 20, "z": 30}

    def test_update_mapping_with_conflict(self):
        """Test that _update_mapping raises on conflicts."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        update = {"x": 100}  # Conflict with existing key
        with pytest.raises(RegistryError):
            self.DummyMutator._update_mapping(update)

    def test_clear_mapping(self):
        """Test that _clear_mapping empties the repository."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        self.DummyMutator._clear_mapping()
        assert self.DummyMutator._get_mapping() == {}

    def test_set_artifact(self):
        """Test that _set_artifact adds a new key-value pair."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        self.DummyMutator._set_artifact("z", 30)
        assert self.DummyMutator._get_mapping() == {"x": 10, "y": 20, "z": 30}

    def test_set_artifact_existing(self):
        """Test that _set_artifact raises on existing keys."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        with pytest.raises(RegistryError):
            self.DummyMutator._set_artifact("x", 100)

    def test_update_artifact(self):
        """Test that _update_artifact modifies an existing key."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        self.DummyMutator._update_artifact("x", 100)
        assert self.DummyMutator._get_mapping() == {"x": 100, "y": 20}

    def test_update_artifact_missing(self):
        """Test that _update_artifact raises on missing keys."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        with pytest.raises(RegistryError):
            self.DummyMutator._update_artifact("z", 30)

    def test_del_artifact(self):
        """Test that _del_artifact removes a key-value pair."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        self.DummyMutator._del_artifact("x")
        assert self.DummyMutator._get_mapping() == {"y": 20}

    def test_del_artifact_missing(self):
        """Test that _del_artifact raises on missing keys."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        with pytest.raises(RegistryError):
            self.DummyMutator._del_artifact("z")

    def test_assert_absence_success(self):
        """Test that _assert_absence succeeds for absent keys."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        mapping = self.DummyMutator._assert_absence("z")
        assert mapping == {"x": 10, "y": 20}

    def test_assert_absence_failure(self):
        """Test that _assert_absence raises for present keys."""
        # Reset for this test
        self.DummyMutator._repository = {"x": 10, "y": 20}

        with pytest.raises(RegistryError):
            self.DummyMutator._assert_absence("x")


# -------------------------------------------------------------------
# Test classes for ImmutableRegistryValidatorMixin
# -------------------------------------------------------------------


class TestImmutableRegistryValidatorMixin:
    """Tests for ImmutableRegistryValidatorMixin."""

    class DummyImmutableValidator(ImmutableRegistryValidatorMixin[str, int]):
        """Concrete implementation of ImmutableRegistryValidatorMixin for testing."""

        _repository: Dict[str, int] = {"a": 10, "b": 20, "c": 30}

        @classmethod
        def _get_mapping(cls) -> Dict[str, int]:
            return cls._repository

    def test_seal_identifier(self):
        """Test that _seal_identifier validates keys."""
        # String is hashable, should be accepted
        key = "test"
        assert self.DummyImmutableValidator._seal_identifier(key) == key

        # List is not hashable, should be rejected
        with pytest.raises(TypeError):
            self.DummyImmutableValidator._seal_identifier([1, 2, 3])

    def test_seal_artifact(self):
        """Test that _seal_artifact is a pass-through by default."""
        value = 42
        assert self.DummyImmutableValidator._seal_artifact(value) == value

    def test_get_artifact(self):
        """Test that get_artifact retrieves and validates values."""
        assert self.DummyImmutableValidator.get_artifact("b") == 20

        with pytest.raises(RegistryError):
            self.DummyImmutableValidator.get_artifact("z")

    def test_has_identifier(self):
        """Test that has_identifier correctly checks key presence."""
        assert self.DummyImmutableValidator.has_identifier("a")
        assert not self.DummyImmutableValidator.has_identifier("z")

    def test_has_artifact(self):
        """Test that has_artifact correctly checks value presence."""
        assert self.DummyImmutableValidator.has_artifact(30)
        assert not self.DummyImmutableValidator.has_artifact(40)

    def test_iter_identifiers(self):
        """Test that iter_identifiers returns an iterator over keys."""
        keys = list(self.DummyImmutableValidator.iter_identifiers())
        assert set(keys) == {"a", "b", "c"}


# -------------------------------------------------------------------
# Test classes for MutableRegistryValidatorMixin
# -------------------------------------------------------------------


class TestMutableRegistryValidatorMixin:
    """Tests for MutableRegistryValidatorMixin."""

    class DummyMutableValidator(MutableRegistryValidatorMixin[str, int]):
        """Concrete implementation of MutableRegistryValidatorMixin for testing."""

        _repository: Dict[str, int] = {"x": 100, "y": 200}

        @classmethod
        def _get_mapping(cls) -> Dict[str, int]:
            return cls._repository

    def test_probe_identifier(self):
        """Test that _probe_identifier validates keys."""
        # String is hashable, should be accepted
        key = "test"
        assert self.DummyMutableValidator._probe_identifier(key) == key

        # List is not hashable, should be rejected
        with pytest.raises(TypeError):
            self.DummyMutableValidator._probe_identifier([1, 2, 3])

    def test_probe_artifact(self):
        """Test that _probe_artifact is a pass-through by default."""
        value = 42
        assert self.DummyMutableValidator._probe_artifact(value) == value

    def test_register_artifact(self):
        """Test that register_artifact validates and adds key-value pairs."""
        # Reset for this test
        self.DummyMutableValidator._repository = {"x": 100, "y": 200}

        self.DummyMutableValidator.register_artifact("z", 300)
        assert self.DummyMutableValidator._get_mapping() == {
            "x": 100,
            "y": 200,
            "z": 300,
        }

        # Non-hashable key should be rejected
        with pytest.raises(TypeError):
            self.DummyMutableValidator.register_artifact([1, 2, 3], 400)

        # Duplicate key should be rejected
        with pytest.raises(RegistryError):
            self.DummyMutableValidator.register_artifact("x", 500)

    def test_unregister_artifact(self):
        """Test that unregister_artifact validates and removes key-value pairs."""
        # Reset for this test
        self.DummyMutableValidator._repository = {"x": 100, "y": 200}

        self.DummyMutableValidator.unregister_artifact("x")
        assert self.DummyMutableValidator._get_mapping() == {"y": 200}

        # Missing key should be rejected
        with pytest.raises(RegistryError):
            self.DummyMutableValidator.unregister_artifact("z")

        # Non-hashable key should be rejected
        with pytest.raises(TypeError):
            self.DummyMutableValidator.unregister_artifact([1, 2, 3])

    def test_clear_artifacts(self):
        """Test that clear_artifacts empties the repository."""
        # Reset for this test
        self.DummyMutableValidator._repository = {"x": 100, "y": 200}

        self.DummyMutableValidator.clear_artifacts()
        assert self.DummyMutableValidator._get_mapping() == {}


# -------------------------------------------------------------------
# Test custom implementation with overridden validators
# -------------------------------------------------------------------


class TestCustomValidators:
    """Tests for a custom implementation with overridden validators."""

    class CustomValidator(MutableRegistryValidatorMixin[str, int]):
        """Custom validator implementation with overridden validation methods."""

        _repository: Dict[str, int] = {}

        @classmethod
        def _get_mapping(cls) -> Dict[str, int]:
            return cls._repository

        @classmethod
        def _probe_artifact(cls, value: Any) -> int:
            # Custom validation to ensure value is positive
            if not isinstance(value, int):
                raise TypeError("Value must be an integer")
            if value <= 0:
                raise ValueError("Value must be positive")
            return value

        @classmethod
        def _probe_identifier(cls, key: Any) -> str:
            # Custom validation to ensure key is a non-empty string
            if not isinstance(key, str):
                raise TypeError("Key must be a string")
            if not key:
                raise ValueError("Key cannot be empty")
            return key

    def setup_method(self):
        """Reset repository before each test."""
        self.CustomValidator._repository = {}

    def test_custom_probe_artifact(self):
        """Test that custom _probe_artifact validation works."""
        # Positive integer should be accepted
        assert self.CustomValidator._probe_artifact(42) == 42

        # Non-integer should be rejected
        with pytest.raises(TypeError):
            self.CustomValidator._probe_artifact("not an int")

        # Non-positive integer should be rejected
        with pytest.raises(ValueError):
            self.CustomValidator._probe_artifact(0)
        with pytest.raises(ValueError):
            self.CustomValidator._probe_artifact(-1)

    def test_custom_probe_identifier(self):
        """Test that custom _probe_identifier validation works."""
        # Non-empty string should be accepted
        assert self.CustomValidator._probe_identifier("valid") == "valid"

        # Non-string should be rejected
        with pytest.raises(TypeError):
            self.CustomValidator._probe_identifier(42)

        # Empty string should be rejected
        with pytest.raises(ValueError):
            self.CustomValidator._probe_identifier("")

    def test_register_with_custom_validation(self):
        """Test registration with custom validation."""
        # Valid key and value should be accepted
        self.CustomValidator.register_artifact("valid", 42)
        assert self.CustomValidator._get_mapping() == {"valid": 42}

        # Invalid key should be rejected
        with pytest.raises(TypeError):
            self.CustomValidator.register_artifact(42, 42)
        with pytest.raises(ValueError):
            self.CustomValidator.register_artifact("", 42)

        # Invalid value should be rejected
        with pytest.raises(TypeError):
            self.CustomValidator.register_artifact("another", "not an int")
        with pytest.raises(ValueError):
            self.CustomValidator.register_artifact("another", 0)
