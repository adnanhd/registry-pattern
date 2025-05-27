from typing import Any, Generic, Hashable, Iterator, TypeVar, overload, Union, Optional, cast

from registry.core._validator import ValidationError

from .accessor import RegistryAccessorMixin
from .mutator import RegistryMutatorMixin

# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------

K = TypeVar("K", bound=Hashable)  # Keys must be hashable.
T = TypeVar("T")  # Registered items.

# -----------------------------------------------------------------------------
# Registry Validator Mixin
# -----------------------------------------------------------------------------


class ImmutableRegistryValidatorMixin(RegistryAccessorMixin[K, T], Generic[K, T]):
    """
    Mixin providing validation for immutable registry operations.

    This mixin extends the RegistryAccessorMixin to add validation
    to all read operations on the registry. It provides methods for
    sealing identifiers and artifacts, which validate keys and values
    during retrieval operations.
    """

    @classmethod
    def _intern_identifier(cls, value: Any) -> K:
        """
        Validate an identifier when storing in registry.

        Parameters:
            value: The value to validate.

        Returns:
            The validated value.
        """
        if not isinstance(value, Hashable):
            raise TypeError(f"Key must be hashable, got {type(value)}")
        return value  # type: ignore

    @classmethod
    def _intern_artifact(cls, value: T) -> T:
        """
        Validate an artifact when storing in registry.

        Parameters:
            value: The artifact to validate.

        Returns:
            The validated artifact.
        """
        return value

    @classmethod
    def _extern_identifier(cls, value: Any) -> K:
        """
        Validate an identifier when retrieving from registry.

        Parameters:
            value: The value to validate.

        Returns:
            The validated value.
        """
        if not isinstance(value, Hashable):
            raise TypeError(f"Key must be hashable, got {type(value)}")
        return value  # type: ignore

    @classmethod
    def _extern_artifact(cls, value: T) -> T:
        """
        Validate an artifact when retrieving from registry.

        Parameters:
            value: The artifact to validate.

        Returns:
            The validated artifact.
        """
        return value

    @classmethod
    def get_artifact(cls, key: K) -> T:
        """
        Retrieve an artifact from the registry with validation.

        Parameters:
            key: The key of the artifact to retrieve.

        Returns:
            The artifact if it exists.

        Raises:
            RegistryError: If the key is not in the registry.
        """
        validated_key = cls._intern_identifier(key)
        item = cls._get_artifact(validated_key)
        return cls._extern_artifact(item)

    @classmethod
    def has_identifier(cls, key: K) -> bool:
        """
        Check if an artifact with the given key exists in the registry.

        Parameters:
            key: The key to check.

        Returns:
            True if the artifact exists, False otherwise.
        """
        try:
            validated_key = cls._intern_identifier(key)
            return cls._has_identifier(validated_key)
        except ValidationError:
            return False

    @classmethod
    def has_artifact(cls, item: T) -> bool:
        """
        Check if an artifact with the given value exists in the registry.

        Parameters:
            item: The artifact to check.

        Returns:
            True if the artifact exists, False otherwise.
        """
        try:
            validated_item = cls._intern_artifact(item)
            return cls._has_artifact(validated_item)
        except ValidationError:
            return False

    @classmethod
    def iter_identifiers(cls) -> Iterator[K]:
        """
        Iterate over all artifact keys in the registry.

        Returns:
            Iterator over all keys in the registry.
        """
        return cls._iter_mapping()

    @classmethod
    def validate_artifact(cls, item: Any) -> T:
        """
        Validate an artifact and return its key.

        Parameters:
            item: The artifact to validate.

        Returns:
            The key of the validated artifact.

        Raises:
            ValidationError: If the artifact is invalid.
        """
        validated_item = cls._intern_artifact(item)
        validated_artifact = cls._extern_artifact(validated_item)
        return validated_artifact

# -----------------------------------------------------------------------------
# Mutable Registry Validator Mixin
# -----------------------------------------------------------------------------


class MutableRegistryValidatorMixin(
    ImmutableRegistryValidatorMixin[K, T], RegistryMutatorMixin[K, T], Generic[K, T]
):
    """
    Mixin providing validation for mutable registry operations.

    This mixin extends both RegistryValidatorMixin and RegistryMutatorMixin
    to add validation to all write operations on the registry. It provides
    methods for probing identifiers and artifacts, which validate keys and
    values during storage operations.
    """
    @classmethod
    def _identifier_of(cls, item: T) -> K:
        """Extract the identifier from an artifact."""
        raise NotImplementedError("Subclasses must implement identifier_of")

    @overload
    @classmethod
    def register_artifact(cls, key: K, item: T) -> T:
        """Register an artifact with explicit key."""
        ...

    @overload
    @classmethod
    def register_artifact(cls, key: T, item: None = None) -> T:
        """Register an artifact using extracted key."""
        ...

    @classmethod
    def register_artifact(cls, key: Union[K, T], item: Optional[T] = None) -> T:
        """
        Register an artifact in the registry with validation.

        Supports two calling patterns:
        1. register_artifact(key, item) - explicit key
        2. register_artifact(item) - extracted key from item

        Parameters:
            key: Either the explicit key or the item itself
            item: The item to register (None for single-argument form)

        Returns:
            The registered item
        """
        if item is None:
            # Single argument case: register_artifact(item)
            artifact = cast(T, key)  # key is actually the item in this case
            identifier = cls._identifier_of(artifact)
            # Extract key from the artifact using _intern_identifier
            validated_item = cls._intern_artifact(artifact)
            validated_key = cls._intern_identifier(identifier)
            cls._set_artifact(validated_key, validated_item)
            return artifact
        else:
            # Two argument case: register_artifact(key, item)
            actual_key = key  # key is actually the key in this case
            artifact = item
            validated_key = cls._intern_identifier(actual_key)
            validated_item = cls._intern_artifact(artifact)
            cls._set_artifact(validated_key, validated_item)
            return artifact

    @classmethod
    def unregister_artifact(cls, key: K) -> None:
        """
        Unregister an artifact from the registry with validation.

        Parameters:
            key: The key of the artifact to unregister.

        Raises:
            TypeError: If the key is not hashable.
            RegistryError: If the key does not exist in the registry.
        """
        validated_key = cls._extern_identifier(key)  # Use seal for deletion
        cls._del_artifact(validated_key)

    @classmethod
    def clear_artifacts(cls) -> None:
        """
        Clear all artifacts from the registry.
        """
        cls._clear_mapping()
