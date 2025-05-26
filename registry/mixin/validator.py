from typing import Any, Generic, Hashable, Iterator, TypeVar

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
    def _probe_identifier(cls, value: Any) -> K:
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
    def _probe_artifact(cls, value: T) -> T:
        """
        Validate an artifact when storing in registry.

        Parameters:
            value: The artifact to validate.

        Returns:
            The validated artifact.
        """
        return value

    @classmethod
    def _seal_identifier(cls, value: Any) -> K:
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
    def _seal_artifact(cls, value: T) -> T:
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
        validated_key = cls._probe_identifier(key)
        item = cls._get_artifact(validated_key)
        return cls._seal_artifact(item)

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
            validated_key = cls._probe_identifier(key)
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
            validated_item = cls._probe_artifact(item)
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
    def register_artifact(cls, key: K, item: T) -> None:
        """
        Register an artifact in the registry with validation.

        Parameters:
            key: The key of the artifact to register.
            item: The artifact to register.

        Raises:
            TypeError: If the key is not hashable.
            RegistryError: If the key already exists in the registry.
        """
        validated_key = cls._probe_identifier(key)
        validated_item = cls._probe_artifact(item)
        cls._set_artifact(validated_key, validated_item)

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
        validated_key = cls._seal_identifier(key)  # Use seal for deletion
        cls._del_artifact(validated_key)

    @classmethod
    def clear_artifacts(cls) -> None:
        """
        Clear all artifacts from the registry.
        """
        cls._clear_mapping()
