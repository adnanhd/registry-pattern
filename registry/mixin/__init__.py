r"""Public API for the registry mixins package.

Exports:
    RegistryAccessorMixin: Read-only registry interface.
    RegistryMutatorMixin: Write-side extensions over accessor.
    ImmutableValidatorMixin: Read-side validation wrapper.
    MutableValidatorMixin: Write-side validation wrapper.
    ContainerMixin: DI container for object graph construction.
    RegistryFactorizorMixin: Alias for ContainerMixin (backward compat).
"""

from .accessor import RegistryAccessorMixin
from .factorizor import ContainerMixin, RegistryFactorizorMixin
from .mutator import RegistryMutatorMixin
from .validator import ImmutableValidatorMixin, MutableValidatorMixin

__all__ = [
    "RegistryAccessorMixin",
    "RegistryMutatorMixin",
    "ImmutableValidatorMixin",
    "MutableValidatorMixin",
    "ContainerMixin",
    "RegistryFactorizorMixin",
]
