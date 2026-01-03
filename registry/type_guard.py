"""Pydantic Type Guard for config-or-instance validation.

This module provides `Buildable[T]`, a Pydantic-compatible type annotation that:
1. Accepts an already-constructed instance of type T
2. Accepts a BuildCfg (or dict) and builds it into an instance of T
3. Validates that the result is indeed of type T

Usage:
    from registry import Buildable, TypeRegistry, ContainerMixin

    class ModelRegistry(TypeRegistry[nn.Module]):
        pass

    class TrainerConfig(BaseModel):
        model: Buildable[nn.Module]  # Accepts nn.Module or BuildCfg -> nn.Module
        optimizer: Buildable[Optimizer]

    # Both work:
    config1 = TrainerConfig(model=my_model_instance, optimizer=my_optim)
    config2 = TrainerConfig(
        model={"type": "ResNet", "repo": "models", "data": {"layers": 18}},
        optimizer={"type": "Adam", "repo": "optimizers", "data": {"lr": 0.001}}
    )
    # config2.model and config2.optimizer are built instances
"""

from __future__ import annotations

import logging
from typing import Any, Generic, Type, TypeVar, get_args, get_origin

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from .container import BuildCfg, is_build_cfg, normalize_cfg

logger = logging.getLogger(__name__)

__all__ = [
    "Buildable",
    "BuildableValidator",
]

T = TypeVar("T")

# Cache for parameterized types to avoid recreation
_PARAMETERIZED_CACHE: dict[type, type] = {}


class BuildableValidator(Generic[T]):
    """Validator that accepts either an instance of T or a BuildCfg that builds to T.

    This class implements Pydantic's custom type protocol, allowing it to be used
    as a type annotation in Pydantic models.
    """

    def __init__(self, expected_type: Type[T]):
        self.expected_type = expected_type

    def validate(self, value: Any) -> T:
        """Validate and potentially build the value.

        Args:
            value: Either an instance of T, a BuildCfg, or a dict that looks like BuildCfg.

        Returns:
            An instance of type T.

        Raises:
            ValueError: If the value cannot be converted to type T.
        """
        # Case 1: Already an instance of the expected type
        if isinstance(value, self.expected_type):
            return value

        # Case 2: BuildCfg or dict that looks like BuildCfg
        if isinstance(value, BuildCfg) or is_build_cfg(value):
            # Import here to avoid circular imports
            from .mixin.factorizor import ContainerMixin

            # Normalize to BuildCfg
            cfg = normalize_cfg(value) if isinstance(value, dict) else value

            # Build the object
            try:
                result = ContainerMixin.build_cfg(cfg)
            except Exception as e:
                raise ValueError(
                    f"Failed to build {self.expected_type.__name__} from config: {e}"
                ) from e

            # Validate the result type
            if not isinstance(result, self.expected_type):
                raise ValueError(
                    f"Built object is {type(result).__name__}, "
                    f"expected {self.expected_type.__name__}"
                )

            return result

        # Case 3: Invalid type
        raise ValueError(
            f"Expected {self.expected_type.__name__} instance or BuildCfg, "
            f"got {type(value).__name__}"
        )


def _create_validator_function(expected_type: Type) -> Any:
    """Create a validation function for the given expected type."""

    def validate_buildable(value: Any) -> Any:
        """Validate and build if necessary."""
        # Already an instance of expected type
        if expected_type is not object and isinstance(value, expected_type):
            return value

        # BuildCfg or dict config
        if isinstance(value, BuildCfg) or is_build_cfg(value):
            from .mixin.factorizor import ContainerMixin

            cfg = normalize_cfg(value) if isinstance(value, dict) else value

            try:
                result = ContainerMixin.build_cfg(cfg)
            except Exception as e:
                raise ValueError(f"Failed to build from config: {e}") from e

            # Type check (optional for object type)
            if expected_type is not object and not isinstance(result, expected_type):
                logger.warning(
                    "Built object type %s doesn't match expected %s",
                    type(result).__name__,
                    expected_type.__name__,
                )

            return result

        # If expected_type is object, accept anything that's not a config
        if expected_type is object:
            return value

        raise ValueError(
            f"Expected {expected_type.__name__} instance or BuildCfg dict, "
            f"got {type(value).__name__}"
        )

    return validate_buildable


def _create_json_schema(expected_type: Type) -> JsonSchemaValue:
    """Create JSON schema for the Buildable type."""
    type_name = getattr(expected_type, "__name__", "object")

    return {
        "oneOf": [
            {
                "type": "object",
                "description": f"Instance of {type_name}",
            },
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "repo": {"type": "string", "default": "default"},
                    "data": {"type": "object", "default": {}},
                    "meta": {"type": "object", "default": {}},
                },
                "required": ["type"],
                "description": f"BuildCfg that builds to {type_name}",
            },
        ]
    }


class Buildable(Generic[T]):
    """Type annotation for values that can be either an instance or a BuildCfg.

    Usage in Pydantic models:
        class MyConfig(BaseModel):
            model: Buildable[nn.Module]      # Accepts nn.Module or BuildCfg
            optimizer: Buildable[Optimizer]  # Accepts Optimizer or BuildCfg

    The Buildable type will:
    1. Pass through values that are already instances of the expected type
    2. Build values that are BuildCfg or dict configs using ContainerMixin.build_cfg()
    3. Validate that built values match the expected type

    Example:
        @ModelRegistry.register_artifact
        class MyModel:
            def __init__(self, size: int):
                self.size = size

        ContainerMixin.configure_repos({"models": ModelRegistry})

        class Config(BaseModel):
            model: Buildable[MyModel]

        # Works with instance
        cfg1 = Config(model=MyModel(size=10))

        # Works with BuildCfg dict
        cfg2 = Config(model={"type": "MyModel", "data": {"size": 20}})
        assert isinstance(cfg2.model, MyModel)
    """

    __slots__ = ()

    def __class_getitem__(cls, item: Type[T]) -> type:
        """Support Buildable[SomeType] syntax."""
        # Check cache first
        if item in _PARAMETERIZED_CACHE:
            return _PARAMETERIZED_CACHE[item]

        # Create a new marker class that carries type info
        class _BuildableType:
            """Marker class for Buildable[T] that implements Pydantic protocol."""

            _expected_type: Type = item

            def __class_getitem__(inner_cls, inner_item: Type) -> type:
                """Prevent nested parameterization causing recursion."""
                return Buildable[inner_item]

            @classmethod
            def __get_pydantic_core_schema__(
                cls,
                source_type: Any,
                handler: GetCoreSchemaHandler,
            ) -> CoreSchema:
                """Generate Pydantic core schema for validation."""
                expected = cls._expected_type
                validator_fn = _create_validator_function(expected)

                return core_schema.no_info_plain_validator_function(
                    validator_fn,
                    serialization=core_schema.plain_serializer_function_ser_schema(
                        lambda x: x,
                        info_arg=False,
                        return_schema=core_schema.any_schema(),
                    ),
                )

            @classmethod
            def __get_pydantic_json_schema__(
                cls,
                _core_schema: CoreSchema,
                handler: GetJsonSchemaHandler,
            ) -> JsonSchemaValue:
                """Generate JSON schema for documentation."""
                return _create_json_schema(cls._expected_type)

        # Give it a nice name
        type_name = getattr(item, "__name__", str(item))
        _BuildableType.__name__ = f"Buildable[{type_name}]"
        _BuildableType.__qualname__ = f"Buildable[{type_name}]"

        # Cache and return
        _PARAMETERIZED_CACHE[item] = _BuildableType
        return _BuildableType

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Fallback schema for unparameterized Buildable or Buildable[T]."""
        # Try to extract type from generic args
        args = get_args(source_type)
        expected_type = args[0] if args else object

        validator_fn = _create_validator_function(expected_type)

        return core_schema.no_info_plain_validator_function(
            validator_fn,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: x,
                info_arg=False,
                return_schema=core_schema.any_schema(),
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Generate JSON schema for unparameterized Buildable."""
        return _create_json_schema(object)
