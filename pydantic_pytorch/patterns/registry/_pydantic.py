"""
This module provides compatibility between Pydantic and PyTorch.
"""

import dataclasses
from typing import TypeVar, Callable, Any
from functools import partial
from pydantic_core import CoreSchema, core_schema
from pydantic import BaseModel, GetCoreSchemaHandler, InstanceOf, ConfigDict
from pydantic import Field
from pydantic._internal import _generics
from .base import BaseRegistry as RegistryMeta


Builder = TypeVar("Builder") #, bound=RegistryMeta)


class Buildable(BaseModel):
    """A configuration for a builder."""

    type: str
    builder: Callable[..., Any] = Field(repr=False, exclude=True)

    model_config = ConfigDict(
        extra="allow",
        validate_default=True,
        validate_assignment=True,
        copy=False,
    )

    @property
    def build_model_config(self) -> dict:
        """Get the model configuration."""
        return self.model_dump(exclude={"builder", "type"})

    def build_model(self, **extra_kwargs) -> Callable[..., Any]:
        """Build the model."""
        kwargs = {**self.build_model_config, **extra_kwargs}
        return partial(self.builder, **kwargs)


@dataclasses.dataclass
class BuilderValidator:
    """A validator for a builder."""

    registry: InstanceOf[RegistryMeta]

    def __post_init__(self):
        if not isinstance(self.registry, RegistryMeta):
            raise TypeError(f"{self.registry} is not a RegistryMeta")
        else:
            print("BuilderValidator.__post_init__")

    def __get_pydantic_core_schema__(
        self, source_type: type, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        instance_of_schema = core_schema.is_instance_schema(
            _generics.get_origin(source_type) or source_type
        )

        def _validate_build_config(value: dict):
            if not isinstance(value, dict) or "type" not in value:
                raise TypeError(f"{value} is not a dict with a 'type' key")
            builder = self.registry.get_registered(value["type"])
            return Buildable(builder=builder, **value).build_model()()

        builder_schema = core_schema.no_info_before_validator_function(
            _validate_build_config, instance_of_schema
        )
        return core_schema.json_or_python_schema(
            json_schema=builder_schema, python_schema=instance_of_schema
        )
