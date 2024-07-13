import dataclasses
from typing import Generic, TypeVar, Annotated
from pydantic import BaseModel, GetCoreSchemaHandler, InstanceOf, ConfigDict, model_validator, field_validator
from pydantic_core import CoreSchema, core_schema
from pydantic._internal import _generics
from .registry import RegistryMeta


Builder = TypeVar("Builder", bound=RegistryMeta)


class BuildFrom(Generic[Builder]):
    """A class that can be built from a builder."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:
        import pdb
        pdb.set_trace()
        # return core_schema(handler, self, ConfigDict(Builder=Builder))


@dataclasses.dataclass  # (extra=True)
class RegistryKey(str, Generic[Builder]):
    # registry: InstanceOf[RegistryMeta]

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:
        import pdb
        pdb.set_trace()
        pass


class BuilderConfig(Generic[Builder], BaseModel):
    """A configuration for a builder."""
    type: RegistryKey[Builder]

    model_config = ConfigDict(
        extra='allow', validate_default=True, validate_assignment=True)

    @model_validator(mode='after')
    def build_model(self) -> InstanceOf[T]:
        return self._builder(**self._builder_config)

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in cls.list_registry():
            raise ValueError(f'Invalid module: {v}')
        return v

    @property
    def _builder(self) -> Type[InstanceOf[T]]:
        return self.__class__.get_registered(self.type)

    @property
    def _builder_config(self) -> dict:
        return self.model_dump(exclude={'type'})


@dataclasses.dataclass
class BuilderValidator:
    registry: InstanceOf[RegistryMeta]

    def __post_init__(self):
        if not isinstance(self.registry, RegistryMeta):
            raise TypeError(f"{self.registry} is not a RegistryMeta")
        else:
            print('BuilderValidator.__post_init__')

    def _validate_build_config(self, value: dict):
        if not isinstance(value, dict) or 'type' not in value:
            raise TypeError(f"{value} is not a dict with a 'type' key")
        builder = self.registry.get_registered(value['type'])

    def __get_pydantic_core_schema__(self, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:

        instance_of_schema = core_schema.is_instance_schema(
            _generics.get_origin(source_type) or source_type)
        return instance_of_schema
        import pdb
        pdb.set_trace()
