from pydantic import BaseModel, ConfigDict, InstanceOf, field_validator, model_validator
from typing import TypeVar, Type
from ..registrations.registry import ClassRegistryMetaclass, FunctionalRegistryMetaclass
from pydantic._internal._model_construction import ModelMetaclass


class PydanticClassRegistryMetaclass(ModelMetaclass, ClassRegistryMetaclass):
    pass


T = TypeVar('T', bound='PydanticClassRegistryBaseModel')


class PydanticClassRegistryBaseModel(BaseModel, metaclass=PydanticClassRegistryMetaclass):
    type: str

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


class PydanticFunctionalRegistryMetaclass(ModelMetaclass, FunctionalRegistryMetaclass):
    pass


class PydanticFunctionalRegistryBaseModel(BaseModel, metaclass=PydanticFunctionalRegistryMetaclass):
    pass
