from typing import Any,  Type
import torch
import inspect
import pydantic
from .base import _BaseModel
from pydantic import Field, SerializerFunctionWrapHandler, BaseModel, Field
from pydantic_core import core_schema

from pydantic_pytorch.registrations import InstanceRegistryMetaclass

__ALL__ = ['TorchDType']


class Str2TorchDeviceMapping(metaclass=InstanceRegistryMetaclass):
    _DUPLICATE_DICT = {}

    @classmethod
    def register_instance(cls, name: str, instance: Type[torch.dtype]) -> None:
        assert isinstance(name, str), f'{name} is not a string'
        if name in cls._DUPLICATE_DICT:
            val = cls.get_registered(cls._DUPLICATE_DICT[name])
            raise ValueError(f'Can\'t register {name} twice. '
                             f'Already registered to {val}')
        cls._DUPLICATE_DICT[name] = str(instance)
        if not cls.has_registered(instance):
            return InstanceRegistryMetaclass.register_instance(cls, instance)
        return instance

    @classmethod
    def get_registered(cls, name: str) -> Type[torch.dtype]:
        if name in cls._DUPLICATE_DICT:
            return InstanceRegistryMetaclass.get_registered(cls, cls._DUPLICATE_DICT[name])
        return InstanceRegistryMetaclass.get_registered(cls, name)


Str2TorchDeviceMapping.register(torch.dtype)


for k, v in inspect.getmembers(torch):
    if isinstance(v, torch.dtype):
        Str2TorchDeviceMapping.register_instance(k, v)


class TorchDType(_BaseModel[torch.dtype]):
    """TypedDict for torch.device"""
    type: str = Field(..., description="String representation of torch dtype")

    @pydantic.field_validator('type')
    def validate_type(cls, v: str) -> str:
        if v not in Str2TorchDeviceMapping.list_registry():
            raise ValueError(f'Invalid dtype: {v}')
        return Str2TorchDeviceMapping._DUPLICATE_DICT[str(v)]

    def _builder(self) -> Type[torch.dtype]:
        return Str2TorchDeviceMapping.get_registered(self.type)

    def _validate(self, v: torch.dtype) -> torch.dtype:
        print('validating instance of torch device')
        if self.type != self._serialize(v):
            raise ValueError(f'Invalid dtype: {v} e.g. {self._serialize(v)} '
                             f'but must be {self.type}')
        return v

    @staticmethod
    def _serialize(v: torch.dtype, nxt: SerializerFunctionWrapHandler = lambda x: x) -> dict[str, str | int]:
        print('serializing torch device')
        return nxt(dict(type=Str2TorchDeviceMapping._DUPLICATE_DICT[str(v)]))
