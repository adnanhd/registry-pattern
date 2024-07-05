from typing import Any,  Type, ClassVar, Literal
import torch
import re
import inspect
import pydantic
from .base import _BaseModel
from pydantic import Field, SerializerFunctionWrapHandler, BaseModel, InstanceOf, ConfigDict
from pydantic_core import core_schema

from pydantic_pytorch.registrations import InstanceRegistryMetaclass

__ALL__ = ['TorchDType']


class TorchDType(_BaseModel[torch.dtype]):
    """TypedDict for torch.device"""

    dtypes: ClassVar[dict[str, InstanceOf[Any]]] = {
        f'torch.{k}': v
        for k, v in inspect.getmembers(torch)
        if isinstance(v, torch.dtype)
    }

    aliases: ClassVar[dict[str, str]] = {
        k: str(v)
        for k, v in dtypes.items()
    }
    aliases.update({k[6:]: v for k, v in aliases.items()})
    
    aliases.update({
        v[0] + re.findall(r'[0-9]+', v)[0]: v
        for v in aliases.values()
        if re.match(r'^[a-z]+[0-9]+$', v)
    })
    
    
    type: str = Field(description="String representation of torch dtype", 
                      json_schema_extra=dict(required=True))

    model_config = ConfigDict(validate_default=True, validate_assignment=True)

    @pydantic.model_validator(mode='before')
    @staticmethod
    def validate_string(values: str) -> dict[str, str]:
        print('validate model')
        if isinstance(values, str):
            return dict(type=values)
        return values

    @pydantic.field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in cls.aliases.keys():
            raise ValueError(f'Invalid dtype: {v}')
        return cls.aliases[v]

    def _builder(self) -> Type[torch.dtype]:
        return (lambda type: self.dtypes[type])

    def _validate(self, v: torch.dtype) -> torch.dtype:
        print('validating instance of torch device')
        if self.type != self._serialize(v)['type']:
            raise ValueError(f'Invalid dtype: {v} e.g. {self._serialize(v)} '
                             f'but must be {self.type}')
        return v

    @staticmethod
    def _serialize(v: torch.dtype, nxt: SerializerFunctionWrapHandler = lambda x: x) -> dict[str, str | int]:
        print('serializing torch device')
        return nxt(dict(type=self.aliases[str(v)]))
