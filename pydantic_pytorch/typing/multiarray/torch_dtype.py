from typing import Any, Sequence, Dict, Generic, TypeVar, Type, Annotated, Literal, Optional
import torch
import numpy as np
import pydantic
from pydantic import Field, GetCoreSchemaHandler, SerializerFunctionWrapHandler, ValidatorFunctionWrapHandler, AfterValidator, WrapValidator, BaseModel, ValidationInfo, NonNegativeInt
from pydantic_core import core_schema
from dataclasses import dataclass
import annotated_types

from typing_extensions import TypedDict
from pydantic import TypeAdapter, InstanceOf, Field
from pydantic_pytorch.registry import InstanceRegistryMetaclass

__ALL__ = ['TorchDType']

class Str2TorchDeviceMapping(metaclass=InstanceRegistryMetaclass):
    pass

Str2TorchDeviceMapping.register(torch.dtype)
Str2TorchDeviceMapping.register_instance('torch.float32', torch.float32)
Str2TorchDeviceMapping.register_instance('torch.float', torch.float)
Str2TorchDeviceMapping.register_instance('torch.float64', torch.float64)
Str2TorchDeviceMapping.register_instance('torch.double', torch.double)


class TorchDType(BaseModel):
    """TypedDict for torch.device"""
    type: str = Field(..., description="String representation of torch dtype")

    @pydantic.model_validator(mode='after')
    def build_model(self) -> torch.dtype:
        return Str2TorchDeviceMapping.get_registered(self.type)

    @pydantic.field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in Str2TorchDeviceMapping.list_registry():
            raise ValueError(f'Invalid dtype: {v}')
        return v

    @pydantic.field_serializer('type')
    @classmethod
    def serialize_type(cls, v: str) -> str:
        print('serialize_dtype')
        return v

    @staticmethod
    def serialize_torch_dtype(v: torch.dtype, nxt: Any) -> Dict[str, Any]:
        print('serializing')
        return nxt(dict(type=str(v)))

    @classmethod
    def validate_torch_dtype(cls, v: torch.dtype) -> 'TorchDType':
        return cls(type=str(v))
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler):
        model_schema = handler(source_type)
        
        instance_validation = core_schema.no_info_after_validator_function(
            cls.validate_torch_dtype,
            core_schema.is_instance_schema(torch.dtype),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                cls.serialize_torch_dtype)
        )
        
        return core_schema.union_schema([model_schema, instance_validation])
