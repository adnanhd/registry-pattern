from typing import Any, Sequence, Dict, Generic, TypeVar, Type, Annotated, Literal, Optional
import torch
import numpy as np
import pydantic
from pydantic import Field, GetCoreSchemaHandler, SerializerFunctionWrapHandler, ValidatorFunctionWrapHandler, AfterValidator, WrapValidator, BaseModel, ValidationInfo, NonNegativeInt
from pydantic_core import core_schema
from dataclasses import dataclass
import annotated_types

from typing_extensions import TypedDict
from pydantic import TypeAdapter


class TorchDevice(BaseModel):
    """TypedDict for torch.device"""

    type: Literal['cpu', 'cuda']
    index: Optional[int] = None
    
    @pydantic.field_validator('index')
    @classmethod
    def validate_index(cls, v: Optional[int], info: ValidationInfo) -> int:
        print('validating index')
        if info.data['type'] == 'cpu'and v is not None:
            raise ValueError('Index must be None for cpu device')
        elif info.data['type'] == 'cuda' and v is None:
            # TODO: change it with the number of available cuda devices
            return 0
        return v
    

    @pydantic.model_validator(mode='before')
    @staticmethod
    def validate_type_naming(values: Dict[str, Any]) -> Dict[str, Any]:
        print('validating type naming')
        type = values.get('type', None)
        index = values.get('index', None)
        
        if type is None:
            raise ValueError('Device type must be provided')
        if ':' in type and index is not None:
            raise ValueError('Index must be provided with device type')
        
        if ':' in type:
            assert isinstance(type, str), f'Invalid device string: {type}'
            assert index is None, 'Index must be provided with device type'

            try:
                type, index = type.split(':')
            except ValueError:
                raise ValueError(f'Invalid device string: {type}')
        print('type:', type, 'index:', index)
        return dict(type=type, index=index)
    
    @pydantic.model_validator(mode='after')
    def validate_model_torch_device(self):
        print('validating model')
        return torch.device(self.type, self.index)
    
    @staticmethod
    def validate_instance_of_torch_device(v: torch.device) -> "TorchDevice":
        print('validating instance of torch device')
        return v #TorchDevice(type=v.type, index=v.index)
    
    @staticmethod
    def serialize(v: torch.device, nxt: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        print('serializing')
        return nxt(dict(type=v.type, index=v.index))
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler):
        model_schema = handler(source_type)
        
        instance_validation = core_schema.no_info_after_validator_function(
            cls.validate_instance_of_torch_device,
            core_schema.is_instance_schema(torch.device),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                cls.serialize)
        )

        return core_schema.union_schema([model_schema, instance_validation])