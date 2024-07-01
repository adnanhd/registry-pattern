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


__ALL__ = ['TorchDevice']

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
        """
        elif info.data['type'] == 'cuda' and v is None:
            # TODO: change it with the number of available cuda devices
            return 0
        """
        return v
    

    @pydantic.model_validator(mode='before')
    @staticmethod
    def validate_device_namenaming(values: Dict[str, Any]) -> Dict[str, Any]:
        print('validating type naming')
        if not isinstance(values, dict):
            raise ValueError(f'Input must be a Dict[str, Any] not {type(values)}')
        device_name = values.get('type', None)
        device_index = values.get('index', None)
        
        if device_name is None:
            raise ValueError('Device type must be provided')
        if ':' in device_name and device_index is not None:
            raise ValueError('Index must be provided with device type')
        
        if ':' in device_name:
            assert isinstance(device_name, str), f'Invalid device string: {device_name}'
            assert device_index is None, 'Index must be provided with device type'

            try:
                device_name, device_index = device_name.split(':')
            except ValueError:
                raise ValueError(f'Invalid device string: {device_name}')
        print('type:', device_name, 'index:', device_index)
        return dict(type=device_name, index=device_index)
    
    @pydantic.model_validator(mode='after')
    def build_model(self):
        print('validating model')
        return torch.device(self.type, self.index)
    
    @staticmethod
    def validate_instance_of_torch_device(v: torch.device) -> torch.device:
        print('validating instance of torch device')
        return TorchDevice(type=v.type, index=v.index if v.index != -1 else None).build_model()
    
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
