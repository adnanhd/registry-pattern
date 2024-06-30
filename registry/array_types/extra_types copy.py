from typing import Any, TypeGuard, Dict, Generic, TypeVar
import torch
import numpy as np
from pydantic import GetCoreSchemaHandler, SerializerFunctionWrapHandler, BaseModel, ValidationInfo
from pydantic_core import core_schema


class TorchDevice(BaseModel):
    type: str


TorchShape = TypeVar('TorchShape', bound=torch.Size)
TorchDevice = TypeVar('TorchDevice', bound=torch.device)
TorchDtype = TypeVar('TorchDtype', bound=torch.dtype)

class TorchTensor:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler):
        return core_schema.with_info_after_validator_function(
            cls.validate,
            core_schema.is_instance_schema(torch.Tensor),
            serialization=core_schema.wrap_serializer_function_ser_schema(cls.serialize)
        )

    @staticmethod
    def validate(value: torch.Tensor, info: ValidationInfo) -> torch.Tensor:
        if info.context is None:
            return value
        assert isinstance(info.context, Dict), "Context must be a dictionary"
        
        if value.dtype != info.context.get('dtype', value.dtype):
            dtype = info.context.get('dtype', value.dtype)
            raise ValueError(f"Tensor must have dtype {dtype} but has {value.dtype}")
        
        assert isinstance(info.context.get('device', None), torch.device), "Device must be a string"
        if value.device != info.context.get('device', value.device):
            device = info.context.get('device', value.device)
            raise ValueError(f"Tensor must be on device {device} but is on {value.device}")
        
        if value.shape != info.context.get('shape', value.shape):
            shape = info.context.get('shape', value.shape)
            raise ValueError(f"Tensor must have shape {shape} but has {value.shape}")
        
        return value
    
    @staticmethod
    def serialize(v: torch.Tensor, nxt: SerializerFunctionWrapHandler) -> Any:
        return v.tolist()


class NumpyNdArray:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.is_instance_schema(np.ndarray)
        )

    @staticmethod
    def validate(value: np.ndarray) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        raise ValueError("Value must be a numpy.ndarray")


# Example usage in a Pydantic model
class ExampleModel(BaseModel):
    tensor: TorchTensor
    array: NumpyNdArray


# Testing the model
example = ExampleModel(tensor=torch.tensor(
    [1, 2, 3]), array=np.array([1, 2, 3]))
print(example)


if __name__ == '__main__':
    from pydantic import TypeAdapter

    tensor_check = TypeAdapter(TorchTensor)
    array_check = TypeAdapter(NumpyNdArray)

    print(tensor_check.validate_python(torch.tensor([1, 2, 3]), context={'device': 'cpu'}))
