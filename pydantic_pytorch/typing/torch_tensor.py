from .multiarray import TorchDevice, TorchDType, TorchShape
from typing import Any
import torch
from pydantic import BaseModel, model_validator, field_validator, InstanceOf

class TorchTensor(BaseModel):
    data: InstanceOf[torch.Tensor]
    shape: TorchShape = None
    device: TorchDevice = None
    dtype: TorchDType = None
    requires_grad: bool = False


    @model_validator(mode='wrap')
    def validate_model(self, handler: Any) -> "TorchTensor":
        tensor_type: TorchTensor = handler(self)
        data = tensor_type.data
        
        if tensor_type.dtype is not None and tensor_type.dtype != data.dtype:
            raise ValueError(f'Data type mismatch: {tensor_type.dtype!r} != {data.dtype!r}')

        if tensor_type.device is not None and tensor_type.device != data.device:
            raise ValueError(f'Device mismatch: {tensor_type.device!r} != {data.device!r}')    

        if tensor_type.shape is not None and tensor_type.shape != data.shape:
            raise ValueError(f'Shape mismatch: {tensor_type.shape!r} != {data.shape!r}')
        
        return data
        
    @model_validator(mode='after')
    def build_model(self) -> torch.Tensor:
        if isinstance(self, torch.Tensor):
            return self
        return torch.tensor(self.data, device=self.device, dtype=self.dtype)
