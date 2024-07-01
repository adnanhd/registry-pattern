from .multiarray import TorchDevice, TorchDType, TorchShape
from typing import Any, ClassVar
import torch
from pydantic import BaseModel, model_validator, field_validator, InstanceOf


class TorchTensor(BaseModel):
    data: InstanceOf[torch.Tensor]
    shape: ClassVar[InstanceOf[TorchShape]] = None
    device: ClassVar[InstanceOf[TorchDevice]] = None
    dtype: ClassVar[InstanceOf[TorchDType]] = None
    requires_grad: ClassVar[bool] = False

    @field_validator('data')
    @classmethod
    def validate_data_shape(cls, data: torch.Tensor) -> "torch.Tensor":
        data_shape = cls.shape.validate_torch_shape(data.shape)
        if cls.shape is not None and cls.shape != data_shape:
            raise ValueError(f"Expected shape to be {cls.shape!r} but got {data_shape!r}")
        return data

    @field_validator('data')
    @classmethod
    def validate_data_device(cls, data: torch.Tensor) -> "torch.Tensor":
        data_device = cls.device.validate_torch_device(data.device)
        if cls.device is not None and cls.device != data_device:
            raise ValueError(f"Expected device to be {cls.device!r}")
        return data

    @field_validator('data')
    @classmethod
    def validate_data_dtype(cls, data: torch.Tensor) -> "torch.Tensor":
        data_dtype = cls.dtype.validate_torch_dtype(data.dtype)
        if cls.dtype is not None and cls.dtype != data_dtype:
            raise ValueError(f"Expected dtype to be {cls.dtype!r}")
        return data

    @field_validator('data')
    @classmethod
    def validate_data_grad(cls, data: torch.Tensor) -> "torch.Tensor":
        if cls.requires_grad is not None:
            if data.requires_grad != cls.requires_grad:
                raise ValueError(
                    f"Expected requires_grad to be {cls.requires_grad}")
        return data

    @model_validator(mode='after')
    def build_model(self) -> torch.Tensor:
        if isinstance(self, torch.Tensor):
            return self
        return self.data
