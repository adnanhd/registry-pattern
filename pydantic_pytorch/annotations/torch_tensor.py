from .multiarray import TorchDevice, TorchDType, TorchShape
from typing import Any, ClassVar
import torch
from pydantic import BaseModel, model_validator, field_validator, InstanceOf, Field


class TorchTensor(BaseModel):
    data: InstanceOf[torch.Tensor]
    shape: TorchShape = Field(init=False, init_var=False)
    device: TorchDevice = Field(init=False, init_var=False)
    dtype: TorchDType = Field(init=False)
    requires_grad: bool = Field(init=False)

