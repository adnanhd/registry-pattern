from typing import Protocol, Annotated
from registry import ClassRegistry, BuildFrom, BuilderValidator
from pydantic import BaseModel, InstanceOf
import torch


# bug with arguments with default values
class TorchModule(Protocol):
    in_features: int
    out_features: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class TorchModuleRegistry(ClassRegistry[TorchModule]):
    """Registry for dtypes."""
    pass


TorchModuleRegistry.register(torch.nn.Module)
TorchModuleRegistry.register_class(torch.nn.Linear)


class TrainModel(BaseModel):
    model: Annotated[torch.nn.Module, BuilderValidator(
        registry=TorchModuleRegistry)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)


model = TrainModel(model=torch.nn.Linear(1, 1))
