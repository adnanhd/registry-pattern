from .base import PydanticClassRegistryBaseModel

import torch
from typing import Annotated, Generator
from pydantic import InstanceOf, AfterValidator, Field, model_validator, field_validator, ConfigDict



class TorchModule(PydanticClassRegistryBaseModel):
    """Module registry."""


TorchModule.register(torch.nn.Module)
TorchModule.register_class(torch.nn.Linear)

def get_parameters(mdl: torch.nn.Module) -> Generator[torch.nn.Parameter, None, None]:
    if isinstance(mdl, torch.nn.Module):
        return mdl.parameters()
    elif isinstance(mdl, Generator):
        return mdl
    else:
        raise ValueError(f"Invalid module: {mdl}")


class TorchOptimizer(PydanticClassRegistryBaseModel):
    """Optimizer registry."""
    params: Annotated[TorchModule, AfterValidator(get_parameters)]
    lr: float = Field(..., description="Learning rate", ge=0.0)

TorchOptimizer.register(torch.optim.Optimizer)
TorchOptimizer.register_class(torch.optim.Adam)
