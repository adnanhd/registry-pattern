from typing import Generic, TypeVar, get_args
from registry import InstanceKeyMeta, InstanceValueMeta
import torch

class DtypeRegistry(InstanceValueMeta[torch.dtype], weak=False):
    """Registry for dtypes."""
    pass


DtypeRegistry.register(torch.dtype)
DtypeRegistry.register_instance(torch.float32)
DtypeRegistry.register_instance(torch.float64)