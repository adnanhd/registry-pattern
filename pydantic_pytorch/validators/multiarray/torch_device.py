import torch
import dataclasses
from typing import Any
from functools import partial
from pydantic_core import core_schema, CoreSchema
from ...utils._types import TORCH_DEVICE_TYPE, TORCH_DEVICE_INDEX
from pydantic import GetCoreSchemaHandler, SerializerFunctionWrapHandler


def validate_torch_device_type(x: torch.device, device_type: TORCH_DEVICE_TYPE) -> torch.device:
    if x.type != device_type:
        raise ValueError(f'Invalid device type: {x.type!r} '
                         f'must be {device_type!r}')
    return x


def validate_torch_device_index(x: torch.device, device_index: TORCH_DEVICE_INDEX) -> torch.device:
    if x.index != device_index:
        raise ValueError(f'Invalid device index: {x.index!r} '
                         f'must be {device_index!r}')
    return x


def validate_torch_device(x: Any, strict: bool = False) -> torch.device:
    print('validate_torch_device', x, strict)
    if isinstance(x, torch.device):
        return x
    elif strict:
        raise ValueError(f'Invalid torch device: {x!r}')
    try:
        return torch.device(x)
    except Exception as e:
        raise ValueError(f'Invalid torch device: {x!r}') from e


def serialize_torch_device(x: torch.device, nxt: SerializerFunctionWrapHandler) -> str:
    if x.index is not None:
        return nxt(f'{x.type}:{x.index}')
    return nxt(x.type)
