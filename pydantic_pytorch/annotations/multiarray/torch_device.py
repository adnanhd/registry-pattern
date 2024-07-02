import re
import sys
from typing import Any, Dict, Optional
import torch
from pydantic import Field, ValidationInfo, field_validator, NonNegativeInt, SerializerFunctionWrapHandler
from pydantic_core import core_schema
from .base import _BaseTypedDictAnnotation
import dataclasses
import pydantic
# from dataclasses import dataclass
# from pydantic.dataclasses import dataclass

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


__ALL__ = ['TorchDevice']

class TorchDevice(pydantic.BaseModel):
    """Dataclass for torch.device"""

    type: Literal['cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 've',
                  'fpga', 'ort', 'xla', 'lazy', 'vulkan', 'mps', 'meta', 'hpu', 'mtia'] = None
    index: Optional[NonNegativeInt] = None

    @field_validator('index')
    @classmethod
    def validate_index(cls, index: int, info: ValidationInfo) -> int:
        values: dict[str, str | int] = info.data
        if index is not None and values.get('type', 0) is None:
            raise ValueError(f'Invalid index: {index!r}')
        return index

    def build_model(self) -> torch.device:
        print('validating model')
        return torch.device(self.type, self.index)

    def _validate(self, v: torch.device) -> torch.device:
        print('validating instance of torch device')
        if self.type is not None and self.type != v.type:
            raise ValueError(
                f'Invalid device type: {v.type!r} must be {self.type!r}')
        if self.index is not None and self.index != v.index:
            raise ValueError(
                f'Invalid device index: {v.index!r} must be {self.index!r}')
        return v

    @staticmethod
    def _serialize(v: torch.device, nxt: SerializerFunctionWrapHandler) -> dict[str, str | int]:
        print('serializing torch device')
        return nxt(dict(type=v.type, index=v.index))


def _validate_semicolon_in_type(values: dict[str, str | int]) -> dict[str, str | int]:
    print('validating semicolon in type')
    assert isinstance(values, dict), f'Invalid type: {type(values)!r}'
    assert 'type' in values, 'type (string) must be passed explicitly'

    if ':' in values['type']:
        ty = values['type']
        idx = values.get('index', None)
        # define a proper regex for this
        if not re.match(r'^[a-z]+:[0-9]+$', ty):
            raise ValueError(f'Invalid device string: {ty!r}')
        if idx is not None:
            raise ValueError('type (string) must not include an index '
                             f'because index was passed explicitly {ty!r}')
        ty, idx = ty.split(':')
        values['type'] = ty
        values['index'] = int(idx)

    return values

from typing import Annotated
from pydantic import BeforeValidator, AfterValidator, InstanceOf, WrapSerializer
class TorchDeviceAnnotated:

    @classmethod
    def __class_getitem__(cls, item: Any) -> Any:
        torch_device = TorchDevice.model_validate(item)
        return Annotated[torch.device, InstanceOf,
                        BeforeValidator(TorchDeviceAnnotated._validate_from_config),
                        BeforeValidator(TorchDeviceAnnotated._validate_from_string),
                        AfterValidator(torch_device._validate),
                         WrapSerializer(torch_device._serialize)]

    @staticmethod
    def _validate_from_config(values: dict[str, str | int]) -> torch.device:
        if isinstance(values, dict) and 'type' in values:
            print('validating config')
            values = _validate_semicolon_in_type(values)
            return torch.device(**values)
        return values

    @staticmethod
    def _validate_from_string(values: str) -> torch.device:
        if isinstance(values, str):
            print('validating string')
            values = dict(type=values, index=None)
            values = _validate_semicolon_in_type(values)
            return torch.device(**values)
        return values
