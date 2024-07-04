from .base import _BaseModel, configures
import re
import sys
from typing import ClassVar, Optional, Type
import torch
from pydantic import Field, model_validator, NonNegativeInt, SerializerFunctionWrapHandler


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


__ALL__ = ['TorchDevice']


@configures(torch.device)
class TorchDevice(_BaseModel[torch.device]):
    """TypedDict Config for torch.device"""
    re_match: ClassVar[re.Pattern] = re.compile(r'^[a-z]+:[0-9]+$')

    type: Literal[
        'cpu', 'cuda', 'fpga', 'hip', 'hpu', 'ideep', 'ipu',
        'lazy', 'meta', 'mkldnn', 'mps', 'mtia',  'opencl',
        'opengl', 'ort', 've', 'vulkan', 'xla', 'xpu'
    ] = Field(json_schema_extra=dict(required=True))

    index: Optional[NonNegativeInt] = Field(
        default=None, json_schema_extra=dict(required=False)
    )

    @model_validator(mode='before')
    @classmethod
    def validate_type_with_semicolon(cls, v: dict[str, str | int]):
        """Validate and parse device type with semicolon notation."""
        if isinstance(v, str):
            v = dict(type=v, index=None)
        if not isinstance(v, dict):
            return v
        if not 'type' in v:
            raise ValueError('type (string) must be passed explicitly')

        ty = v['type']
        if ':' not in ty:
            return v
        if v.get('index', None) is not None:
            raise ValueError('type (string) must not include an index '
                             f'because index was passed explicitly {ty!r}')

        if not cls.re_match.match(ty):
            raise ValueError(f'Invalid device string: {ty!r}')
        v['type'], v['index'] = ty.split(':')
        return v

    def _builder(self) -> Type[torch.device]:
        return torch.device

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
    def _serialize(v: torch.device, nxt: SerializerFunctionWrapHandler = lambda x: x) -> dict[str, str | int]:
        print('serializing torch device')
        return nxt(dict(type=v.type, index=v.index))
