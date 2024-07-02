import re
import sys
from typing import Any, Dict, Optional
import torch
import pydantic
from pydantic import Field, GetCoreSchemaHandler, SerializerFunctionWrapHandler, BaseModel
from pydantic_core import core_schema

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


__ALL__ = ['TorchDevice']


class TorchDevice(BaseModel):
    """TypedDict for torch.device"""

    type: Literal['cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 've',
                  'fpga', 'ort', 'xla', 'lazy', 'vulkan', 'mps', 'meta', 'hpu', 'mtia']
    index: Optional[int] = None

    @pydantic.model_validator(mode='after')
    def build_model(self):
        print('validating model')
        return torch.device(self.type, self.index)

    @pydantic.model_validator(mode='before')
    @classmethod
    def validate_string_without_device(cls, values: Any) -> Dict[str, Any]:
        print('validate_string_without_device')
        if isinstance(values, str):
            return dict(type=values, index=None)
        return values

    @pydantic.model_validator(mode='wrap')
    @classmethod
    def validate_semicolon(cls, values: Any, handler: Any) -> Dict[str, Any]:
        print('validate_semicolon')
        if isinstance(values, dict) and 'type' in values and ':' in values['type']:
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
        return handler(values)

    @classmethod
    def validate_torch_device(cls, v: torch.device) -> 'TorchDevice':
        print('validating instance of torch device')
        return cls(type=v.type, index=v.index if v.index != -1 else None)

    @staticmethod
    def serialize_torch_device(v: torch.device, nxt: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        print('serializing')
        return nxt(dict(type=v.type, index=v.index))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler):
        instance_validation = core_schema.no_info_after_validator_function(
            cls.validate_torch_device,
            core_schema.is_instance_schema(torch.device),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                cls.serialize_torch_device)
        )
        return core_schema.union_schema([handler(source_type), instance_validation])
