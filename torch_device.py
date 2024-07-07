from typing import Annotated
from pydantic import BaseModel
from pydantic_core import SchemaValidator, SchemaSerializer
from pydantic._internal import _known_annotated_metadata, _typing_extra, _validators
import dataclasses
from pydantic_pytorch.annotations.multiarray.torch_device import TORCH_DEVICE_TYPE
from typing_extensions import TypedDict, Literal, Required, Any
from pydantic import NonNegativeInt, ConfigDict, validate_call, GetCoreSchemaHandler, ValidatorFunctionWrapHandler, SerializerFunctionWrapHandler, InstanceOf, PositiveInt
from pydantic_core import core_schema, CoreSchema
from typing import Iterable
import torch
from pydantic_pytorch.api.schema import TorchDeviceSchema
from pydantic_pytorch.api.validators import TorchDeviceValidator


def to_string(x: Any, nxt: SerializerFunctionWrapHandler) -> str:
    return nxt(str(x))


def serialize_torch_device(x: torch.device, nxt: SerializerFunctionWrapHandler) -> str:
    print('serialize_torch_device', x)
    if x.index is not None:
        return nxt(f'{x.type}:{x.index}')
    return nxt(x.type)


int_validator_schema = {'type': 'int', 'ge': 0}
int_serializer_schema = {'type': 'str', 'serialization': {'type': 'function-wrap', 'function': to_string}}

int_validator = SchemaValidator(int_validator_schema)
int_serializer = SchemaSerializer(int_serializer_schema)


torch_device_validator_schema = core_schema.is_instance_schema(torch.device)
torch_device_serializer_schema = {'type': 'str', 'serialization': {'type': 'function-wrap', 'function': serialize_torch_device}}

torch_device_validator = SchemaValidator(torch_device_validator_schema)
torch_device_serializer = SchemaSerializer(torch_device_serializer_schema)


class MyModel(BaseModel):
    a: TorchDeviceValidator(device_type='cuda')
