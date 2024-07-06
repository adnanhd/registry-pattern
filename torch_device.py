from typing import Annotated
from pydantic import BaseModel
from pydantic_core import SchemaValidator, SchemaSerializer
from pydantic._internal import _known_annotated_metadata, _typing_extra, _validators
import dataclasses
from pydantic_pytorch.annotations.multiarray.torch_device import TORCH_DEVICE_TYPE
from typing_extensions import TypedDict, Literal, Required, Any
from pydantic import NonNegativeInt, ConfigDict, validate_call, GetCoreSchemaHandler, ValidatorFunctionWrapHandler, SerializerFunctionWrapHandler
from pydantic_core import core_schema, CoreSchema
from typing import Iterable
import torch

def to_string(x: Any, nxt: SerializerFunctionWrapHandler) -> str:
    return nxt(str(x))


int_validator = SchemaValidator({'type': 'int', 'ge': 0, 'serialization': 'int'}).validate_python
int_serializer = SchemaSerializer({'type': 'str', 'serialization': {'type': 'function-wrap', 'function': to_string}}).to_python
str_validator = SchemaValidator({'type': 'str', 'strict': False}).validate_python
