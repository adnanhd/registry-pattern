import re
import torch
import inspect
from typing import Any, Annotated, Literal
from functools import partial
from pydantic_core import core_schema, CoreSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic import GetCoreSchemaHandler, SerializerFunctionWrapHandler


def _dtype_to_str(v: torch.dtype) -> str:
    return str(v)[6:]

def _shorten_name(v: str) -> str:
    return v[0] + re.findall(r'[0-9]+', v)[0]

def _is_shortanable(v: str) -> bool:
    return re.match(r'^[a-z]+[0-9]+$', v) is not None

def _is_shortened_name(v: str) -> bool:
    return re.match(r'^[a-z][0-9]+$', v) is not None


_TORCH_DTYPE_MAPPING = {
    k: _dtype_to_str(v)
    for k, v in inspect.getmembers(torch)
    if isinstance(v, torch.dtype)
}


_SHORTENED_TORCH_DTYPE_MAPPING = {
    _shorten_name(v): v
    for v in _TORCH_DTYPE_MAPPING.values()
    if _is_shortanable(v)
}


TORCH_DTYPE_TYPE = Literal['float32', 'float64']


def validate_js_torch_dtype(x: str) -> torch.dtype:
    if x in _TORCH_DTYPE_MAPPING:
        return getattr(torch, _TORCH_DTYPE_MAPPING[x])
    elif x in _SHORTENED_TORCH_DTYPE_MAPPING:
        return getattr(torch, _SHORTENED_TORCH_DTYPE_MAPPING[x])
    raise ValueError(f'Invalid torch dtype: {x!r}')


def serialize_js_torch_dtype(x: torch.dtype, nxt: SerializerFunctionWrapHandler) -> str:
    x = _dtype_to_str(x)
    if _is_shortanable(x):
        x = _shorten_name(x)
    return nxt(x)


def validate_py_torch_dtype(x: Any, strict: bool = False) -> torch.dtype:
    if isinstance(x, torch.dtype):
        return x
    elif strict:
        raise ValueError(f'Invalid torch dtype: {x!r}')
    return validate_js_torch_dtype(x)


def serialize_py_torch_dtype(x: torch.dtype, nxt: SerializerFunctionWrapHandler) -> torch.dtype:
    return nxt(x)


def torch_dtype_schema(
    dtype: TORCH_DTYPE_TYPE | None = None,
    strict: bool = True
) -> CoreSchema:

    schema = core_schema.is_instance_schema(torch.dtype)
    constraints = []

    def validate_torch_device_(x: torch.device) -> torch.device:
        str_x = _dtype_to_str(x)
        if str_x != dtype:
            raise ValueError(f'Invalid dtype: {str_x!r} '
                             f'must be {dtype!r}')
        return x


    if dtype is not None:
        constraints.append(
            core_schema.no_info_after_validator_function(
                function=validate_torch_device_,
                schema=schema
            )
        )

    deserializer = core_schema.no_info_before_validator_function(
            function=partial(validate_py_torch_dtype, strict=strict),
            schema=schema
        )

    python_schema = core_schema.chain_schema([deserializer, schema, *constraints])
    python_schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(
        serialize_py_torch_dtype,
        schema=core_schema.str_schema(),
        info_arg=False
    )

    deserializer = core_schema.no_info_before_validator_function(
            function=validate_js_torch_dtype,
            schema=schema
        )
    json_schema = core_schema.chain_schema([deserializer, schema, *constraints])
    json_schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(
        serialize_js_torch_dtype,
        schema=core_schema.str_schema(),
        info_arg=False
    )

    return core_schema.json_or_python_schema(json_schema=json_schema, python_schema=python_schema)
