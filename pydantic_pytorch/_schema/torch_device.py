from typing import Any, Annotated, Literal
from functools import partial
from annotated_types import Ge
from pydantic_core import core_schema, CoreSchema
from pydantic.json_schema import JsonSchemaValue
import torch

TORCH_DEVICE_INDEX = Annotated[int, Ge(0)] | None

TORCH_DEVICE_TYPE = Literal['cpu', 'cuda', 'fpga', 'hip', 'hpu', 'ideep', 'ipu',
                            'lazy', 'meta', 'mkldnn', 'mps', 'mtia',  'opencl',
                            'opengl', 'ort', 've', 'vulkan', 'xla', 'xpu']


import torch
from typing import Any
from functools import partial
from pydantic import GetCoreSchemaHandler, SerializerFunctionWrapHandler


def validate_js_torch_device(x: str) -> torch.device:
    try:
        return torch.device(x)
    except Exception as e:
        raise ValueError(f'Invalid torch device: {x!r}') from e


def serialize_js_torch_device(x: torch.device, nxt: SerializerFunctionWrapHandler) -> str:
    if x.index is not None:
        return nxt(f'{x.type}:{x.index}')
    return nxt(x.type)


def validate_py_torch_device(x: Any, strict: bool = False) -> torch.device:
    if isinstance(x, torch.device):
        return x
    elif strict:
        raise ValueError(f'Invalid torch device: {x!r}')
    return validate_js_torch_device(x)


def serialize_py_torch_device(x: torch.device, nxt: SerializerFunctionWrapHandler) -> torch.device:
    return nxt(x)


def torch_device_schema(
    device_type: TORCH_DEVICE_TYPE | None = None,
    device_index: TORCH_DEVICE_INDEX | None = None,
    strict: bool = True
) -> CoreSchema:
    
    schema = core_schema.is_instance_schema(torch.device)
    constraints = []

    def validate_torch_device_type(x: torch.device) -> torch.device:
        if x.type != device_type:
            raise ValueError(f'Invalid device type: {x.type!r} '
                             f'must be {device_type!r}')
        return x

    def validate_torch_device_index(x: torch.device) -> torch.device:
        if x.index != device_index:
            raise ValueError(f'Invalid device index: {x.index!r} '
                             f'must be {device_index!r}')
        return x
    
    if device_type is not None:
        constraints.append(
            core_schema.no_info_after_validator_function(
                function=validate_torch_device_type,
                schema=schema
            )
        )

    if device_index is not None:
        constraints.append(
            core_schema.no_info_after_validator_function(
                function=validate_torch_device_index,
                schema=schema
            )
        )

    deserializer = core_schema.no_info_before_validator_function(
            function=partial(validate_py_torch_device, strict=strict),
            schema=schema
        )
    
    python_schema = core_schema.chain_schema([deserializer, schema, *constraints])
    python_schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(
        serialize_py_torch_device,
        schema=core_schema.str_schema(),
        info_arg=False
    )

    deserializer = core_schema.no_info_before_validator_function(
            function=validate_js_torch_device,
            schema=schema
        )
    json_schema = core_schema.chain_schema([deserializer, schema, *constraints])
    json_schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(
        serialize_js_torch_device,
        schema=core_schema.str_schema(),
        info_arg=False
    )

    return core_schema.json_or_python_schema(json_schema=json_schema, python_schema=python_schema)
