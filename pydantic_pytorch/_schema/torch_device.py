import dataclasses
from typing import Any, Annotated, Literal
from functools import partial
from annotated_types import Ge
from torch import device as torch_device
from pydantic_core import core_schema, CoreSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic import GetCoreSchemaHandler, SerializerFunctionWrapHandler

TORCH_DEVICE_INDEX = Annotated[int, Ge(0)] | None

TORCH_DEVICE_TYPE = Literal['cpu', 'cuda', 'fpga', 'hip', 'hpu', 'ideep', 'ipu',
                            'lazy', 'meta', 'mkldnn', 'mps', 'mtia',  'opencl',
                            'opengl', 'ort', 've', 'vulkan', 'xla', 'xpu']


def validate_js_torch_device(x: str) -> torch_device:
    try:
        return torch_device(x)
    except Exception as e:
        raise ValueError(f'Invalid torch device: {x!r}') from e


def torch_device_schema(
    device_type: TORCH_DEVICE_TYPE | None = None,
    device_index: TORCH_DEVICE_INDEX | None = None,
    strict: bool = True
) -> CoreSchema:

    schema = core_schema.is_instance_schema(torch_device)
    constraints = []

    def validate_torch_device_type(x: torch_device) -> torch_device:
        if x.type != device_type:
            raise ValueError(f'Invalid device type: {x.type!r} '
                             f'must be {device_type!r}')
        return x

    def validate_torch_device_index(x: torch_device) -> torch_device:
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
        function=validate_js_torch_device,
        schema=schema
    )

    if strict:
        if len(constraints) == 0:
            python_schema = schema
        else:
            python_schema = core_schema.chain_schema(constraints)
    else:
        python_schema = core_schema.chain_schema([deserializer, *constraints])
    python_schema['serialization'] = core_schema.plain_serializer_function_ser_schema(
        lambda x: x,
    )

    json_schema = core_schema.chain_schema([deserializer, *constraints])
    json_schema['serialization'] = core_schema.plain_serializer_function_ser_schema(
        lambda x: x.type if x.index is None else f'{x.type}:{x.index}',
    )

    return core_schema.json_or_python_schema(
        json_schema=json_schema,
        python_schema=python_schema
    )


@dataclasses.dataclass
class TorchDeviceValidator:
    device_type: TORCH_DEVICE_TYPE | None = None
    device_index: TORCH_DEVICE_INDEX | None = None
    strict: bool = False

    def __get_pydantic_core_schema__(self, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:
        # validate arguments
        return torch_device_schema(
            device_type=self.device_type,
            device_index=self.device_index,
            strict=self.strict
        )
