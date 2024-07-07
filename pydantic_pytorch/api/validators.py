import torch
import dataclasses
from typing import Any
from .source_types import TORCH_DEVICE_TYPE, TORCH_DEVICE_INDEX
from pydantic_core import core_schema, CoreSchema
from pydantic import ValidatorFunctionWrapHandler, SerializerFunctionWrapHandler, GetCoreSchemaHandler


@dataclasses.dataclass(slots=True)
class TorchDeviceValidator:
    device_type: TORCH_DEVICE_TYPE = None
    device_index: TORCH_DEVICE_INDEX = None

    def torch_device_type_validator(self, x: torch.device, handler: ValidatorFunctionWrapHandler) -> torch.device | None:
        if (x := handler(x)) and x.type != self.device_type:
            raise ValueError(f'Invalid device type: {x.type!r} '
                             f'must be {self.device_type!r}')
        return x

    def torch_device_index_validator(self, x: torch.device, handler: ValidatorFunctionWrapHandler) -> torch.device | None:
        if (x := handler(x)) and x.index != self.device_index:
            raise ValueError(f'Invalid device index: {x.index!r} '
                             f'must be {self.device_index!r}')
        return x

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        # TODO: no validation for device_type and device_index
        print('hello!!!')

        serialization = core_schema.wrap_serializer_function_ser_schema(serialize_torch_device)
        schema = core_schema.is_instance_schema(torch.device, serialization=serialization)
        configuration = core_schema.no_info_before_validator_function(validate_torch_device, schema=schema)
        validators = [configuration, schema]

        if self.device_type is not None:
            validators.append(core_schema.no_info_wrap_validator_function(
                self.torch_device_type_validator, schema=schema))

        if self.device_index is not None:
            validators.append(core_schema.no_info_wrap_validator_function(
                self.torch_device_index_validator, schema=schema))

        return core_schema.chain_schema(validators)


def serialize_torch_device(x: torch.device, nxt: SerializerFunctionWrapHandler) -> str:
    if x.index is not None:
        return nxt(f'{x.type}:{x.index}')
    return nxt(x.type)


def validate_torch_device(x: Any, strict : bool = False) -> torch.device:
    if isinstance(x, torch.device):
        return x
    elif strict:
        raise ValueError(f'Invalid torch device: {x!r}')
    try:
        return torch.device(x)
    except Exception as e:
        raise ValueError(f'Invalid torch device: {x!r}') from e
