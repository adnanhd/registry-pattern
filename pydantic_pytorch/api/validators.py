import torch
import dataclasses
from typing import Any
from .source_types import TORCH_DEVICE_TYPE, TORCH_DEVICE_INDEX
from pydantic_core import core_schema, CoreSchema
from pydantic import ValidatorFunctionWrapHandler, GetCoreSchemaHandler


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
        import pdb; pdb.set_trace()

        schema = core_schema.is_instance_schema(torch.device)
        validators = [schema]

        if self.device_type is not None:
            validators.append(core_schema.no_info_wrap_validator_function(
                self.torch_device_type_validator, schema=schema))

        if self.device_index is not None:
            validators.append(core_schema.no_info_wrap_validator_function(
                self.torch_device_index_validator, schema=schema))

        return core_schema.chain_schema(validators)