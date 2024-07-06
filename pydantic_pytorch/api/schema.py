from typing import Any, Iterable, TypedDict
from typing_extensions import Literal, Required
from pydantic import ConfigDict, validate_call
from .source_types import TORCH_DEVICE_TYPE, TORCH_DEVICE_INDEX


def _dict_not_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


class TorchDeviceSchema(TypedDict, total=False):
    type: Required[Literal['torch-device']]
    to_device: TORCH_DEVICE_TYPE
    to_index: TORCH_DEVICE_INDEX
    strict: bool


@validate_call
def torch_device_schema(
    to_device: TORCH_DEVICE_TYPE | None = None,
    to_index: TORCH_DEVICE_INDEX | None = None,
    strict: bool | None = None
) -> TorchDeviceSchema:
    return _dict_not_none(type='torch-device', to_device=to_device, to_index=to_index, strict=strict)


def torch_device_prepare_pydantic_annotations(
    source_type: Any, annotations: Iterable[Any], _config: ConfigDict
) -> tuple[Any, list[Any]] | None:
    pass
