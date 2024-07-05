from typing import Any, List, Type
from pydantic import BaseModel, NonNegativeInt, Field, SerializerFunctionWrapHandler
from pydantic_core import core_schema
import pydantic
import torch
from .base import _BaseModel

TORCH_SHAPE_TYPE = List[NonNegativeInt]


class TorchShape(_BaseModel[torch.Size]):
    """TypedDict Config for torch.device"""
    shape: TORCH_SHAPE_TYPE = Field(json_schema_extra=dict(required=True))

    @pydantic.model_validator(mode='before')
    @staticmethod
    def validate_string(values: list) -> dict[str, list]:
        print('validate model')
        if isinstance(values, list):
            return dict(shape=values)
        return values

    def _builder(self) -> Type[torch.Size]:
        return lambda shape: torch.Size(shape)

    def _validate(self, v: torch.Size) -> torch.Size:
        print('validating instance of torch device')
        if self.shape is not None and self.shape != list(v):
            raise ValueError(
                f'Invalid shape: {list(v)!r} must be {self.shape!r}')

    @staticmethod
    def _serialize(v: torch.Size, nxt: SerializerFunctionWrapHandler = lambda x: x) -> dict[str, TORCH_SHAPE_TYPE]:
        print('serializing torch device')
        return nxt(dict(shape=list(v)))
