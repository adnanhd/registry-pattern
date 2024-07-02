from typing import Any, List
from pydantic import BaseModel, NonNegativeInt, Field
from pydantic_core import core_schema
import pydantic
import torch

TORCH_SHAPE_TYPE = List[NonNegativeInt]


class TorchShape(BaseModel):
    """Model for validating tensor shapes."""
    shape: TORCH_SHAPE_TYPE = Field(..., description="Sequence of positive integers representing the shape of the tensor")

    @pydantic.model_validator(mode='after')
    def build_model(self) -> torch.Size:
        return torch.Size(self.shape)

    @classmethod
    def validate_torch_shape(cls, v: TORCH_SHAPE_TYPE) -> 'TorchShape':
        return cls(shape=v)

    @staticmethod
    def serialize_torch_shape(v: TORCH_SHAPE_TYPE, nxt: Any) -> dict:
        return nxt({'shape': list(v)})

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        instance_validation = core_schema.no_info_after_validator_function(
            cls.validate_torch_shape,
            core_schema.is_instance_schema(torch.Size),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                cls.serialize_torch_shape)
        )
        return core_schema.union_schema([handler(source_type), instance_validation])
