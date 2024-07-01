from typing import Any, Sequence, List
from pydantic import BaseModel, NonNegativeInt, ValidationError, root_validator, Field
from pydantic_core import core_schema
from typing_extensions import Annotated
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
    def validate_instance_of_torch_shape(cls, v: TORCH_SHAPE_TYPE) -> torch.Size:
        return cls(shape=v).build_model()

    @staticmethod
    def serialize_torch_shape(v: TORCH_SHAPE_TYPE, nxt: Any) -> dict:
        return nxt({'shape': list(v)})

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        model_schema = handler(source_type)
        instance_validation = core_schema.no_info_after_validator_function(
            cls.validate_instance_of_torch_shape,
            core_schema.is_instance_schema(torch.Size),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                cls.serialize_torch_shape)
        )
        return core_schema.union_schema([model_schema, instance_validation])
