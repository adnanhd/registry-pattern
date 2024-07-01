from typing import Any, Sequence, Dict, Generic, TypeVar, Type, Annotated, Literal, Optional
import torch
import numpy as np
import pydantic
from pydantic import Field, GetCoreSchemaHandler, SerializerFunctionWrapHandler, ValidatorFunctionWrapHandler, AfterValidator, WrapValidator, BaseModel, ValidationInfo, NonNegativeInt
from pydantic_core import core_schema
from dataclasses import dataclass
import annotated_types

from typing_extensions import TypedDict
from pydantic import TypeAdapter


__ALL__ = ['TorchLayout']

class TorchLayout(BaseModel):
    """TypedDict for torch.device"""

    type: Literal['torch.strided', 'torch.sparse_coo', 'torch._mkldnn']
    
    # TODO: buraya bir sistematik bir ekleme yontemi lazim 