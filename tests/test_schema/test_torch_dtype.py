import pytest
import torch
from pydantic import ValidationError, BaseModel, validate_call
from pydantic_pytorch.annotations.multiarray.torch_dtype import TorchDType


def test_valid_float32_dtype():
    dtype = TorchDType(type='torch.float32')
    assert dtype._build()() == torch.float32


def test_valid_float64_dtype():
    dtype = TorchDType(type='torch.float64')
    assert dtype._build()() == torch.float64


def test_invalid_dtype():
    with pytest.raises(ValidationError):
        TorchDType(type='torch.invalid')


def test_serialize_float32_dtype():
    dtype = TorchDType(type='torch.float32')
    serialized = dtype.model_dump()
    assert serialized == {'type': 'torch.float32'}


def test_serialize_float64_dtype():
    dtype = TorchDType(type='torch.float64')
    serialized = dtype.model_dump()
    assert serialized == {'type': 'torch.float64'}

