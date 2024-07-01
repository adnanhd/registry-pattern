import pytest
import torch
from pydantic import ValidationError, BaseModel, validate_call
from pydantic_pytorch.typing.multiarray.torch_dtype import TorchDType

def test_valid_float32_dtype():
    dtype = TorchDType(type='torch.float32')
    assert dtype.build_model() == torch.float32

def test_valid_float64_dtype():
    dtype = TorchDType(type='torch.float64')
    assert dtype.build_model() == torch.float64

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

def test_validate_instance_of_torch_dtype():
    torch_dtype = torch.float32
    dtype = TorchDType.validate_instance_of_torch_dtype(torch_dtype)
    assert dtype == torch_dtype


def test_validate_twice():
    torch_dtype_str = 'torch.float32'
    torch_dtype_ins = torch.float32

    torch_dtype = TorchDType.model_validate(dict(type=torch_dtype_str))
    assert torch_dtype == TorchDType.model_validate(torch_dtype)


class TorchTensor(BaseModel):
    dtype: TorchDType


def test_torch_dtype_within_model():
    torch_dtype = torch.float32
    assert torch_dtype == TorchTensor(dtype=torch_dtype).dtype

@validate_call
def example_function(dtype: TorchDType) -> None:
    pass

def test_torch_dtype_within_function():
    torch_dtype = torch.float32
    example_function(torch_dtype)

def test_non_torch_dtype_within_function():
    with pytest.raises(ValidationError):
        example_function("invalid_dtype")

