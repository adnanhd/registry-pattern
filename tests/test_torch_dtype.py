import pytest
import torch
from pydantic import ValidationError, BaseModel, validate_call
from pydantic_pytorch.annotations.multiarray.torch_dtype import TorchDType


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


def test_validate_config_and_instance():
    torch_dtype_cfg = dict(type='torch.float32')
    torch_dtype_ins = torch.float32

    assert torch_dtype_ins == TorchDType.model_validate(torch_dtype_cfg)
    assert torch_dtype_ins == TorchDType.model_validate(
        torch_dtype_ins).build_model()


def test_validate_model_and_instance():
    torch_dtype_cfg = dict(type='torch.float32')
    torch_dtype_ins = torch.float32
    torch_dtype_mdl = TorchDType(**torch_dtype_cfg)

    assert torch_dtype_ins == torch_dtype_mdl.build_model()
    assert torch_dtype_ins == TorchDType.model_validate(torch_dtype_mdl)


class TorchTensor(BaseModel):
    dtype: TorchDType


def test_torch_dtype_within_model():
    torch_dtype = torch.float32
    assert torch_dtype == TorchTensor(dtype=torch_dtype).dtype.build_model()


@validate_call
def example_function(dtype: TorchDType) -> None:
    pass


def test_torch_dtype_within_function():
    torch_dtype = torch.float32
    example_function(torch_dtype)


def test_non_torch_dtype_within_function():
    with pytest.raises(ValidationError):
        example_function("invalid_dtype")
