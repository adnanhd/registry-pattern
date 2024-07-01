import torch
import pytest
from pydantic import ValidationError
from pydantic_pytorch.typing.multiarray import TorchShape, TorchDevice, TorchDType  # Adjust import paths
from pydantic_pytorch.typing import TorchTensor  # Adjust import paths

class ValidTensor(TorchTensor):
    shape = TorchShape(shape=[5, 10])
    device = TorchDevice(type='cpu')
    dtype = TorchDType(type='torch.float32')
    requires_grad = False


def test_valid_tensor():
    tensor = torch.randn(5, 10, dtype=torch.float32, device='cpu', requires_grad=False)
    model = ValidTensor(data=tensor)
    assert isinstance(model.data, torch.Tensor)
    assert model.data.shape == (5, 10)
    assert model.data.device == torch.device('cpu')
    assert model.data.dtype == torch.float32
    assert model.data.requires_grad == False


class InvalidTensor(TorchTensor):
    shape = TorchShape(shape=[5, 10])
    device = TorchDevice(type='cpu')
    dtype = TorchDType(type='torch.float32')
    requires_grad = False


def test_invalid_tensor():
    tensor = torch.randn(5, 10, dtype=torch.float32, device='cuda', requires_grad=False)
    with pytest.raises(ValidationError):
        InvalidTensor(data=tensor)