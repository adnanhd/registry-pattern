import pytest
from pydantic import ValidationError, BaseModel, validate_call
import torch
from typing import Dict

from pydantic_pytorch._schema.torch_device import TorchDevice


def test_valid_cpu_device():
    device = TorchDevice(type='cpu')
    assert device.type == 'cpu'
    assert device.index is None


def test_valid_cuda_device():
    device = TorchDevice(type='cuda', index=0)
    assert device.type == 'cuda'
    assert device.index == 0


def test_invalid_cuda_device_without_index():
    device = TorchDevice(type='cuda')
    assert device.type == 'cuda'
    assert device.index == None


def test_missing_device_type():
    with pytest.raises(ValidationError):
        TorchDevice(index=0)


def test_invalid_device_type_format():
    with pytest.raises(ValidationError):
        TorchDevice(type='xlmf')


def test_invalid_device_index_format():
    with pytest.raises(ValidationError):
        TorchDevice(type='cuda:?')


def test_valid_device_string_format():
    device = TorchDevice(type='cuda:1')
    assert device.type == 'cuda'
    assert device.index == 1


def test_valid_device_string_format_multiple_colons():
    with pytest.raises(ValidationError):
        device = TorchDevice(type='cuda:1:2')


def test_invalid_device_string_with_index():
    with pytest.raises(ValidationError):
        TorchDevice(type='cuda:1', index=1)


def test_valid_device_string_format_with_model_validate():
    device = TorchDevice.model_validate('cuda:1')
    assert device.type == 'cuda'
    assert device.index == 1


def test_invalid_device_dict_with_same_index_and_model_validate():
    with pytest.raises(ValidationError):
        TorchDevice.model_validate({'type':'cuda:1', 'index':1})



def test_invalid_device_dict_with_different_index_and_model_validate():
    with pytest.raises(ValidationError):
        TorchDevice.model_validate({'type':'cuda:1', 'index':2})
