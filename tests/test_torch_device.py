import pytest
from pydantic import ValidationError, BaseModel, validate_call
import torch
from typing import Dict

from pydantic_pytorch.typing.multiarray.torch_device import TorchDevice


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


def test_invalid_device_string_with_index():
    with pytest.raises(ValidationError):
        TorchDevice(type='cuda:1', index=1)


def test_validate_instance_of_torch_device():
    torch_device = torch.device('cuda', 0)
    device = TorchDevice.validate_torch_device(torch_device)
    assert device.type == 'cuda'
    assert device.index == 0


def test_serialize_torch_device():
    torch_device = torch.device('cuda', 0)
    serialized = TorchDevice.serialize_torch_device(torch_device, lambda x: x)
    assert serialized == {'type': 'cuda', 'index': 0}

def test_validate_config_and_instance():
    torch_device_cfg = dict(type='cuda', index=2)
    torch_device_ins = torch.device(type='cuda', index=2)

    assert torch_device_ins == TorchDevice.model_validate(torch_device_cfg)
    assert torch_device_ins == TorchDevice.model_validate(torch_device_ins).build_model()

def test_validate_model_and_instance():
    torch_device_cfg = dict(type='cuda', index=2)
    torch_device_ins = torch.device(type='cuda', index=2)
    torch_device_mdl = TorchDevice(**torch_device_cfg)

    assert torch_device_ins == torch_device_mdl.build_model()
    assert torch_device_ins == TorchDevice.model_validate(torch_device_mdl)


class TorchTensor(BaseModel):
    device: TorchDevice


def test_torch_device_within_model():
    torch_device = torch.device('cuda')
    assert torch_device == TorchTensor(device=torch_device).device.build_model()


@validate_call
def example_function(device: TorchDevice) -> None:
    pass


def test_torch_device_within_function():
    torch_device = torch.device('cuda')
    example_function(torch_device)


def test_non_torch_device_within_function():
    with pytest.raises(ValidationError):
        example_function(2)
