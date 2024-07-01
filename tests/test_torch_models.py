import pytest
import abc
import torch
from torch import nn, optim
from pydantic import ValidationError, BaseModel, InstanceOf
from pydantic_pytorch.models import TorchModule, TorchOptimizer  # Adjust import paths
# Adjust import paths
from pydantic_pytorch.models.base import PydanticClassRegistryBaseModel


class TrainerModel(BaseModel):
    model: TorchModule


def test_torch_module_registration():
    # Ensure that torch.nn.Module and torch.nn.Linear are registered
    assert issubclass(torch.nn.Linear, TorchModule)
    assert TorchModule.has_registered(torch.nn.Linear)

    # Create an instance of TorchModule
    mdl = TorchModule(type='Linear', in_features=10, out_features=1)
    assert isinstance(mdl, PydanticClassRegistryBaseModel)
    assert mdl.type == 'Linear'


def test_torch_optimizer_registration():
    # Ensure that torch.optim.Optimizer and torch.optim.Adam are registered
    assert issubclass(torch.optim.Adam, TorchOptimizer)
    assert TorchOptimizer.has_registered(torch.optim.Adam)

    # Create a model and an optimizer
    model = nn.Linear(10, 1)
    assert abc._abc_instancecheck(TorchModule, model)
    optimizer = TorchOptimizer(type='Adam', params=dict(
        type='Linear', in_features=10, out_features=1), lr=0.01)
    assert isinstance(optimizer, PydanticClassRegistryBaseModel)
    assert optimizer.type == 'Adam'
    assert optimizer.lr == 0.01


def test_initialize_trainer_model_with_valid_config():
    config = dict(type='Linear', in_features=10, out_features=1)
    
    trainer = TrainerModel(model=config)
    assert trainer.model == torch.nn.Linear(10, 1), f"Expected {torch.nn.Linear(10, 1)}, got {trainer.model}"
    assert isinstance(trainer.model, TorchModule), f"Expected {TorchModule}, got {type(trainer.model)}"
    assert trainer.model.type == 'Linear', f"Expected Linear, got {trainer.model.type}"
    assert trainer.model.in_features == 10, f"Expected 10, got {trainer.model.in_features}"
    assert trainer.model.out_features == 1, f"Expected 1, got {trainer.model.out_features}"

def test_initialize_trainer_model_with_valid_module():
    model = nn.Linear(10, 1)
    
    trainer = TrainerModel(model=model)
    assert isinstance(trainer.model, TorchModule), f"Expected {TorchModule}, got {type(trainer.model)}"
    assert trainer.model.type == 'Linear', f"Expected Linear, got {trainer.model.type}"
    assert trainer.model.in_features == 10, f"Expected 10, got {trainer.model.in_features}"
    assert trainer.model.out_features == 1, f"Expected 1, got {trainer.model.out_features}"


"""
def test_get_parameters_with_valid_module():
    model = nn.Linear(10, 1)
    params = list(get_parameters(model))
    assert len(params) > 0
    assert all(isinstance(p, nn.Parameter) for p in params)

def test_get_parameters_with_invalid_module():
    with pytest.raises(ValueError, match="Invalid module:"):
        get_parameters("invalid_module")

def test_torch_optimizer_with_invalid_params():
    with pytest.raises(ValidationError):
        TorchOptimizer(type='torch.optim.Adam', params="invalid_params", lr=0.01)

def test_torch_optimizer_with_invalid_lr():
    model = nn.Linear(10, 1)
    with pytest.raises(ValidationError):
        TorchOptimizer(type='torch.optim.Adam', params=model, lr=-0.01)

def test_torch_optimizer_with_valid_lr():
    model = nn.Linear(10, 1)
    optimizer = TorchOptimizer(type='torch.optim.Adam', params=model, lr=0.01)
    assert optimizer.lr == 0.01

if __name__ == '__main__':
    pytest.main()
"""
