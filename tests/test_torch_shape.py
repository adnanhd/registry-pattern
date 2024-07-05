import pytest
import torch
from pydantic import ValidationError, BaseModel
from pydantic_pytorch.annotations.multiarray.torch_shape import TorchShape
from pydantic import validate_call


def test_valid_shape():
    shape = TorchShape(shape=[3, 3])
    assert shape._build()() == torch.Size([3, 3])

def test_invalid_shape_not_sequence():
    with pytest.raises(ValidationError):
        TorchShape(shape=3)

def test_invalid_shape_negative_dimension():
    with pytest.raises(ValidationError):
        TorchShape(shape=[3, -3])

def test_serialize_shape():
    shape = TorchShape(shape=[3, 3])
    serialized = shape.model_dump()
    assert serialized == {'shape': [3, 3]}

def test_validate_torch_shape():
    shape = [3, 3]
    torch_shape = TorchShape.model_validate(shape)._build()()
    assert torch_shape == torch.Size([3, 3])
    