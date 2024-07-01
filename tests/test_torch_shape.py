import pytest
import torch
from pydantic import ValidationError, BaseModel
from pydantic_pytorch.typing.multiarray.torch_shape import TorchShape
from pydantic import validate_call

class TensorWithShape(BaseModel):
    shape: TorchShape

def test_valid_shape():
    shape = TorchShape(shape=[3, 3])
    assert shape.build_model() == torch.Size([3, 3])

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

def test_validate_instance_of_torch_shape():
    shape = [3, 3]
    torch_shape = TorchShape.validate_instance_of_torch_shape(shape)
    assert torch_shape == torch.Size([3, 3])

def test_torch_shape_within_model():
    shape = [3, 3]
    tensor_with_shape = TensorWithShape(shape=TorchShape(shape=shape))
    assert tensor_with_shape.shape == torch.Size([3, 3])

@validate_call
def example_function(shape: TorchShape) -> None:
    pass

def test_torch_shape_within_function():
    shape = [3, 3]
    example_function(TorchShape(shape=shape))

def test_non_torch_shape_within_function():
    with pytest.raises(ValidationError):
        example_function(TorchShape(shape=[3, -3]))

if __name__ == '__main__':
    pytest.main()
