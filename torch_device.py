import torch
from typing import Annotated
from pydantic import BaseModel, InstanceOf, Field
from pydantic_pytorch._schema import pytorch_schema
import dataclasses
from validator import CoreSchemaValidator



class MyModel(BaseModel):
    lst: CoreSchemaValidator[torch.device, pytorch_schema.torch_device_schema(device_index=2, device_type='cuda', strict=True)]


ff = MyModel.model_validate_json(bytes('{"lst": "cuda:2"}', 'utf-8'))
gg = MyModel.model_validate({"lst": "cuda:2"})
hh = MyModel(lst='cuda:2')