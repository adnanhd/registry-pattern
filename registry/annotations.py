from typing import get_type_hints, Callable, Any, Annotated
from pydantic_numpy import NpNDArray
from pydantic import SkipValidation
import numpy as np
from functools import partial
import torch


def annotate_numpy_array(func: Callable[..., Any]):
    annotations = get_type_hints(func)
    np_annotations = {k: NpNDArray for k, v in annotations.items() if v is np.ndarray}
    annotations.update(np_annotations)
    func.__annotations__ = annotations
    return func


def annotate_torch_tensor(func: Callable[..., Any]):
    annotations = get_type_hints(func)
    np_annotations = {k: Any for k, v in annotations.items() if v is torch.Tensor}
    annotations.update(np_annotations)
    func.__annotations__ = annotations
    
    return func

