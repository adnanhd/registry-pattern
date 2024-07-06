from typing import Literal
from pydantic import NonNegativeInt


TORCH_DEVICE_TYPE = Literal[
    'cpu', 'cuda', 'fpga', 'hip', 'hpu', 'ideep', 'ipu',
    'lazy', 'meta', 'mkldnn', 'mps', 'mtia',  'opencl',
    'opengl', 'ort', 've', 'vulkan', 'xla', 'xpu'
]

TORCH_DEVICE_INDEX = NonNegativeInt