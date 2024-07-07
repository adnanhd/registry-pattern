from typing import Annotated, Literal
from annotated_types import Ge

TORCH_DEVICE_INDEX = Annotated[int, Ge(0)] | None

TORCH_DEVICE_TYPE = Literal['cpu', 'cuda', 'fpga', 'hip', 'hpu', 'ideep', 'ipu',
                            'lazy', 'meta', 'mkldnn', 'mps', 'mtia',  'opencl',
                            'opengl', 'ort', 've', 'vulkan', 'xla', 'xpu']
