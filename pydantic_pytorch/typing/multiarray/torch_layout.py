import sys
from pydantic import BaseModel


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal



__ALL__ = ['TorchLayout']

class TorchLayout(BaseModel):
    """TypedDict for torch.device"""

    type: Literal['torch.strided', 'torch.sparse_coo', 'torch._mkldnn']
    