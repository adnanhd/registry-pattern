import sys
from types import new_class
from typing import TYPE_CHECKING, Any, Callable


if sys.version_info >= (3, 10):
    from types import EllipsisType
    from typing import ParamSpec

    UnionOrNone = Union[None, str]
    from typing import (
        Union,
        TypeGuard,
        Concatenate,
        ParamSpec,
        Optional,
        TypeVar,
        ClassVar,
        Generic,
        Annotated,
    )
else:
    from typing_extensions import ParamSpec

    EllipsisType = type(Ellipsis)
    from typing_extensions import TypeGuard, Concatenate, ParamSpec, Annotated
    from typing import Union, Optional, TypeVar, ClassVar, Generic

    UnionOrNone = Union[str, None]

if sys.version_info >= (3, 9):
    List = list
    Set = set
    Dict = dict
    Tuple = tuple
    Type = type
else:
    from typing import List, Set, Dict, Tuple, Type

if sys.version_info >= (3, 8):
    from typing import Literal, Protocol, TypedDict, Final
else:
    from typing_extensions import Literal, Protocol, TypedDict, Final

if sys.version_info >= (3, 7):
    from typing import ForwardRef
else:
    from typing_extensions import ForwardRef

# Conditional logic for Python version-specific functionality
if TYPE_CHECKING:
    # Add any type-checking only imports or logic here
    from typing import (
        List,
        Dict,
        Set,
        Tuple,
        Union,
        Type,
        Optional,
        TypeVar,
        ClassVar,
        Generic,
    )
