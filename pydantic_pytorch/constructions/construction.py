from typing import Callable, Any
from inspect import signature, Parameter
from pydantic.fields import FieldInfo
from pydantic import BaseModel, create_model
from typing_extensions import TypedDict

def wrap_annotation(annot: type):
    if annot is Parameter.empty:
        return Any
    if annot is None:
        return type(None)
    return annot


def parameter_to_field(param: Parameter, /, **kwargs) -> tuple[type, FieldInfo]:
    if param.default is not Parameter.empty:
        kwargs['default'] = param.default
    kwargs['required'] = param.default is Parameter.empty
    kwargs['annotation'] = wrap_annotation(param.annotation)
    kwargs['kw_only'] = param.kind is Parameter.KEYWORD_ONLY

    return wrap_annotation(param.annotation), FieldInfo(**kwargs)


def to_typed_dict(tp: Callable) -> BaseModel:
    assert callable(tp), 'Expected a callable type'
    kwargs = {name: parameter_to_field(param) 
              for name, param in signature(tp).parameters.items()
              if param.kind not in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)}
    return create_model(tp.__name__ + 'Config', **kwargs)
