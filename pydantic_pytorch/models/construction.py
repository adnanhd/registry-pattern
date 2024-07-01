from pydantic._internal._model_construction import ModelMetaclass
from pydantic._internal import _decorators, _generics
from pydantic_core import SchemaSerializer, SchemaValidator, CoreSchema
from typing import Any, ClassVar, Literal, get_type_hints, TypedDict, Generic, TypeVar, Tuple
from inspect import Signature, signature, Parameter, _empty
from ..registry import ClassRegistryMetaclass, FunctionalRegistryMetaclass
from pydantic.fields import ModelPrivateAttr, FieldInfo
from pydantic import BaseModel, ConfigDict, create_model
from torch.nn import Linear, Transformer


def wrap_annotation(annot: type):
    if annot is Parameter.empty:
        return Any
    if annot is None:
        return type(None)
    return annot


def parameter_to_field(param: Parameter, /, **kwargs) -> Tuple[type, FieldInfo]:
    if param.default is not Parameter.empty:
        kwargs['default'] = param.default
    kwargs['required'] = param.default is Parameter.empty
    kwargs['annotation'] = wrap_annotation(param.annotation)
    kwargs['kw_only'] = param.kind is Parameter.KEYWORD_ONLY

    return wrap_annotation(param.annotation), FieldInfo(**kwargs)


def to_typed_dict(tp: Any) -> BaseModel:
    sign = signature(tp)

    kwargs = {name: parameter_to_field(
        param) for name, param in sign.parameters.items()}
    return create_model(tp.__name__ + 'Config', **kwargs)


TransformerConfig = to_typed_dict(Transformer)
LinearConfig = to_typed_dict(Linear)
