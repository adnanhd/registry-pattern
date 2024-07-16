"""
This module contains the composite class, which is a wrapper for composite functions.
"""

from typing import Callable, Any, ParamSpec, TypeVar
from functools import reduce, wraps


def _none_wrapper(func: Callable[..., Any]):
    return lambda x: func(x) if x is not None else None


class Composable(object):
    """Wrapper for composite functions"""

    def __init__(self, *funcs: Callable[..., Any], ignore_none: bool = False):
        assert all(map(callable, funcs)), "All functions must be callable"
        self._funcs = reversed(funcs)
        self._ignore_none = ignore_none

    def push_proloque(self, func: Callable[[Any], Any]):
        """Add a function to the proloque"""
        assert callable(func), "Callback must be callable"
        if self._ignore_none:
            func = _none_wrapper(func)
        self._funcs = (*self._funcs, func)
        return self
    
    def push_epiloque(self, func: Callable[[Any], Any]):
        """Add a function to the epilogue"""
        assert callable(func), "Callback must be callable"
        if self._ignore_none:
            func = _none_wrapper(func)
        self._funcs = (func, *self._funcs)
        return self

    def proloque_callback(self, func: Callable[[Any], Any]):
        """Add a function to the proloque"""
        assert callable(func), "Callback must be callable"
        if self._ignore_none:
            func = _none_wrapper(func)
        self._funcs = (*self._funcs, func)
        return func

    def epiloque_callback(self, func: Callable[[Any], Any]):
        """Add a function to the epilogue"""
        assert callable(func), "Callback must be callable"
        if self._ignore_none:
            func = _none_wrapper(func)
        self._funcs = (func, *self._funcs)
        return func

    def __call__(self, *args: Any, **kwargs: Any):
        return self.composed_function(*args, **kwargs)

    @property
    def composed_function(self) -> Callable[..., Any]:
        """Return the composed function"""
        return compose(*self._funcs)


def compositefunction(func: Callable[..., Any]) -> Composable:
    """Decorator for composite functions"""
    return Composable(func)


def compositemethod(func: Callable[..., Any]) -> Composable:
    """Decorator for composite methods"""
    return Composable(func)


P = ParamSpec("P")
MidValType = TypeVar("MidValType")
ResultType = TypeVar("ResultType")


def compose_two_funcs(
        f: Callable[P, MidValType],
        g: Callable[[MidValType], ResultType]
) -> Callable[P, ResultType]:
    """Compose two functions"""
    assert callable(f), "First function must be callable"
    assert callable(g), "Second function must be callable"
    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> ResultType:
        return g(f(*args, **kwargs))
    return wrapper


def compose(*functions: Callable[..., Any]) -> Callable[..., Any]:
    """Compose functions"""
    return reduce(compose_two_funcs, reversed(functions), lambda x: x)


def on_proloque(
        func: Callable[P, MidValType]
) -> Callable[[Callable[[MidValType], ResultType]], Callable[P, ResultType]]:
    """Decorator for post-processing functions"""
    def wrapper(fn: Callable[[MidValType], ResultType]) -> Callable[P, ResultType]:
        return compose_two_funcs(func, fn)
    return wrapper


def on_epiloque(
        func: Callable[[MidValType], ResultType]
) -> Callable[[Callable[P, MidValType]], Callable[P, ResultType]]:
    """Decorator for post-processing functions"""
    def wrapper(fn: Callable[P, MidValType]) -> Callable[P, ResultType]:
        return compose_two_funcs(fn, func)
    return wrapper
