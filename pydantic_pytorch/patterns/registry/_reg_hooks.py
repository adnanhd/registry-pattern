"""
This module contains the composite class, which is a wrapper for composite functions.
"""

from typing import Callable, Any
from functools import wraps, reduce


def _none_wrapper(func: Callable[..., Any]):
    return lambda x: func(x) if x is not None else None


class Composable(object):
    """Wrapper for composite functions"""

    def __init__(self, *funcs: Callable[..., Any], ignore_none: bool = False):
        assert all(map(callable, funcs)), "All functions must be callable"
        self._funcs = funcs
        self._ignore_none = ignore_none

    def proloque_callback(self, func: Callable[[Any], Any]):
        """Add a function to the proloque"""
        assert callable(func), "Callback must be callable"
        if self._ignore_none:
            func = _none_wrapper(func)
        self._funcs = (func, *self._funcs)
        return func

    def epiloque_callback(self, func: Callable[[Any], Any]):
        """Add a function to the epilogue"""
        assert callable(func), "Callback must be callable"
        if self._ignore_none:
            func = _none_wrapper(func)
        self._funcs = (*self._funcs, func)
        return func

    def __call__(self, *args: Any, **kwargs: Any):
        return self.composed_function(*args, **kwargs)

    @property
    def composed_function(self) -> Callable[..., Any]:
        """Return the composed function"""
        return reduce(lambda f, g: lambda x: f(g(x)), self._funcs, lambda x: x)


def compose(*functions: Callable[..., Any]) -> Callable[..., Any]:
    """Compose functions"""
    return reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)


def compositefunction(func: Callable[..., Any]) -> Composable:
    """Decorator for composite functions"""
    return Composable(func)


def compositemethod(func: Callable[..., Any]) -> Composable:
    """Decorator for composite methods"""
    return Composable(func)


def on_proloque(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for post-processing functions"""
    def wrapper(fn: Callable[..., Any]):
        return compose(func, fn)
    return wrapper


def on_epiloque(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for pre-processing functions"""
    def wrapper(fn: Callable[..., Any]):
        return compose(fn, func)
    return wrapper
