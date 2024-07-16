from functools import wraps
from types import FunctionType
from typing import ParamSpec, TypeVar, Generic, Callable, Any

P = ParamSpec("P")
T = TypeVar("T")

class Hookable(Generic[P, T]):
    __slots__ = ('func', 'pre_hook', 'post_hook')
    def __init__(
        self,
        func: Callable[P, T],
        pre_hook: Callable[P, T] | None = None,
        post_hook: Callable[P, T] | None = None
) -> None:
        self.func: Callable[P, T] = func
        self.pre_hook: Callable[P , T] | None = pre_hook
        self.post_hook: Callable[P , T] | None = post_hook

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if self.pre_hook:
            self.pre_hook(*args, **kwargs)
        result = self.func(*args, **kwargs)
        if self.post_hook:
            self.post_hook(*args, **kwargs)
        return result

    def __repr__(self) -> str:
        if isinstance(self.func, Hookable):
            func_repr = repr(self.func)
        else:
            func_repr = (self.func.__name__)
        pre_func_repr = self.pre_hook.__name__ + ' :- ' if self.pre_hook else ''
        post_func_repr = ' -: ' + self.post_hook.__name__ if self.post_hook else ''

        return f'<functree {pre_func_repr}{func_repr}{post_func_repr}>'

class HookMeta(type):
    def __new__(cls, name, bases, dct):
        for attr_name, attr_value in dct.items():
            if callable(attr_value) and attr_name != '__init__':
                @wraps(attr_value)
                def cast(fn: Callable[P ,T]) -> Hookable[P, T]:
                    return Hookable[P, T](fn)
                dct[attr_name] = cast(attr_value)
        return super(HookMeta, cls).__new__(cls, name, bases, dct)

def hookable(func: Callable[P, T]) -> Hookable[P, T]:
    return Hookable[P, T](func)


def before(hook_func : Callable[P, T]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    @wraps(hook_func)
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        return Hookable[P, T](func, pre_hook=hook_func)
    return decorator

def after(hook_func: Callable[P, T]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    @wraps(hook_func)
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        return Hookable[P, T](func, post_hook=hook_func)
    return decorator
