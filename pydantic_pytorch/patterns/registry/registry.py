import abc
import inspect
from functools import lru_cache
from typeguard import check_type, TypeCheckError as TypeguardTypeCheckError
from typing import TypeVar, Generic, ClassVar, Dict, Callable, ParamSpec, get_args, List, Any, Type


__all__ = [
    'RegistryError',
    'ValidationError',
    'RegistryMeta',
    'ClassRegistryMetaclass',
    'FunctionalRegistryMetaclass',
    'InstanceRegistryMetaclass'
]


class RegistryError(ValueError):
    """Exception raised for registration errors."""
    pass


class ValidationError(RuntimeError):
    """Exception raised for validation errors."""
    pass


class TypeCheckError(TypeError):
    """Exception raised for type checking errors."""
    pass


Error = RegistryError | ValidationError | TypeCheckError


class RegistryMeta(abc.ABCMeta):
    """Metaclass for managing registrations."""
    _registrar: ClassVar[Dict[str, Any]]

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        cls._registrar = dict()
        return cls

    def __call__(cls, *args, **kwargs):
        raise TypeError(f"Cannot instantiate registry class {cls.__name__}")

    def _add_registry(cls, name: str, value: Any) -> Any:
        if name not in cls.list_registry():
            return cls._registrar.setdefault(name, value)
        raise RegistryError(f'{cls.__name__}: {name!r} is already registered')

    def _pop_registry(cls, name: str) -> Any:
        if name in cls.list_registry():
            return cls._registrar.pop(name)
        raise RegistryError(f'{cls.__name__}: {name!r} is not registered')

    def _clear_registry(cls) -> None:
        cls._registrar.clear()

    def list_registry(cls) -> List[str]:
        """Return a list of registered class/function names."""
        return list(cls._registrar.keys())

    def has_registered(cls, value: Any) -> bool:
        """Check if a class/function is registered."""
        return value in cls._registrar.values()

    @lru_cache(maxsize=16, typed=False)
    def get_registered(cls, name: str) -> Any:
        """Return a registered class/function by name."""
        if name not in cls._registrar.keys():
            raise RegistryError(f'{cls}: {name} not registered')
        return cls._registrar[name]


def _get_members(module) -> Dict[str, Any]:
    """Get members of a module."""
    assert inspect.ismodule(module), f"{module} is not a module"
    if not hasattr(module, '__all__'):
        return dict(inspect.getmembers(module))
    return {name: getattr(module, name) for name in module.__all__}


def _get_subclasses(subcls) -> List[Type[Any]]:
    """Get members of a module."""
    assert inspect.isclass(subcls), f"{subcls} is not a class"
    return subcls.__subclasses__()


Protocol = TypeVar("Protocol")


class ClassRegistry(Generic[Protocol], metaclass=RegistryMeta):
    """Metaclass for registering classes."""
    __orig_bases__: ClassVar[tuple[type, ...]]
    __check_type__: ClassVar[Protocol]
    _registrar: ClassVar[Dict[str, type[Protocol]]]

    @classmethod
    def __init_subclass__(cls, strict: bool = True):
        if strict:
            cls.__check_type__, = get_args(cls.__orig_bases__[0])
        else:
            cls.__check_type__ = type

    @classmethod
    def validate_class(cls, subcls: type[Protocol]) -> type[Protocol]:
        """Validate a class."""
        if not inspect.isclass(subcls):
            raise ValidationError(f"{subcls} is not a class")
        if not issubclass(subcls, cls):
            raise ValidationError(f"{subcls} is not of cls {cls}")
        try:
            return check_type(subcls, cls.__check_type__)
        except TypeguardTypeCheckError as exc:
            raise ValidationError(
                f"{subcls} is not of type {cls.__check_type__}") from exc

    @classmethod
    def register_class(cls, subcls: type[Protocol]) -> type[Protocol]:
        """Register a subclass."""
        # Validate the class
        subcls = cls.validate_class(subcls)

        # Register the class
        return cls._add_registry(subcls.__name__, subcls)

    @classmethod
    def register_module_subclasses(cls, module):
        """Register all subclasses of a given module."""
        assert inspect.ismodule(module), f"{module} is not a module"
        for obj in _get_members(module).values():
            try:
                cls.register_class(obj)
            except (Error, KeyError, AssertionError):
                pass
        return module

    @classmethod
    def register_subclasses(cls, supercls: type[Protocol], recursive: bool = False) -> type[Protocol]:
        """Register all subclasses of a given superclass."""
        assert inspect.isclass(supercls), f"{supercls} is not a class"
        abc.ABCMeta.register(cls, supercls)
        for subcls in _get_subclasses(supercls):
            if recursive and _get_subclasses(subcls):
                cls.register_subclasses(subcls, recursive)
            cls.register_class(subcls)
        return supercls


R = TypeVar("R")
P = ParamSpec("P")


class FunctionalRegistry(Generic[P, R], metaclass=RegistryMeta):
    """Metaclass for registering functions."""
    __orig_bases__: ClassVar[tuple[type, ...]]
    __check_type__: ClassVar[Callable[P, R]]
    _registrar: ClassVar[dict[str, Callable[P, R]]]

    @classmethod
    def __init_subclass__(cls, static: bool = True):
        def _f(): pass
        abc.ABCMeta.register(cls, type(_f))
        if static:
            param, ret = get_args(cls.__orig_bases__[0])
            cls.__check_type__ = Callable[param, ret]  # type: ignore
        else:
            cls.__check_type__ = Callable

    @classmethod
    def validate_function(cls, func: Callable[P, R]) -> Callable[P, R]:
        """Validate a function."""
        if not inspect.isfunction(func):
            raise ValidationError(f"{func} is not a function")
        try:
            return check_type(func, cls.__check_type__)
        except TypeguardTypeCheckError as exc:
            raise ValidationError(
                f"{func} is not of type {cls.__check_type__}") from exc

    @classmethod
    def register_function(cls, func: Callable[P, R]) -> Callable[P, R]:
        """Register a function."""
        # Validate the function
        func = cls.validate_function(func)

        # Register the function
        return cls._add_registry(func.__name__, func)

    @classmethod
    def register_module_functions(cls, module):
        """Register all functions in a given module."""
        assert inspect.ismodule(module), f"{module} is not a module"
        for obj in _get_members(module).values():
            try:
                cls.register_function(obj)
            except (Error, KeyError, AssertionError):
                pass
        return module
