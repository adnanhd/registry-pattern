import typing
import abc
import inspect
import weakref


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


class ValidationError(RegistryError):
    """Exception raised for validation errors."""
    pass


class RegistryMeta(abc.ABCMeta):
    """Metaclass for managing registrations."""
    def __init__(cls, name, bases, attrs):
        cls._registrar = dict()
        super().__init__(name, bases, attrs)

    """
    def __call__(cls, *args, **kwargs):
        raise TypeError(f"Cannot instantiate registry class {cls.__name__}")
    """

    def _add_registry(cls, name: str, value: typing.Any) -> typing.Any:
        if name not in cls.list_registry():
            return cls._registrar.setdefault(name, value)
        raise RegistryError(f'{cls.__name__}: {name} registered')

    def _pop_registry(cls, name: str) -> typing.Any:
        if name in cls.list_registry():
            return cls._registrar.pop(name)
        raise RegistryError(f'{cls.__name__}: {name} not registered')

    def _clear_registry(cls) -> None:
        cls._registrar.clear()

    def list_registry(cls) -> typing.List[str]:
        """Return a list of registered class/function names."""
        return list(cls._registrar.keys())

    def has_registered(cls, value: typing.Any) -> bool:
        """Check if a class/function is registered."""
        return value in cls._registrar.values()

    def get_registered(cls, name: str) -> typing.Any:
        """Return a registered class/function by name."""
        if name not in cls._registrar.keys():
            raise RegistryError(f'{cls}: {name} not registered')
        return cls._registrar[name]


class ClassRegistryMetaclass(RegistryMeta):
    """Metaclass for registering classes."""
    def validate_class(cls, subcls: type) -> type:
        """Validate a class."""
        if not inspect.isclass(subcls):
            raise ValidationError(f"{subcls} is not a class")
        if not issubclass(subcls, cls):
            raise ValidationError(f"{subcls} is not of cls {cls}")
        return subcls

    def register_class(cls, subcls) -> type:
        """Register a subclass."""
        # Validate the class
        subcls = cls.validate_class(subcls)

        # Register the class
        return cls._add_registry(subcls.__name__, subcls)

    def register_module_subclasses(cls, module):
        """Register all subclasses of a given module."""
        assert inspect.ismodule(module), f"{module} is not a module"
        for obj in _get_members(module).values():
            try:
                cls.register_class(obj)
            except (RegistryError, KeyError, AssertionError):
                pass
        return module

    def register_subclasses(cls, supercls: type, recursive: bool = False):
        """Register all subclasses of a given superclass."""
        assert inspect.isclass(supercls), f"{supercls} is not a class"
        abc.ABCMeta.register(cls, supercls)
        for subcls in _get_subclasses(supercls):
            if recursive and _get_subclasses(subcls):
                cls.register_subclasses(subcls, recursive)
            cls.register_class(subcls)
        return supercls


class FunctionalRegistryMetaclass(RegistryMeta):
    """Metaclass for registering functions."""
    def __init__(cls, name, bases, attrs):
        def _f(): pass
        abc.ABCMeta.register(cls, type(_f))
        super().__init__(name, bases, attrs)

    def validate_function(cls, func: typing.Callable) -> typing.Callable:
        """Validate a function."""
        if inspect.isfunction(func):
            return func
        raise ValidationError(f"{func} is not a function")

    def register_function(cls, func: typing.Callable) -> typing.Callable:
        """Register a function."""
        # Validate the function
        func = cls.validate_function(func)

        # Register the function
        return cls._add_registry(func.__name__, func)

    def register_module_functions(cls, module):
        """Register all functions in a given module."""
        assert inspect.ismodule(module), f"{module} is not a module"
        for obj in _get_members(module).values():
            try:
                cls.register_function(obj)
            except (RegistryError, KeyError, AssertionError):
                pass
        return module


class InstanceRegistryMetaclass(RegistryMeta):
    """Metaclass for registering instances."""
    def __init__(cls, name, bases, attrs):
        cls._registrar = weakref.WeakSet()
        super().__init__(name, bases, attrs)

    def validate_instance(cls, instance: object) -> object:
        """Validate an instance."""
        if isinstance(instance, cls):
            return instance
        raise ValidationError(f"{instance} is not an instance of {cls}")

    def register_instance(cls, instance: object) -> object:
        """Register an instance."""
        # Validate the instance
        instance = cls.validate_instance(instance)

        # Register the instance
        return cls._add_registry(instance.__class__.__name__, instance)


def _get_members(module) -> typing.Dict[str, typing.Any]:
    """Get members of a module."""
    assert inspect.ismodule(module), f"{module} is not a module"
    if not hasattr(module, '__all__'):
        return dict(inspect.getmembers(module))
    return {name: getattr(module, name) for name in module.__all__}


def _get_subclasses(subcls) -> typing.List[typing.Type[typing.Any]]:
    """Get members of a module."""
    assert inspect.isclass(subcls), f"{subcls} is not a class"
    return subcls.__subclasses__()
