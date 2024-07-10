import weakref
from typing import Generic, TypeVar, get_args, MutableMapping, ClassVar, Protocol, Any
from functools import wraps, lru_cache
from .registry import RegistryError, ValidationError, TypeCheckError
from typeguard import check_type, TypeCheckError
from abc import ABCMeta
class Hashable(Protocol):
    def __hash__(self) -> int:
        ...


class Stringable(Protocol):
    def __str__(self) -> str:
        ...


class TrackableMetaclass(type):
    def __init__(cls, name, bases, attrs):
        cls._trackables = weakref.WeakSet()
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        cls._trackables.add(instance)
        return instance

    def get_instances(cls) -> set[object]:
        return weakref.WeakSet(cls._trackables)


class TrackerMeta(ABCMeta):
    """Metaclass for managing registrations."""
    _trackables: ClassVar[MutableMapping]

    def __call__(cls, *args, **kwargs):
        raise TypeError(f"Cannot instantiate registry class {cls.__name__}")

    def _add_registry(cls, key: Hashable, value: Any) -> Any:
        if key not in cls.list_registry():
            return cls._trackables.setdefault(key, value)
        raise RegistryError(f'{cls.__name__}: {key!r} is already registered')

    def _pop_registry(cls, key: Hashable) -> Any:
        if key in cls.list_registry():
            return cls._trackables.pop(key)
        raise RegistryError(f'{cls.__name__}: {key!r} is not registered')

    def _clear_registry(cls) -> None:
        cls._trackables.clear()

    def list_registry(cls) -> list[Hashable]:
        """Return a list of registered class/function names."""
        return list(cls._trackables.keys())

    def has_registered(cls, value: Any) -> bool:
        """Check if a class/function is registered."""
        return value in cls._trackables.values()

    @lru_cache(maxsize=64, typed=True)
    def get_registered(cls, key: Any) -> Any:
        """Return a registered class/function by name."""
        if key not in cls._trackables.keys():
            raise RegistryError(f'{cls}: {key} not registered')
        return cls._trackables[key]


K = TypeVar("K", bound=Hashable)
CfgT = dict[str, Any]

class InstanceKeyMeta(Generic[K], metaclass=TrackerMeta):
    __orig_bases__: ClassVar[tuple[type, ...]]
    __check_type__: ClassVar[type[K]]
    _trackables: ClassVar[MutableMapping[K, CfgT]]
    
    @classmethod
    def __init_subclass__(cls, weak: bool = True, static: bool = True) -> None:
        cls._trackables = weakref.WeakKeyDictionary[K]() if weak else dict[K, CfgT]()
        cls.__check_type__, = get_args(cls.__orig_bases__[0]) if static else (type,)

    @classmethod
    def validate_instance(cls, instance: K) -> K:
        """Validate an instance."""
        if not isinstance(instance, cls):
            raise ValidationError(f"{instance} is not an instance of {cls}")
        try:
            return check_type(instance, cls.__check_type__)
        except TypeCheckError:
            raise ValidationError(
                f"{instance} is not of type {cls.__check_type__}")

    @classmethod
    def register_instance(cls, instance: K, config: CfgT) -> K:
        """Register an instance."""
        # Validate the instance
        instance = cls.validate_instance(instance)

        # Register the instance
        return cls._add_registry(instance, config)


V = TypeVar("V", bound=Stringable)


class InstanceValueMeta(Generic[V], metaclass=TrackerMeta):
    __orig_bases__: ClassVar[tuple[type, ...]]
    __check_type__: ClassVar[type[V]]
    _trackables: ClassVar[MutableMapping[Hashable, V]]

    @classmethod
    def __init_subclass__(cls, weak: bool = True, static: bool = True) -> None:
        cls._trackables = weakref.WeakValueDictionary[V]() if weak else dict[Hashable, V]()
        cls.__check_type__, = get_args(cls.__orig_bases__[0]) if static else (type,)

    @classmethod
    def validate_instance(cls, instance: V) -> V:
        """Validate an instance."""
        if not isinstance(instance, cls):
            raise ValidationError(f"{instance} is not an instance of {cls}")
        try:
            return check_type(instance, cls.__check_type__)
        except TypeCheckError:
            raise ValidationError(
                f"{instance} is not of type {cls.__check_type__}")

    @classmethod
    def register_instance(cls, instance: V) -> V:
        """Register an instance."""
        # Validate the instance
        instance = cls.validate_instance(instance)

        # Register the instance
        return cls._add_registry(str(instance), instance)

    @classmethod
    def track_instances(cls, subcls: V, validate_class: bool = True) -> V:
        if validate_class:
            check_type(subcls, cls.__check_type__)

        cls.register(subcls)

        @wraps(subcls.__init__)
        def __init__(self, *args, **kwargs):
            super(subcls, self).__init__(*args, **kwargs)
            cls.register_instance(self)

        return type(subcls.__qualname__, (subcls,), {**subcls.__dict__, "__init__": __init__})
