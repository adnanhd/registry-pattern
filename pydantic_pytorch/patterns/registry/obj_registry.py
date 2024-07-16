"""Object registry pattern."""
from abc import ABC
import weakref
from functools import partial
from typing import TypeVar, ClassVar, Hashable, MutableMapping, Generic, Any, cast
from types import new_class
from .base import BaseMutableRegistry
from ._validator import validate_instance, validate_class_structure
from ._dev_utils import compose, get_protocol
from .obj_extra import ClassTracker, Wrapped


K = TypeVar("K")
V = TypeVar("V")


class _BaseInstanceRegistry(BaseMutableRegistry[K, V], ABC, Generic[K, V]):
    """Base class for registering instances."""
    ignore_structural_subtyping: ClassVar[bool] = False
    ignore_abcnominal_subtyping: ClassVar[bool] = False
    __slots__ = ()


class InstanceRegistry(_BaseInstanceRegistry[Hashable, V]):
    """Functional registry for registering instances."""
    _repository: ClassVar[MutableMapping]  # [Hashable, V]

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        cls._repository = weakref.WeakValueDictionary[Hashable, V]()
        validators = []

        if not cls.ignore_abcnominal_subtyping:
            @validators.append
            def _val_class_hierarchy(value: Any) -> V:
                return validate_instance(value, expected_type=cls)

        if not cls.ignore_structural_subtyping:
            validators.append(
                partial(validate_class_structure,
                        expected_type=get_protocol(cls))
            )

        cls._validate_item = compose(*validators)

    def register_instance(self, instance: V) -> V:
        """Register a subclass."""
        self.add_registry_item(str(instance), instance)
        return instance

    def unregister_instance(self, instance: V) -> V:
        """Unregister a subclass."""
        self.del_registry_item(str(instance))
        return instance

    def track_class_instances(self, cls: type) -> type:
        """Track a class."""
        # TODO: validate if cls is of type V
        kwds = cast(dict[str, Any], ClassTracker[cls].__dict__)
        # validator = compose(self.register_instance, ClassTracker[cls].__call__)

        def validator(cls: ClassTracker[cls], *args, **kwargs):
            print("bok")
            return self.register_instance(super(cls.__class__, cls).__call__(*args, **kwargs))
        kwds['__call__'] = validator
        newmcs = type(ClassTracker.__name__, (ClassTracker,), {'__call__': validator})
        newcls = newmcs(cls.__name__, cls.__bases__, {})
        print(self.__class__, 'registering', newcls)
        return self.__class__.register(newcls)


'''
class InstanceKeyRegistry(_BaseInstanceRegistry[V, dict[str, Any]], Generic[V]):
    """Functional registry for registering instances."""
    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        cls._repository = weakref.WeakKeyDictionary[V]()
        validators = [partial(validate_instance, expected_type=cls)]

        if not cls.ignore_abcnominal_subtyping:
            validators.append(
                partial(validate_class_hierarchy, abc_class=cls)
            )

        if not cls.ignore_structural_subtyping:
            validators.append(
                partial(validate_class_structure,
                        expected_type=get_protocol(cls))
            )

        cls.get_lookup_key = compose(*validators)

    @classmethod
    def _validate_item(cls, value: dict) -> V:
        if not isinstance(value, dict):
            raise TypeError(f'{value} is not a dict')
        return value

    def register_instance(self, cls: V, cfg: dict[str, Any]) -> V:
        """Track a class."""
        self.add_registry_item(cls, Wrapped(cls, cfg))
        return cls

    def unregister_instance(self, cls: V) -> V:
        """Track a class."""
        self.del_registry_item(cls)
        return cls

'''
