from typing import Any, Generic, Callable, Type, TypeVar, final
from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler
from abc import abstractmethod
from functools import partial


def configures(*args, **kwargs):
    def _configures(cls):
        return cls
    return _configures


T = TypeVar('T')


class _BaseModel(BaseModel, Generic[T]):

    
    @final
    def _config(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)

    @final
    def _build(self) -> T | Callable[..., T]:
        """Build a torch.device instance from the model."""
        return partial(self._builder(), **self._config())

    @abstractmethod
    def _builder(self) -> Type[T]:
        ...


    @abstractmethod
    def _validate(self, v: T) -> T:
        ...

    @staticmethod
    @abstractmethod
    def _serialize(v: T, nxt: SerializerFunctionWrapHandler) -> dict[str, Any]:
        ...
