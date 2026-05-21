r"""Thread-safe local storage for registries."""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, ValuesView
from threading import RLock
from typing import Dict, Generic, Hashable, Iterator, MutableMapping, TypeVar

__all__ = ["ThreadSafeLocalStorage"]

KeyType = TypeVar("KeyType", bound=Hashable)
ValType = TypeVar("ValType")


class ThreadSafeLocalStorage(
    MutableMapping[KeyType, ValType], Generic[KeyType, ValType]
):
    """Thread-safe local storage with single-write multi-read semantics."""

    def __init__(self):
        self._storage: Dict[KeyType, ValType] = {}
        self._lock = RLock()

    def __getitem__(self, key: KeyType) -> ValType:
        with self._lock:
            return self._storage[key]

    def __setitem__(self, key: KeyType, value: ValType) -> None:
        with self._lock:
            self._storage[key] = value

    def __delitem__(self, key: KeyType) -> None:
        with self._lock:
            del self._storage[key]

    def __iter__(self) -> Iterator[KeyType]:
        with self._lock:
            return iter(list(self._storage.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._storage)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._storage

    def clear(self) -> None:
        with self._lock:
            self._storage.clear()

    def keys(self) -> KeysView[KeyType]:
        with self._lock:
            return self._storage.keys()

    def values(self) -> ValuesView[ValType]:
        with self._lock:
            return self._storage.values()

    def items(self) -> ItemsView[KeyType, ValType]:
        with self._lock:
            return self._storage.items()
