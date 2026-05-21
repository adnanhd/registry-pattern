"""Protocols for ``Annotated[T, ...]`` markers consumed by the factory.

The factory's ``registry.schema.process_validate`` and ``process_compute``
helpers duck-type on ``.validate`` / ``.compute`` -- any object with those
methods works. These Protocols are here as a *typing contract*: subclass
them for static type-checker support, or just duck-type. Either path is
fine.

The library ships ZERO concrete markers. Domain-specific markers
(``SameDeviceAs``, ``Checksum``, ...) belong in your project. See
``examples/02_factory_pipeline.py`` for torch-flavoured examples.

Usage::

    from dataclasses import dataclass
    from typing import Any
    from registry.markers import ValidateMarker, ComputeMarker

    @dataclass(frozen=True)
    class MustBe(ValidateMarker):
        expected: Any
        def validate(self, value, kwargs, ctx):
            if value != self.expected:
                raise ValueError(f"expected {self.expected!r}, got {value!r}")

    @dataclass(frozen=True)
    class Doubled(ComputeMarker):
        name: str
        def compute(self, value):
            return value * 2
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["ValidateMarker", "ComputeMarker", "Marker"]


@runtime_checkable
class ValidateMarker(Protocol):
    """A marker that validates an annotated kwarg before the target runs.

    The factory calls ``validate(value, kwargs, ctx)`` for every marker that
    has this method, after kwargs are assembled and before the target is
    invoked. Raising terminates the build with the original exception.
    """

    def validate(
        self,
        value: Any,
        kwargs: dict[str, Any],
        ctx: dict[str, Any],
    ) -> None: ...


@runtime_checkable
class ComputeMarker(Protocol):
    """A marker that derives a meta value from an annotated kwarg.

    The factory calls ``compute(value)`` for every marker that has this
    method, after the target returns. The result is written into the
    envelope's ``meta`` dict under ``self.name``.
    """

    name: str

    def compute(self, value: Any) -> Any: ...


@runtime_checkable
class Marker(ValidateMarker, ComputeMarker, Protocol):
    """A marker that both validates AND derives meta.

    The factory dispatches on method presence (`.validate`, `.compute`), so a
    marker can implement either or both. ``Marker`` is the convenience union.
    """
