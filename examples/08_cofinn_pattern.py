#!/usr/bin/env python
"""Cofinn-style one-liner methods bound to factory primitives.

Existing cofinn classes have ``add_args`` / ``from_args`` / ``to_config``
methods with duplicated implementations. The library exposes ``build()`` and
``serialize()`` as generic primitives so each method body collapses to a
single line that delegates to the pipeline.

Pipeline (validators, meters, reporters, post hooks, meta) fires through both
``build()`` and ``serialize()``.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from registry import FunctionalRegistry, TypeRegistry, build, serialize


class NetworkRegistry(TypeRegistry[object]):
    pass


@NetworkRegistry.register_artifact
class CoFINN:
    def __init__(self, in_channels: int = 3, hidden: int = 64) -> None:
        self.in_channels: int = in_channels
        self.hidden: int = hidden

    # ---- inbound: one-liners delegating to build() with a medium name ----

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoFINN":
        return build(cls, data, validator="python")

    @classmethod
    def from_yaml(cls, text: str) -> "CoFINN":
        return build(cls, text, validator="yaml")

    @classmethod
    def from_json(cls, text: str) -> "CoFINN":
        return build(cls, text, validator="json")

    @classmethod
    def from_args(cls, args: Namespace) -> "CoFINN":
        return build(cls, args, validator="argparse")

    # ---- outbound: one-liners delegating to serialize() ----

    def to_dict(self) -> dict[str, Any]:
        return serialize(self, serializator="python")

    def to_yaml(self) -> str:
        return serialize(self, serializator="yaml")

    def to_json(self) -> str:
        return serialize(self, serializator="json")

    def to_config(self) -> Namespace:
        return Namespace(**serialize(self, serializator="python"))


def main() -> None:
    # 1) From a YAML string
    m1 = CoFINN.from_yaml("in_channels: 1\nhidden: 128\n")
    print("from_yaml:", m1.in_channels, m1.hidden)

    # 2) From a Python dict
    m2 = CoFINN.from_dict({"in_channels": 4, "hidden": 32})
    print("from_dict:", m2.in_channels, m2.hidden)

    # 3) From argparse
    parser = ArgumentParser()
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args(["--in-channels", "8", "--hidden", "16"])
    m3 = CoFINN.from_args(args)
    print("from_args:", m3.in_channels, m3.hidden)

    # 4) Round-trip
    yaml_text = m1.to_yaml()
    m4 = CoFINN.from_yaml(yaml_text)
    assert (m1.in_channels, m1.hidden) == (m4.in_channels, m4.hidden)
    print("yaml round-trip OK")
    print("to_dict :", m1.to_dict())
    print("to_yaml :", repr(m1.to_yaml()))
    print("to_json :", m1.to_json())


if __name__ == "__main__":
    main()
