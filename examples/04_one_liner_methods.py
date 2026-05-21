#!/usr/bin/env python
"""One-liner ``from_X`` / ``to_X`` methods bound to factory primitives.

Two distinct paths, both expressible in one line per method:

  - **envelope round-trip** (``{type, data, meta}``-shaped): yaml / json text.
    Decode -> envelope-mode ``build(envelope)``. Serializes WITH meta.

  - **flat kwargs**: argparse.Namespace or a plain dict of constructor args.
    Class-mode ``build(cls, kwargs, validator=...)``. No meta path.

These two stay separate on purpose -- the library does no auto-detection
between them.
"""

from __future__ import annotations

import json as json_lib
from argparse import ArgumentParser, Namespace
from typing import Any

import yaml as yaml_lib

from registry import TypeRegistry, build, serialize


class NetworkRegistry(TypeRegistry[Any]):
    pass


@NetworkRegistry.register_artifact
class MyModel:
    def __init__(self, in_channels: int = 3, hidden: int = 64) -> None:
        self.in_channels: int = in_channels
        self.hidden: int = hidden

    # ---- envelope-shaped round-trip (carries meta) ----

    @classmethod
    def from_yaml(cls, text: str) -> "MyModel":
        return build(yaml_lib.safe_load(text))

    @classmethod
    def from_json(cls, text: str) -> "MyModel":
        return build(json_lib.loads(text))

    def to_yaml(self) -> str:
        return serialize(self, serializator="yaml")

    def to_json(self) -> str:
        return serialize(self, serializator="json")

    # ---- flat kwargs (no envelope; for argparse / plain dicts) ----

    @classmethod
    def from_args(cls, args: Namespace) -> "MyModel":
        return build(cls, args, validator="argparse")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "MyModel":
        return build(cls, kwargs, validator="python")

    def to_config(self) -> Namespace:
        return Namespace(**serialize(self, serializator="python")["data"])


def main() -> None:
    # 1. Flat kwargs (argparse)
    parser = ArgumentParser()
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args(["--in-channels", "8", "--hidden", "16"])
    m1 = MyModel.from_args(args)
    print("from_args:", m1.in_channels, m1.hidden)

    # 2. Flat kwargs (direct)
    m2 = MyModel.from_kwargs(in_channels=4, hidden=32)
    print("from_kwargs:", m2.in_channels, m2.hidden)

    # 3. Envelope round-trip (yaml carries meta if any)
    src = MyModel(in_channels=1, hidden=128)
    yaml_text = src.to_yaml()
    print("to_yaml :", repr(yaml_text))
    m3 = MyModel.from_yaml(yaml_text)
    assert (src.in_channels, src.hidden) == (m3.in_channels, m3.hidden)
    print("yaml round-trip OK")

    # 4. JSON envelope round-trip
    json_text = src.to_json()
    print("to_json :", json_text)
    m4 = MyModel.from_json(json_text)
    assert (src.in_channels, src.hidden) == (m4.in_channels, m4.hidden)
    print("json round-trip OK")


if __name__ == "__main__":
    main()
