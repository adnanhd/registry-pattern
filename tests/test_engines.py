"""Coverage for registry.engines.ConfigFileEngine loaders.

Round-trips a small config through each registered loader (json, yaml, yml,
toml, xml), driving them via ``ConfigFileEngine.get_artifact(ext)`` the same
way the CLI resolves them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from registry.engines import ConfigFileEngine


def _load(ext: str, path: Path) -> Dict[str, Any]:
    return ConfigFileEngine.get_artifact(ext)(path)


def test_engine_registers_expected_extensions() -> None:
    known = set(ConfigFileEngine.iter_identifiers())
    assert {"json", "yaml", "yml", "toml", "xml"} <= known


def test_json_loader(tmp_path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text('{"name": "widget", "count": 3, "nested": {"ok": true}}')
    cfg = _load("json", p)
    assert cfg == {"name": "widget", "count": 3, "nested": {"ok": True}}


def test_yaml_loader(tmp_path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("name: widget\ncount: 3\nitems:\n  - a\n  - b\n")
    cfg = _load("yaml", p)
    assert cfg == {"name": "widget", "count": 3, "items": ["a", "b"]}


def test_yml_loader_is_yaml_alias(tmp_path) -> None:
    p = tmp_path / "cfg.yml"
    p.write_text("flag: true\nvalue: 7\n")
    cfg = _load("yml", p)
    assert cfg == {"flag": True, "value": 7}


def test_toml_loader(tmp_path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text('title = "demo"\n[section]\nkey = "value"\nnum = 42\n')
    cfg = _load("toml", p)
    assert cfg == {"title": "demo", "section": {"key": "value", "num": 42}}


def test_xml_loader_attributes_children_and_text(tmp_path) -> None:
    p = tmp_path / "cfg.xml"
    p.write_text(
        '<root version="1">'
        "<single>hello</single>"
        "<multi>a</multi>"
        "<multi>b</multi>"
        '<parent kind="k">lead<child>c</child></parent>'
        "<empty></empty>"
        "</root>"
    )
    cfg = _load("xml", p)
    root = cfg["root"]

    # Attribute becomes an @-prefixed key.
    assert root["@version"] == "1"

    # A text-only element collapses to its string value.
    assert root["single"] == "hello"

    # Repeated tags collapse into a list.
    assert root["multi"] == ["a", "b"]

    # Mixed content: attribute, child, and text captured together.
    parent = root["parent"]
    assert parent["@kind"] == "k"
    assert parent["child"] == "c"
    assert parent["_text"] == "lead"

    # Empty element yields an empty mapping.
    assert root["empty"] == {}
