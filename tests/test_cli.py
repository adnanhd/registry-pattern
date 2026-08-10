"""CLI coverage: ``python -m registry`` info / build / run + config loading.

Exercises ``registry.__main__`` end to end -- argument dispatch, the three
command handlers across their success and error branches, and the config-file
loader -- by driving ``main()`` with a patched ``sys.argv`` and json fixtures.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from registry import TypeRegistry
from registry import __main__ as cli


class Doodad:
    def __init__(self, n: int = 1) -> None:
        self.n = n

    def run(self) -> str:
        return "ran-%d" % self.n


class Sprocket:
    def __init__(self, tag: str = "x") -> None:
        self.tag = tag

    def __call__(self) -> str:
        return "called-%s" % self.tag


class Blank:
    def __init__(self, x: int = 0) -> None:
        self.x = x


class DoodadRegistry(TypeRegistry[Doodad], repo="cli_doodad"):
    pass


class SprocketRegistry(TypeRegistry[Sprocket], repo="cli_sprocket"):
    pass


class BlankRegistry(TypeRegistry[Blank], repo="cli_blank"):
    pass


DoodadRegistry.register_artifact(Doodad)
SprocketRegistry.register_artifact(Sprocket)
BlankRegistry.register_artifact(Blank)


def _write_json(tmp_path: Path, name: str, data: Dict[str, Any]) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def _run(monkeypatch, argv: List[str]) -> int:
    monkeypatch.setattr("sys.argv", ["python -m registry", *argv])
    return cli.main()


# -- dispatch / info ---------------------------------------------------------


def test_version_flag_exits_zero(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog", "--version"])
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 0


def test_no_command_prints_help(monkeypatch, capsys) -> None:
    assert _run(monkeypatch, []) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_info_command(monkeypatch, capsys) -> None:
    assert _run(monkeypatch, ["info"]) == 0
    assert capsys.readouterr().out  # printed something


# -- load_config_file --------------------------------------------------------


def test_load_config_no_extension() -> None:
    with pytest.raises(ValueError):
        cli.load_config_file(Path("noext"))


def test_load_config_unsupported_extension(tmp_path) -> None:
    p = tmp_path / "cfg.unknownext"
    p.write_text("x")
    with pytest.raises(ValueError):
        cli.load_config_file(p)


def test_load_config_json(tmp_path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"type": "Doodad", "repo": "cli_doodad", "data": {"n": 2}}))
    cfg = cli.load_config_file(p)
    assert cfg["type"] == "Doodad"


# -- build -------------------------------------------------------------------


def test_build_file_not_found(monkeypatch, capsys) -> None:
    assert _run(monkeypatch, ["build", "/no/such/file.json"]) == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_build_single_buildcfg(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(tmp_path, "c.json", {"type": "Doodad", "repo": "cli_doodad", "data": {"n": 3}})
    assert _run(monkeypatch, ["build", cfg]) == 0
    assert "Successfully built 1" in capsys.readouterr().out


def test_build_dry_run(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(tmp_path, "c.json", {"type": "Doodad", "repo": "cli_doodad", "data": {"n": 3}})
    assert _run(monkeypatch, ["build", cfg, "--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out.lower()


def test_build_verbose(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(tmp_path, "c.json", {"type": "Doodad", "repo": "cli_doodad", "data": {"n": 3}})
    assert _run(monkeypatch, ["build", cfg, "--verbose"]) == 0
    assert "Loaded config" in capsys.readouterr().out


def test_build_dict_of_buildcfgs(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(
        tmp_path,
        "c.json",
        {
            "a": {"type": "Doodad", "repo": "cli_doodad", "data": {"n": 1}},
            "b": {"type": "Sprocket", "repo": "cli_sprocket", "data": {"tag": "z"}},
        },
    )
    assert _run(monkeypatch, ["build", cfg]) == 0
    assert "Successfully built 2" in capsys.readouterr().out


def test_build_no_buildcfg_entries(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(tmp_path, "c.json", {"foo": 1, "bar": 2})
    assert _run(monkeypatch, ["build", cfg]) == 1
    assert "No BuildCfg" in capsys.readouterr().err


def test_build_output_file(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(tmp_path, "c.json", {"type": "Doodad", "repo": "cli_doodad", "data": {"n": 7}})
    out = tmp_path / "out.json"
    assert _run(monkeypatch, ["build", cfg, "-o", str(out)]) == 0
    assert out.exists()
    assert "root" in json.loads(out.read_text())


def test_build_error_bad_type(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(tmp_path, "c.json", {"type": "NoSuchType", "repo": "cli_doodad", "data": {}})
    assert _run(monkeypatch, ["build", cfg]) == 1
    assert "Error building" in capsys.readouterr().err


# -- run ---------------------------------------------------------------------


def test_run_file_not_found(monkeypatch, capsys) -> None:
    assert _run(monkeypatch, ["run", "/no/file.json"]) == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_run_callable_entry(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(tmp_path, "c.json", {"type": "Sprocket", "repo": "cli_sprocket", "data": {"tag": "q"}})
    assert _run(monkeypatch, ["run", cfg]) == 0
    assert "called-q" in capsys.readouterr().out


def test_run_method_entry(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(
        tmp_path, "c.json", {"main": {"type": "Doodad", "repo": "cli_doodad", "data": {"n": 4}}}
    )
    assert _run(monkeypatch, ["run", cfg]) == 0
    assert "ran-4" in capsys.readouterr().out


def test_run_entry_not_found(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(
        tmp_path, "c.json", {"other": {"type": "Doodad", "repo": "cli_doodad", "data": {"n": 1}}}
    )
    assert _run(monkeypatch, ["run", cfg, "--entry", "missing"]) == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_run_not_callable_no_run(monkeypatch, capsys, tmp_path) -> None:
    cfg = _write_json(tmp_path, "c.json", {"type": "Blank", "repo": "cli_blank", "data": {"x": 1}})
    assert _run(monkeypatch, ["run", cfg]) == 1
    assert "not callable" in capsys.readouterr().err.lower()
