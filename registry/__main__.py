#!/usr/bin/env python3
r"""Registry Pattern CLI.

Commands:
    python -m registry --version     Show version
    python -m registry info          Show detailed version and system info
    python -m registry build         Build objects from config files
    python -m registry run           Load config, build, and execute

Examples:
    # Show version
    python -m registry --version

    # Show detailed system info (for bug reports)
    python -m registry info

    # Build objects from a config file
    python -m registry build config.yaml

    # Load and execute an entry point
    python -m registry run config.yaml --entry main
"""

from __future__ import annotations

import argparse
import json
import pprint
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from ._version import __version__, print_version_info
from .container import is_build_cfg, normalize_cfg
from .engines import ConfigFileEngine
from .mixin import ContainerMixin


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed version and system information."""
    print_version_info()
    return 0


def load_config_file(filepath: Path) -> Dict[str, Any]:
    """Load a config file using the appropriate engine.

    Args:
        filepath: Path to the config file.

    Returns:
        Parsed config dictionary.

    Raises:
        ValueError: If file extension is not supported.
    """
    ext = filepath.suffix.lstrip(".")
    if not ext:
        raise ValueError(f"Cannot determine file type for: {filepath}")

    try:
        loader = ConfigFileEngine.get_artifact(ext)
    except Exception:
        available = list(ConfigFileEngine.iter_identifiers())
        raise ValueError(
            f"Unsupported config file type: .{ext}\n"
            f"Supported types: {', '.join(f'.{e}' for e in available)}"
        )

    return loader(filepath)


def cmd_build(args: argparse.Namespace) -> int:
    """Build objects from a config file."""
    filepath = Path(args.config_file)
    if not filepath.exists():
        print(f"Error: Config file not found: {filepath}", file=sys.stderr)
        return 1

    try:
        config = load_config_file(filepath)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded config from: {filepath}")
        print("Config contents:")
        pprint.pprint(config)
        print()

    if is_build_cfg(config):
        configs_to_build = [("root", config)]
    elif isinstance(config, dict):
        configs_to_build = []
        for key, value in config.items():
            if is_build_cfg(value):
                configs_to_build.append((key, value))
            elif args.verbose:
                print(f"Skipping non-BuildCfg key: {key}")
    else:
        print("Error: Config must be a BuildCfg or dict of BuildCfgs", file=sys.stderr)
        return 1

    if not configs_to_build:
        print("Error: No BuildCfg entries found in config", file=sys.stderr)
        return 1

    results: Dict[str, Any] = {}
    for name, cfg_dict in configs_to_build:
        try:
            cfg = normalize_cfg(cfg_dict)
            if args.verbose:
                print(f"Building '{name}': type={cfg.type}, repo={cfg.repo}")

            if args.dry_run:
                print(
                    f"[dry-run] Would build: {name} (type={cfg.type}, repo={cfg.repo})"
                )
                continue

            obj = ContainerMixin.build_cfg(cfg)
            results[name] = obj

            if args.verbose:
                print(f"  Built: {type(obj).__name__}")
                if hasattr(obj, "__meta__") and obj.__meta__:
                    print(f"  Meta: {obj.__meta__}")

        except Exception as e:
            print(f"Error building '{name}': {e}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 1

    if not args.dry_run:
        print(f"Successfully built {len(results)} object(s)")
        for name, obj in results.items():
            print(f"  {name}: {type(obj).__name__}")

    if args.output:
        output_path = Path(args.output)
        output_data: Dict[str, Any] = {}
        for name, obj in results.items():
            try:
                if hasattr(obj, "model_dump"):
                    output_data[name] = obj.model_dump()
                elif hasattr(obj, "__dict__"):
                    output_data[name] = {
                        "_type": type(obj).__name__,
                        "_meta": getattr(obj, "__meta__", {}),
                    }
                else:
                    output_data[name] = {"_type": str(type(obj))}
            except Exception:
                output_data[name] = {"_type": str(type(obj))}

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Output written to: {output_path}")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Load config, build objects, and execute an entry point."""
    filepath = Path(args.config_file)
    if not filepath.exists():
        print(f"Error: Config file not found: {filepath}", file=sys.stderr)
        return 1

    try:
        config = load_config_file(filepath)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded config from: {filepath}")

    ContainerMixin.clear_context()

    if is_build_cfg(config):
        try:
            obj = ContainerMixin.build_cfg(normalize_cfg(config))
            ContainerMixin._ctx["main"] = obj
        except Exception as e:
            print(f"Error building config: {e}", file=sys.stderr)
            return 1
    elif isinstance(config, dict):
        for key, value in config.items():
            if is_build_cfg(value):
                try:
                    obj = ContainerMixin.build_named(key, value)
                    if args.verbose:
                        print(f"Built '{key}': {type(obj).__name__}")
                except Exception as e:
                    print(f"Error building '{key}': {e}", file=sys.stderr)
                    return 1

    ctx = ContainerMixin.get_context()

    entry = args.entry or "main"
    if entry not in ctx:
        print(f"Error: Entry point '{entry}' not found in context", file=sys.stderr)
        print(f"Available: {list(ctx.keys())}", file=sys.stderr)
        return 1

    target = ctx[entry]

    if callable(target):
        if args.verbose:
            print(f"Executing '{entry}'...")
        try:
            result = target()
            if result is not None:
                print(f"Result: {result}")
        except Exception as e:
            print(f"Error executing '{entry}': {e}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 1
    else:
        run_method = getattr(target, "run", None) or getattr(target, "__call__", None)
        if run_method and callable(run_method):
            if args.verbose:
                print(f"Running '{entry}.run()'...")
            try:
                result = run_method()
                if result is not None:
                    print(f"Result: {result}")
            except Exception as e:
                print(f"Error running '{entry}': {e}", file=sys.stderr)
                if args.verbose:
                    traceback.print_exc()
                return 1
        else:
            print(f"'{entry}' is not callable and has no run() method", file=sys.stderr)
            print(f"Type: {type(target).__name__}", file=sys.stderr)
            return 1

    return 0


def main() -> int:
    """CLI entry point for registry-pattern."""
    parser = argparse.ArgumentParser(
        prog="python -m registry",
        description="Registry Pattern -- DI Container / IoC Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m registry --version      Show version
  python -m registry info           Show detailed system info
  python -m registry build cfg.yaml Build objects from a config
  python -m registry run cfg.yaml   Build + execute entry point
        """,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"registry-pattern {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    info_parser = subparsers.add_parser(
        "info",
        help="Show detailed version and system information",
        description="Display version, Python, platform, and dependency information.",
    )
    info_parser.set_defaults(func=cmd_info)

    build_parser = subparsers.add_parser(
        "build",
        help="Build objects from a config file",
        description="Load a config file (JSON/YAML/TOML/XML) and build objects.",
    )
    build_parser.add_argument(
        "config_file",
        help="Path to config file (json, yaml, toml, xml)",
    )
    build_parser.add_argument(
        "--output",
        "-o",
        help="Output file for build results (JSON)",
    )
    build_parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be built without building",
    )
    build_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    build_parser.set_defaults(func=cmd_build)

    run_parser = subparsers.add_parser(
        "run",
        help="Load config, build objects, and execute entry point",
        description="Build objects from config and execute the specified entry point.",
    )
    run_parser.add_argument(
        "config_file",
        help="Path to config file (json, yaml, toml, xml)",
    )
    run_parser.add_argument(
        "--entry",
        "-e",
        default="main",
        help="Entry point to execute (default: main)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
