#!/usr/bin/env python3
r"""Registry Pattern CLI.

Commands:
    python -m registry --version     Show version
    python -m registry info          Show detailed version and system info
    python -m registry server        Start the registry storage server
    python -m registry build         Build objects from config files
    python -m registry run           Load config, build, and execute

Examples:
    # Show version
    python -m registry --version

    # Show detailed system info (for bug reports)
    python -m registry info

    # Start registry server
    python -m registry server --host 0.0.0.0 --port 8001

    # Build objects from config file
    python -m registry build config.yaml --repo models

    # Connect to remote registry and build
    python -m registry build config.json --server http://localhost:8001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed version and system information."""
    from ._version import print_version_info

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
    from .engines import ConfigFileEngine

    ext = filepath.suffix.lstrip(".")
    if not ext:
        raise ValueError(f"Cannot determine file type for: {filepath}")

    try:
        loader = ConfigFileEngine.get_artifact(ext)
    except Exception:
        available = list(ConfigFileEngine.keys())
        raise ValueError(
            f"Unsupported config file type: .{ext}\n"
            f"Supported types: {', '.join(f'.{e}' for e in available)}"
        )

    return loader(filepath)


def connect_to_server(server_url: str) -> Optional[Dict[str, Any]]:
    """Connect to a registry server and fetch available repos.

    Args:
        server_url: URL of the registry server.

    Returns:
        Server info dict or None on failure.
    """
    try:
        import requests
    except ImportError:
        print(
            "Error: requests library required for server connection.", file=sys.stderr
        )
        print("Install with: pip install requests", file=sys.stderr)
        return None

    try:
        response = requests.get(f"{server_url.rstrip('/')}/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error connecting to server: {e}", file=sys.stderr)
        return None


def setup_remote_repos(server_url: str, namespaces: list) -> Dict[str, type]:
    """Set up remote registry proxies for given namespaces.

    Args:
        server_url: URL of the registry server (e.g., http://localhost:8001)
        namespaces: List of namespace names to connect to.

    Returns:
        Dict mapping namespace names to remote registry proxy classes.
    """
    from urllib.parse import urlparse

    from .mixin import ContainerMixin
    from .storage import RemoteStorageProxy
    from .typ_registry import TypeRegistry

    parsed = urlparse(server_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8001

    repos = {}
    for ns in namespaces:
        # Create a dynamic registry class that uses remote storage

        class RemoteRegistry(TypeRegistry[object], ContainerMixin):
            """Remote registry connected to server."""

            pass

        # Replace storage with remote proxy
        RemoteRegistry._storage = RemoteStorageProxy(
            namespace=ns,
            host=host,
            port=port,
        )
        RemoteRegistry.__name__ = f"RemoteRegistry[{ns}]"
        repos[ns] = RemoteRegistry

    return repos


def cmd_build(args: argparse.Namespace) -> int:
    """Build objects from a config file."""
    import json
    import pprint

    from .container import BuildCfg, is_build_cfg, normalize_cfg
    from .mixin import ContainerMixin

    filepath = Path(args.config_file)
    if not filepath.exists():
        print(f"Error: Config file not found: {filepath}", file=sys.stderr)
        return 1

    # Load config
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

    # Connect to server if specified
    if args.server:
        server_info = connect_to_server(args.server)
        if server_info:
            namespaces = list(server_info.get("namespace_sizes", {}).keys())
            print(f"Connected to server: {args.server}")
            print(f"Server namespaces: {namespaces}")

            # Set up remote repos
            if namespaces:
                repos = setup_remote_repos(args.server, namespaces)
                ContainerMixin.configure_repos(repos)
                if args.verbose:
                    print(f"Configured {len(repos)} remote repo(s)")
        else:
            return 1

    # Handle different config structures
    if is_build_cfg(config):
        # Single BuildCfg at root
        configs_to_build = [("root", config)]
    elif isinstance(config, dict):
        # Multiple named configs
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

    # Build each config
    results = {}
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
                import traceback

                traceback.print_exc()
            return 1

    if not args.dry_run:
        print(f"Successfully built {len(results)} object(s)")
        for name, obj in results.items():
            print(f"  {name}: {type(obj).__name__}")

    # Output as JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {}
        for name, obj in results.items():
            try:
                # Try to serialize
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
    import importlib.util

    from .container import is_build_cfg, normalize_cfg
    from .mixin import ContainerMixin

    filepath = Path(args.config_file)
    if not filepath.exists():
        print(f"Error: Config file not found: {filepath}", file=sys.stderr)
        return 1

    # Load config
    try:
        config = load_config_file(filepath)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded config from: {filepath}")

    # Connect to server if specified
    if args.server:
        server_info = connect_to_server(args.server)
        if server_info:
            namespaces = list(server_info.get("namespace_sizes", {}).keys())
            if args.verbose:
                print(f"Connected to server: {args.server}")
                print(f"Server namespaces: {namespaces}")

            # Set up remote repos
            if namespaces:
                repos = setup_remote_repos(args.server, namespaces)
                ContainerMixin.configure_repos(repos)
        else:
            return 1

    # Build all objects into context
    ContainerMixin.clear_context()

    if is_build_cfg(config):
        # Single config - build and run if callable
        try:
            obj = ContainerMixin.build_cfg(normalize_cfg(config))
            ContainerMixin._ctx["main"] = obj
        except Exception as e:
            print(f"Error building config: {e}", file=sys.stderr)
            return 1
    elif isinstance(config, dict):
        # Multiple configs - build all
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

    # Execute entry point
    entry = args.entry or "main"
    if entry not in ctx:
        print(f"Error: Entry point '{entry}' not found in context", file=sys.stderr)
        print(f"Available: {list(ctx.keys())}", file=sys.stderr)
        return 1

    target = ctx[entry]

    # If target is callable, call it
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
                import traceback

                traceback.print_exc()
            return 1
    else:
        # If it has a 'run' or '__call__' method
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
                    import traceback

                    traceback.print_exc()
                return 1
        else:
            print(f"'{entry}' is not callable and has no run() method", file=sys.stderr)
            print(f"Type: {type(target).__name__}", file=sys.stderr)
            return 1

    return 0


def cmd_server(args: argparse.Namespace) -> int:
    """Start the registry storage server."""
    import json
    import logging
    import os
    from collections import defaultdict
    from datetime import datetime
    from threading import Lock
    from typing import Any, Dict
    from urllib.parse import unquote

    try:
        from flask import Flask, jsonify, request
    except ImportError:
        print("Error: Flask is required for the server.", file=sys.stderr)
        print("Install with: pip install flask", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    class RegistryStorage:
        """Thread-safe in-memory storage for registry namespaces."""

        def __init__(self) -> None:
            self._storage: Dict[str, Dict[str, Any]] = defaultdict(dict)
            self._locks: Dict[str, Lock] = defaultdict(Lock)
            self._stats = {
                "created_at": datetime.now().isoformat(),
                "total_requests": 0,
                "get_requests": 0,
                "set_requests": 0,
                "delete_requests": 0,
            }
            self._stats_lock = Lock()

        def _get_lock(self, namespace: str) -> Lock:
            return self._locks[namespace]

        def _increment_stat(self, key: str) -> None:
            with self._stats_lock:
                self._stats["total_requests"] += 1
                if key in self._stats:
                    self._stats[key] += 1

        def get(self, namespace: str, key: str) -> Any | None:
            self._increment_stat("get_requests")
            with self._get_lock(namespace):
                return self._storage[namespace].get(key)

        def set(self, namespace: str, key: str, value: Any) -> None:
            self._increment_stat("set_requests")
            with self._get_lock(namespace):
                self._storage[namespace][key] = value
                logger.debug(f"Set {namespace}.{key}")

        def delete(self, namespace: str, key: str) -> bool:
            self._increment_stat("delete_requests")
            with self._get_lock(namespace):
                if key in self._storage[namespace]:
                    del self._storage[namespace][key]
                    logger.debug(f"Deleted {namespace}.{key}")
                    return True
                return False

        def keys(self, namespace: str) -> list[str]:
            self._increment_stat("get_requests")
            with self._get_lock(namespace):
                return list(self._storage[namespace].keys())

        def values(self, namespace: str) -> list[Any]:
            self._increment_stat("get_requests")
            with self._get_lock(namespace):
                return list(self._storage[namespace].values())

        def items(self, namespace: str) -> list[tuple[str, Any]]:
            self._increment_stat("get_requests")
            with self._get_lock(namespace):
                return list(self._storage[namespace].items())

        def length(self, namespace: str) -> int:
            with self._get_lock(namespace):
                return len(self._storage[namespace])

        def contains(self, namespace: str, key: str) -> bool:
            with self._get_lock(namespace):
                return key in self._storage[namespace]

        def clear(self, namespace: str) -> int:
            self._increment_stat("delete_requests")
            with self._get_lock(namespace):
                count = len(self._storage[namespace])
                self._storage[namespace].clear()
                logger.info(f"Cleared {namespace} ({count} entries)")
                return count

        def get_stats(self) -> Dict[str, Any]:
            with self._stats_lock:
                return {
                    **self._stats,
                    "namespaces": len(self._storage),
                    "total_entries": sum(
                        len(entries) for entries in self._storage.values()
                    ),
                    "namespace_sizes": {
                        ns: len(entries) for ns, entries in self._storage.items()
                    },
                }

    def create_app(storage: RegistryStorage) -> Flask:
        """Create Flask application with registry endpoints."""
        app = Flask(__name__)

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "healthy", "service": "registry-server"}), 200

        @app.route("/stats", methods=["GET"])
        def stats():
            return jsonify(storage.get_stats()), 200

        @app.route("/registry/<path:namespace>/get/<path:key_encoded>", methods=["GET"])
        def get_value(namespace: str, key_encoded: str):
            try:
                key_json = unquote(key_encoded)
                key_serialized = json.loads(key_json)
                key_str = json.dumps(key_serialized)

                value = storage.get(namespace, key_str)
                if value is None:
                    return jsonify({"error": "Key not found"}), 404

                return jsonify({"value": value}), 200
            except Exception as e:
                logger.error(f"Error in get_value: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/registry/<path:namespace>/set", methods=["POST"])
        def set_value(namespace: str):
            try:
                data = request.get_json()
                key_serialized = data["key"]
                value_serialized = data["value"]

                key_str = json.dumps(key_serialized)
                storage.set(namespace, key_str, value_serialized)
                return jsonify({"status": "success"}), 200
            except Exception as e:
                logger.error(f"Error in set_value: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route(
            "/registry/<path:namespace>/delete/<path:key_encoded>", methods=["DELETE"]
        )
        def delete_value(namespace: str, key_encoded: str):
            try:
                key_json = unquote(key_encoded)
                key_serialized = json.loads(key_json)
                key_str = json.dumps(key_serialized)

                deleted = storage.delete(namespace, key_str)
                if not deleted:
                    return jsonify({"error": "Key not found"}), 404

                return jsonify({"status": "deleted"}), 200
            except Exception as e:
                logger.error(f"Error in delete_value: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/registry/<path:namespace>/keys", methods=["GET"])
        def get_keys(namespace: str):
            try:
                keys_str = storage.keys(namespace)
                keys = [json.loads(k) for k in keys_str]
                return jsonify({"keys": keys}), 200
            except Exception as e:
                logger.error(f"Error in get_keys: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/registry/<path:namespace>/values", methods=["GET"])
        def get_values(namespace: str):
            try:
                values = storage.values(namespace)
                return jsonify({"values": values}), 200
            except Exception as e:
                logger.error(f"Error in get_values: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/registry/<path:namespace>/items", methods=["GET"])
        def get_items(namespace: str):
            try:
                items_str = storage.items(namespace)
                items = [[json.loads(k), v] for k, v in items_str]
                return jsonify({"items": items}), 200
            except Exception as e:
                logger.error(f"Error in get_items: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/registry/<path:namespace>/length", methods=["GET"])
        def get_length(namespace: str):
            try:
                length = storage.length(namespace)
                return jsonify({"length": length}), 200
            except Exception as e:
                logger.error(f"Error in get_length: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route(
            "/registry/<path:namespace>/contains/<path:key_encoded>", methods=["GET"]
        )
        def check_contains(namespace: str, key_encoded: str):
            try:
                key_json = unquote(key_encoded)
                key_serialized = json.loads(key_json)
                key_str = json.dumps(key_serialized)

                contains = storage.contains(namespace, key_str)
                return jsonify({"contains": contains}), 200
            except Exception as e:
                logger.error(f"Error in check_contains: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/registry/<path:namespace>/clear", methods=["DELETE"])
        def clear_namespace(namespace: str):
            try:
                count = storage.clear(namespace)
                return jsonify({"status": "cleared", "count": count}), 200
            except Exception as e:
                logger.error(f"Error in clear_namespace: {e}")
                return jsonify({"error": str(e)}), 500

        return app

    storage = RegistryStorage()
    app = create_app(storage)

    logger.info(f"Starting registry server on {args.host}:{args.port}")
    logger.info(f"Health check: http://{args.host}:{args.port}/health")
    logger.info(f"Statistics: http://{args.host}:{args.port}/stats")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        stats = storage.get_stats()
        logger.info(f"Final stats: {stats}")

    return 0


def main() -> int:
    """CLI entry point for registry-pattern."""
    from ._version import __version__

    parser = argparse.ArgumentParser(
        prog="python -m registry",
        description="Registry Pattern - DI Container / IoC Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m registry --version      Show version
  python -m registry info           Show detailed system info
  python -m registry server         Start registry server
        """,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"registry-pattern {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show detailed version and system information",
        description="Display version, Python, platform, and dependency information.",
    )
    info_parser.set_defaults(func=cmd_info)

    # build command
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
        "--server",
        "-s",
        help="Registry server URL to connect to",
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

    # run command
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
        "--server",
        "-s",
        help="Registry server URL to connect to",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    run_parser.set_defaults(func=cmd_run)

    # server command
    server_parser = subparsers.add_parser(
        "server",
        help="Start the registry storage server",
        description="Start a Flask-based registry storage server.",
    )
    server_parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind to (default: 8001)",
    )
    server_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    server_parser.set_defaults(func=cmd_server)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
