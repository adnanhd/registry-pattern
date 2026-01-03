#!/usr/bin/env python3
r"""Registry Pattern CLI.

Commands:
    python -m registry --version     Show version
    python -m registry info          Show detailed version and system info
    python -m registry server        Start the registry storage server

Examples:
    # Show version
    python -m registry --version

    # Show detailed system info (for bug reports)
    python -m registry info

    # Start registry server
    python -m registry server --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import argparse
import sys


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed version and system information."""
    from ._version import print_version_info

    print_version_info()
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
