#!/usr/bin/env python3
"""Simple example showing synchronized TypeRegistry usage."""

import sys
from pathlib import Path

from registry import TypeRegistry


# Base class for all layers
class BaseLayer:
    pass


# Example layer classes
class LinearLayer(BaseLayer):
    pass


class ConvLayer(BaseLayer):
    pass


class TransformerLayer(BaseLayer):
    pass


# Local sync (file-based, same machine)
class RemoteSyncedRegistry(
    TypeRegistry[BaseLayer], proxy_namespace="nn.layers", abstract=False
):
    """Registry synced across processes via file."""

    pass


class LocalRegistry(TypeRegistry[BaseLayer], abstract=True):
    """Registry without synchronization."""

    pass


# Network sync (TCP-based, cross-machine)
# NOTE: Requires server running first:
#   python -m registry


def main():
    if len(list(RemoteSyncedRegistry.iter_identifiers())) == 0:
        print("Registering types...")
        RemoteSyncedRegistry.register_artifact(LinearLayer)
        RemoteSyncedRegistry.register_artifact(ConvLayer)
        RemoteSyncedRegistry.register_artifact(TransformerLayer)

        print(f"\nRegistered: {list(RemoteSyncedRegistry.iter_identifiers())}")
    else:
        print("Listing already registered types:")
        for identifier in RemoteSyncedRegistry.iter_identifiers():
            print(f"\t{identifier} : ", RemoteSyncedRegistry.get_artifact(identifier))


if __name__ == "__main__":
    main()
