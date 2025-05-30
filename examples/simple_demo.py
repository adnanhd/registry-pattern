# examples/simple_demo.py
"""Simple demo that works without complex dependencies."""

from registry import ObjectRegistry
from registry.plugins import load_plugin
from types import new_class

def main():
    print("=== Simple Registry Demo ===\n")

    # Create basic registry
    class SimpleRegistry(ObjectRegistry[str]):
        pass

    print("✓ Created SimpleRegistry")

    # Test basic functionality
    print("\n--- Basic Registration ---")
    SimpleRegistry.register_artifact("key1", "value1")
    SimpleRegistry.register_artifact("key2", "value2")
    SimpleRegistry.register_artifact("key3", "value3")

    print(f"✓ Registered 3 items, registry size: {SimpleRegistry._len_mapping()}")

    # Test retrieval
    print("\n--- Testing Retrieval ---")
    retrieved = SimpleRegistry.get_artifact("key1")
    print(f"Retrieved key1: {retrieved}")

    # Test plugins that don't require external dependencies
    print("\n--- Testing Plugins ---")

    # Test serialization (basic JSON)
    serialization_plugin = load_plugin('serialization')
    if serialization_plugin:
        try:
            serialization_plugin.initialize(SimpleRegistry)

            # Test JSON serialization
            SimpleRegistry.save_to_json("simple_test.json")
            print("✓ Saved to JSON")

            # Test reload
            original_size = SimpleRegistry._len_mapping()
            SimpleRegistry.clear_artifacts()
            SimpleRegistry.load_from_json("simple_test.json")
            new_size = SimpleRegistry._len_mapping()

            print(f"✓ Reloaded from JSON (size: {original_size} -> {new_size})")

            # Cleanup
            import os
            if os.path.exists("simple_test.json"):
                os.remove("simple_test.json")

        except Exception as e:
            print(f"✗ Serialization test failed: {e}")

    # Test observability (if available)
    obs_registry_class = new_class("ObsRegistry", (ObjectRegistry[str],), {})
    obs_plugin = load_plugin('observability')
    if obs_plugin:
        try:
            obs_plugin.initialize(obs_registry_class)

            # Register with observability
            obs_registry_class.register_artifact("obs1", "value1")
            obs_registry_class.register_artifact("obs2", "value2")

            print(f"✓ Observability plugin works (size: {obs_registry_class._len_mapping()})")

        except Exception as e:
            print(f"✗ Observability test failed: {e}")

    # Test memory optimization
    print("\n--- Testing Memory Optimization ---")
    try:
        from registry.core.memory_optimized import MemoryEfficientRegistry

        class MemoryRegistry(MemoryEfficientRegistry, ObjectRegistry[str]):
            pass

        # Register items
        for i in range(10):
            MemoryRegistry.register_artifact(f"mem_key_{i}", f"mem_value_{i}")

        print(f"✓ Memory-optimized registry works (size: {MemoryRegistry._len_mapping()})")

        # Test memory optimization
        results = MemoryRegistry.optimize_memory()
        print(f"✓ Memory optimization completed: {len(results)} metrics")

    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}")

    # Test Pydantic (if available)
    print("\n--- Testing Pydantic Plugin ---")
    try:
        from pydantic import BaseModel
        from registry import TypeRegistry

        class PydanticRegistry(TypeRegistry[object]):
            pass

        pydantic_plugin = load_plugin('pydantic')
        if pydantic_plugin:
            pydantic_plugin.initialize(PydanticRegistry)

            class TestModel(BaseModel):
                name: str
                value: int

            # Register the model
            PydanticRegistry.register_artifact(TestModel)

            # Test validation
            data = {"name": "test", "value": 42}
            validated = PydanticRegistry.validate_with_pydantic(data, TestModel)

            print(f"✓ Pydantic validation works: {validated.name}, {validated.value}")

    except ImportError:
        print("✗ Pydantic not available")
    except Exception as e:
        print(f"✗ Pydantic test failed: {e}")

    print(f"\n--- Final Results ---")
    print(f"SimpleRegistry size: {SimpleRegistry._len_mapping()}")
    print("✓ Simple demo completed successfully!")

if __name__ == "__main__":
    main()
