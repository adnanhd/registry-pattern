# examples/comprehensive_demo.py
"""Comprehensive demonstration of registry features."""

import asyncio
from pathlib import Path
from registry import ObjectRegistry
from registry.plugins import get_plugin_manager
from registry.core.memory_optimized import MemoryEfficientRegistry, MemoryMonitor

class ComprehensiveRegistry(MemoryEfficientRegistry, ObjectRegistry[object]):
    """Registry with all optimizations and features."""
    pass

async def comprehensive_demo():
    """Comprehensive demonstration of registry features."""
    print("=== Comprehensive Registry Demo ===\n")
    
    # Load all available plugins with fresh instances
    plugin_manager = get_plugin_manager()
    available_plugins = ['pydantic', 'observability', 'serialization', 'caching', 'remote', 'cli']
    
    loaded_plugins = []
    for plugin_name in available_plugins:
        plugin = plugin_manager.load_plugin(plugin_name, force_new_instance=True)
        if plugin:
            try:
                plugin.initialize(ComprehensiveRegistry)
                loaded_plugins.append(plugin_name)
                print(f"✓ Loaded {plugin_name} plugin")
            except Exception as e:
                print(f"✗ Failed to initialize {plugin_name} plugin: {e}")
        else:
            print(f"✗ Failed to load {plugin_name} plugin")
    
    print(f"\nLoaded {len(loaded_plugins)} plugins: {', '.join(loaded_plugins)}\n")
    
    # Setup monitoring
    monitor = MemoryMonitor(ComprehensiveRegistry)
    monitor.start_monitoring()
    print("✓ Memory monitoring started")
    
    # Setup caching if available
    if 'caching' in loaded_plugins:
        try:
            ComprehensiveRegistry.setup_memory_cache("lru", max_size=1000)
            ComprehensiveRegistry.setup_disk_cache("./demo_cache")
            print("✓ Caching configured")
        except Exception as e:
            print(f"✗ Caching setup failed: {e}")
    
    # Register various objects
    print("\n--- Registering Objects ---")
    
    # Register simple objects
    for i in range(10):
        obj = f"demo_object_{i}"
        ComprehensiveRegistry.register_artifact(f"demo_{i}", obj)
    
    print("✓ Registered 10 demo objects")
    
    # Register complex objects
    class DemoClass:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        
        def __str__(self):
            return f"DemoClass(name={self.name}, value={self.value})"
    
    for i in range(5):
        obj = DemoClass(f"demo_{i}", i * 10)
        ComprehensiveRegistry.register_artifact(f"complex_{i}", obj)
    
    print("✓ Registered 5 complex objects")
    
    # Test retrieval
    print("\n--- Testing Retrieval ---")
    retrieved = ComprehensiveRegistry.get_artifact("demo_0")
    print(f"Retrieved: {retrieved}")
    
    complex_obj = ComprehensiveRegistry.get_artifact("complex_0")
    print(f"Retrieved complex: {complex_obj}")
    
    # Test serialization if available
    if 'serialization' in loaded_plugins:
        print("\n--- Testing Serialization ---")
        try:
            ComprehensiveRegistry.save_to_json("demo_registry.json")
            print("✓ Registry saved to JSON")
            
            # Clear and reload
            original_size = ComprehensiveRegistry._len_mapping()
            ComprehensiveRegistry.clear_artifacts()
            print("✓ Registry cleared")
            
            ComprehensiveRegistry.load_from_json("demo_registry.json")
            new_size = ComprehensiveRegistry._len_mapping()
            print(f"✓ Registry reloaded (size: {original_size} -> {new_size})")
            
        except Exception as e:
            print(f"✗ Serialization failed: {e}")
    
    # Test memory optimization
    print("\n--- Memory Optimization ---")
    memory_before = ComprehensiveRegistry.get_memory_report()
    print(f"Memory before optimization: {memory_before['memory_usage']['total_bytes']} bytes")
    
    optimization_results = ComprehensiveRegistry.optimize_memory()
    print("✓ Memory optimization completed")
    
    memory_after = ComprehensiveRegistry.get_memory_report()
    print(f"Memory after optimization: {memory_after['memory_usage']['total_bytes']} bytes")
    
    # Test caching stats if available
    if 'caching' in loaded_plugins:
        try:
            cache_stats = ComprehensiveRegistry.get_cache_stats()
            print(f"Cache statistics: {cache_stats}")
        except Exception as e:
            print(f"Cache stats not available: {e}")
    
    # Test remote operations if available
    if 'remote' in loaded_plugins:
        print("\n--- Testing Remote Operations ---")
        try:
            # This would require Redis to be running
            ComprehensiveRegistry.setup_redis_backend()
            ComprehensiveRegistry.sync_to_redis()
            print("✓ Synced to Redis")
        except Exception as e:
            print(f"✗ Remote operations failed: {e}")
    
    # Stop monitoring and get results
    monitor.stop_monitoring()
    
    if monitor.snapshots:
        trend = monitor.get_memory_trend(minutes=1)
        if trend:
            print(f"\nMemory trend: {trend}")
        
        leak_detection = monitor.detect_memory_leaks()
        if leak_detection:
            print(f"⚠️ {leak_detection}")
        else:
            print("✓ No memory leaks detected")
    
    # Final statistics
    print(f"\n--- Final Statistics ---")
    print(f"Registry size: {ComprehensiveRegistry._len_mapping()} items")
    print(f"Loaded plugins: {len(loaded_plugins)}")
    
    # Cleanup
    Path("demo_registry.json").unlink(missing_ok=True)
    if Path("demo_cache").exists():
        import shutil
        shutil.rmtree("demo_cache", ignore_errors=True)
    
    print("\n✓ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(comprehensive_demo())
