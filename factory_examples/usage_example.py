"""Example: Recursive artifact factorization with registries.

This demonstrates:
1. Registering types and functions
2. Recursive factorization from dicts
3. Factorization from config files
4. Extending with custom config loaders
"""

from registry import TypeRegistry, FunctionalRegistry, engines
from pathlib import Path
from typing import Optional


# ============================================================================
# Setup Registries
# ============================================================================

class TypeRepo(TypeRegistry):
    """Repository for types with factorization support."""
    pass


class FuncRepo(FunctionalRegistry):
    """Repository for functions with factorization support."""
    pass


# ============================================================================
# Define and Register Types
# ============================================================================

@TypeRepo.register_artifact
class Foo:
    def __init__(self, x: int, y: str = "default"):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Foo(x={self.x}, y={self.y!r})"


@TypeRepo.register_artifact
class Bar:
    def __init__(self, name: str, foo: Foo, count: int = 1):
        self.name = name
        self.foo = foo
        self.count = count
    
    def __repr__(self):
        return f"Bar(name={self.name!r}, foo={self.foo}, count={self.count})"


# ============================================================================
# Register Functions
# ============================================================================

@FuncRepo.register_artifact
def process(bar: Bar, multiplier: float = 1.0) -> str:
    """Process a Bar instance."""
    result = bar.foo.x * bar.count * multiplier
    return f"{bar.name}: {result}"


# ============================================================================
# Example 1: Basic Factorization from Dict
# ============================================================================

def example_basic():
    print("=" * 60)
    print("Example 1: Basic Factorization from Dict")
    print("=" * 60)
    
    # Factorize Foo directly
    foo = TypeRepo.factorize_artifact("Foo", x=42, y="hello")
    print(f"Factorized: {foo}")
    
    # Factorize Bar with nested Foo (recursive!)
    bar = TypeRepo.factorize_artifact(
        "Bar",
        name="test_bar",
        foo={"x": 100, "y": "nested"},  # This dict becomes a Foo!
        count=3
    )
    print(f"Factorized: {bar}")
    
    # Factorize function call
    result = FuncRepo.factorize_artifact(
        "process",
        bar={"name": "bar2", "foo": {"x": 5, "y": "deep"}, "count": 2},
        multiplier=2.5
    )
    print(f"Function result: {result}")
    print()


# ============================================================================
# Example 2: Factorization from Config Files
# ============================================================================

def example_config_files():
    print("=" * 60)
    print("Example 2: Factorization from Config Files")
    print("=" * 60)
    
    # Create a test config file
    import json
    config = {
        "name": "config_bar",
        "foo": {"x": 999, "y": "from_file"},
        "count": 5
    }
    
    config_path = Path("test_bar_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    # Factorize from file
    bar = TypeRepo.factorize_from_file("Bar", config_path)
    print(f"Factorized from file: {bar}")
    
    # Clean up
    config_path.unlink()
    print()


# ============================================================================
# Example 3: Extending with Custom Config Loader
# ============================================================================

def example_custom_loader():
    print("=" * 60)
    print("Example 3: Custom Config Loader")
    print("=" * 60)
    
    # Register a custom INI loader
    @engines.ConfigFileEngine.register_artifact
    def ini(filepath: Path) -> dict:
        """Load config from INI file."""
        import configparser
        parser = configparser.ConfigParser()
        parser.read(filepath)
        
        # Convert to dict (simplified - just use DEFAULT section)
        return dict(parser["DEFAULT"])
    
    print("Registered custom 'ini' loader")
    print(f"Available loaders: {list(engines.ConfigFileEngine.iter_identifiers())}")
    print()


# ============================================================================
# Example 4: Using host:port notation
# ============================================================================

def example_host_port():
    print("=" * 60)
    print("Example 4: Host:Port Namespace Notation")
    print("=" * 60)
    
    # These demonstrate the supported formats:
    examples = [
        "my.app.models",                      # Local storage
        "my.app.models@server.example.com:8001",  # Remote with @
        "server.example.com:8001/my.app.models",  # Remote with /
    ]
    
    for spec in examples:
        print(f"Namespace spec: {spec}")
        # Note: Would need running server to actually use these
    print()


# ============================================================================
# Example 5: Scheme Storage Inspection
# ============================================================================

def example_scheme_inspection():
    print("=" * 60)
    print("Example 5: Inspect Stored Schemes")
    print("=" * 60)
    
    # Show what schemes were saved
    print("Stored schemes in TypeRepo:")
    for name in TypeRepo._schemetry.keys():
        scheme = TypeRepo._schemetry[name]
        print(f"  {name}: {scheme.__name__}")
        print(f"    Fields: {list(scheme.model_fields.keys())}")
    
    print("\nStored schemes in FuncRepo:")
    for name in FuncRepo._schemetry.keys():
        scheme = FuncRepo._schemetry[name]
        print(f"  {name}: {scheme.__name__}")
        print(f"    Fields: {list(scheme.model_fields.keys())}")
    print()


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    example_basic()
    example_config_files()
    example_custom_loader()
    example_host_port()
    example_scheme_inspection()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
