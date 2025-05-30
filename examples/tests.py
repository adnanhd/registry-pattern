# Complete examples of ObjectRegistry modes

from registry import ObjectRegistry

class ExampleRegistry(ObjectRegistry[object]):
    pass

print("=== ObjectRegistry Registration Modes ===\n")

# MODE 1: Two-argument (explicit keys)
print("MODE 1: Two-argument registration with explicit keys")
ExampleRegistry.register_artifact("user_key", "user_value")
print(f"✓ Registered: key='user_key', value='user_value'")
print(f"  Has identifier 'user_key': {ExampleRegistry.has_identifier('user_key')}")
print(f"  Has artifact 'user_value': {ExampleRegistry.has_artifact('user_value')}")
print(f"  Get artifact: {ExampleRegistry.get_artifact('user_key')}")

# Unregister using the key
ExampleRegistry.unregister_identifier("user_key")
print(f"✓ Unregistered using key")
print(f"  Has identifier after unregister: {ExampleRegistry.has_identifier('user_key')}")

print()

# MODE 2: Single-argument (auto keys)
print("MODE 2: Single-argument registration with auto keys")
my_object = {"name": "test_object", "value": 42}
ExampleRegistry.register_artifact(my_object)  # Only one argument!
print(f"✓ Registered: object={my_object}")
print(f"  Has artifact: {ExampleRegistry.has_artifact(my_object)}")

# Unregister using the artifact itself
ExampleRegistry.unregister_artifact(my_object)
print(f"✓ Unregistered using artifact")
print(f"  Has artifact after unregister: {ExampleRegistry.has_artifact(my_object)}")

print()

# MODE 3: Practical example with objects
print("MODE 3: Practical example with custom objects")

class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def __str__(self):
        return f"User({self.name}, {self.email})"

# Create users
alice = User("Alice", "alice@example.com")
bob = User("Bob", "bob@example.com")

# Option A: Register with meaningful keys
ExampleRegistry.register_artifact("alice", alice)
ExampleRegistry.register_artifact("bob", bob)

print(f"✓ Users registered with string keys")
print(f"  Alice: {ExampleRegistry.get_artifact('alice')}")
print(f"  Bob: {ExampleRegistry.get_artifact('bob')}")

# Unregister by key
ExampleRegistry.unregister_identifier("alice")
ExampleRegistry.unregister_identifier("bob")
print(f"✓ Users unregistered by key")

# Option B: Register without explicit keys (auto-tracking)
charlie = User("Charlie", "charlie@example.com")
diana = User("Diana", "diana@example.com")

ExampleRegistry.register_artifact(charlie)  # Auto key (memory address)
ExampleRegistry.register_artifact(diana)    # Auto key (memory address)

print(f"✓ Users registered with auto keys")
print(f"  Has Charlie: {ExampleRegistry.has_artifact(charlie)}")
print(f"  Has Diana: {ExampleRegistry.has_artifact(diana)}")

# Unregister by artifact
ExampleRegistry.unregister_artifact(charlie)
ExampleRegistry.unregister_artifact(diana)
print(f"✓ Users unregistered by artifact")

print()

# COMMON MISTAKE DEMONSTRATION
print("COMMON MISTAKE: Mixing registration modes")
try:
    # Register with explicit key
    ExampleRegistry.register_artifact("mistake_key", "mistake_value")
    
    # Try to unregister with wrong method
    ExampleRegistry.unregister_artifact("mistake_key")  # ❌ This will fail!
    
except Exception as e:
    print(f"✗ Error (as expected): {type(e).__name__}")
    print(f"  Reason: Used two-argument registration but artifact-based unregistration")
    
    # Fix it properly
    ExampleRegistry.unregister_identifier("mistake_key")  # ✓ Correct way
    print(f"✓ Fixed by using unregister_identifier()")

print("\n=== Summary ===")
print("✓ Two-argument registration → unregister_identifier(key)")
print("✓ Single-argument registration → unregister_artifact(object)")
print("✗ Never mix registration and unregistration modes!")
