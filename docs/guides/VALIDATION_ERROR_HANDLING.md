# Validation & Error Handling

## Overview

The registry system provides rich error context with structured exception hierarchy, validation caching, and actionable suggestions for debugging.

## Exception Hierarchy

```
ValidationError (base)
├── CoercionError          # Value cannot be coerced to required type
├── ConformanceError       # Callable signature doesn't match protocol
├── InheritanceError       # Class doesn't inherit from required base
└── RegistryError          # Mapping key not found / already exists
    └── SerializationError # Pickle/JSON serialization failed
```

### ValidationError Base Class

All exceptions carry three components:

```python
class ValidationError(Exception):
    message: str           # Human-readable description
    suggestions: List[str] # Actionable remediation steps
    context: Dict[str, Any]  # Machine-readable debugging info
```

**Access:**

```python
try:
    MyRegistry.register_artifact(invalid)
except ValidationError as e:
    print(e.message)       # "Key must be hashable, got dict"
    print(e.suggestions)   # ["Ensure the key is hashable...", ...]
    print(e.context)       # {"expected_type": "Hashable", "actual_type": "dict"}
```

## Exception Types

### ValidationError

Generic validation failure.

```python
raise ValidationError(
    message="Value is invalid",
    suggestions=["Check the type", "Verify format"],
    context={"value_type": str(type(value))}
)
```

### CoercionError

Value cannot be coerced to required representation.

```python
from registry import CoercionError

try:
    key = {"nested": "dict"}  # Not hashable
    MyRegistry.register_artifact(func)
except CoercionError as e:
    print(e.message)
    # "Key must be hashable, got dict"
    print(e.suggestions)
    # ["Ensure the key is hashable (str, int, tuple, etc.)", ...]
```

Common causes:
- Non-hashable key types (dict, list, set)
- Type mismatches
- Invalid annotations

### ConformanceError

Callable signature doesn't match required signature.

```python
from registry import ConformanceError

class StrictFunctions(FunctionalRegistry, strict=True):
    pass

def wrong_signature(x):  # Missing type annotations
    return x + 1

try:
    StrictFunctions.register_artifact(wrong_signature)
except ConformanceError as e:
    print(e.message)
    # "Method signature validation failed:\n  • Parameter 'x' missing type annotation"
    print(e.suggestions)
    # ["Annotate: x: int", "Align parameter type annotation"]
    print(e.context)
    # {"expected_type": "Callable[[int], int]", "actual_type": "..."}
```

Common causes:
- Missing type annotations
- Wrong parameter count
- Wrong return type
- Variadic parameters (*args, **kwargs)

### InheritanceError

Class doesn't inherit from required base.

```python
from registry import InheritanceError

class PluginBase(ABC):
    @abstractmethod
    def process(self, data): pass

class Plugins(TypeRegistry[PluginBase], abstract=True):
    pass

class NotAPlugin:  # Doesn't inherit from PluginBase
    def process(self, data):
        return data

try:
    Plugins.register_artifact(NotAPlugin)
except InheritanceError as e:
    print(e.message)
    # "NotAPlugin is not a subclass of PluginBase"
    print(e.suggestions)
    # ["Inherit from PluginBase"]
```

### RegistryError

Mapping operation failed (key not found, already exists, etc).

```python
from registry import RegistryError

try:
    MyRegistry.get_artifact("nonexistent")
except RegistryError as e:
    print(e.message)
    # "Key 'nonexistent' is not found in the mapping"
    print(e.suggestions)
    # ["Key 'nonexistent' not found in MyRegistry", ...]
    print(e.context)
    # {"registry_name": "MyRegistry", "available_keys": [...]}
```

## Validation Modes

### Optional Validation (Default)

By default, registries validate basic types:

```python
class Losses(FunctionalRegistry):
    pass

# Validates that it's callable
Losses.register_artifact(lambda x: x)  # OK

try:
    Losses.register_artifact("not a function")
except ValidationError:
    print("Caught!")
```

### Strict Validation

Enforce signature matching:

```python
from typing import Callable

class StrictLosses(FunctionalRegistry, strict=True):
    pass

def properly_typed(y_true: float, y_pred: float) -> float:
    return (y_true - y_pred) ** 2

StrictLosses.register_artifact(properly_typed)  # OK

def untyped(y_true, y_pred):
    return (y_true - y_pred) ** 2

try:
    StrictLosses.register_artifact(untyped)
except ConformanceError as e:
    print(e.message)
    # Shows which parameters are missing type annotations
```

### Abstract/Protocol Validation

Enforce inheritance:

```python
from abc import ABC, abstractmethod

class ModelBase(ABC):
    @abstractmethod
    def forward(self, x): pass

class StrictModels(TypeRegistry[ModelBase], abstract=True, strict=True):
    pass

class MyModel(ModelBase):
    def forward(self, x):
        return x

StrictModels.register_artifact(MyModel)  # OK

class NotAModel:
    def forward(self, x):
        return x

try:
    StrictModels.register_artifact(NotAModel)
except InheritanceError:
    print("Caught!")
```

## Error Context

### What Gets Included

The `context` dict contains machine-readable debugging info:

```python
try:
    MyRegistry.register_artifact(invalid)
except ValidationError as e:
    print(e.context)
    # {
    #   "expected_type": "function",
    #   "actual_type": "str",
    #   "artifact_name": "...",
    #   "operation": "register_artifact",
    #   "registry_name": "MyRegistry",
    #   "registry_type": "FunctionalRegistry"
    # }
```

Common context fields:

| Field | Purpose | Example |
|-------|---------|---------|
| `expected_type` | What was required | `"Callable[[int], str]"` |
| `actual_type` | What was provided | `"function"` |
| `artifact_name` | Name of the artifact | `"my_loss_function"` |
| `operation` | What was being done | `"register_artifact"` |
| `registry_name` | Registry class name | `"MyRegistry"` |
| `available_keys` | What's already registered | `["key1", "key2", ...]` |
| `key_type` | Type of the key | `"str"` |
| `registry_size` | How many items | `42` |

### Accessing Context Programmatically

```python
from registry import RegistryError

try:
    item = MyRegistry.get_artifact("unknown")
except RegistryError as e:
    # Machine-readable check
    if e.context.get("registry_size") == 0:
        print("Registry is empty")
    else:
        available = e.context.get("available_keys", [])
        print(f"Did you mean one of: {available}")
```

## Validation Caching

The system includes a process-local validation cache to speed up repeated validations.

### How It Works

```
Input: artifact
  ↓
Check cache: (id(artifact), type(artifact).__name__, operation)
  ↓
If found and not expired → return cached result
  ↓
Otherwise → validate → cache result → return
```

**Cache Key:**
- Object identity (`id(obj)`)
- Type name (`type(obj).__name__`)
- Operation (`"read"`, `"write"`, etc)

**Cache Value:**
- `(result: bool, suggestions: List[str], timestamp: float)`

**Expiry:**
- TTL: 300 seconds (5 minutes) by default
- LRU eviction when max size reached (default 1000 entries)
- Entries are "touched" on read (updated timestamp)

### Configuring Cache

```python
from registry import FunctionalRegistry
from registry.mixin.validator import configure_validation_cache

class MyRegistry(FunctionalRegistry):
    pass

# Configure globally (affects all registries)
configure_validation_cache(max_size=500, ttl_seconds=600)
# max_size: evict when > 500 entries
# ttl_seconds: expire entries after 10 minutes
```

### Monitoring Cache

```python
from registry.mixin.validator import get_cache_stats

stats = get_cache_stats()
print(f"Cached entries: {stats['active_entries']}")
print(f"Expired: {stats['expired_entries']}")
print(f"Capacity: {stats['max_size']}")
```

### Clearing Cache

```python
from registry import FunctionalRegistry

class MyRegistry(FunctionalRegistry):
    pass

cleared = MyRegistry.clear_validation_cache()
print(f"Cleared {cleared} entries")
```

## Batch Error Handling

When registering multiple items, capture errors per-item:

```python
items = {
    "valid1": valid_function,
    "invalid": "not a function",
    "valid2": another_function,
}

result = MyRegistry.safe_register_batch(items, skip_invalid=True)

print(f"Successful: {result['successful']}")
# → ['valid1', 'valid2']

print(f"Failed: {result['failed']}")
# → ['invalid']

for error in result['errors']:
    print(f"{error['key']}: {error['error']}")
    # → "invalid: 'not a function' is not a function"
    print(f"  Suggestions: {error['suggestions']}")
```

### Options

- `skip_invalid=True` — Continue after errors (default)
- `skip_invalid=False` — Raise on first error

```python
try:
    MyRegistry.safe_register_batch(items, skip_invalid=False)
except ValidationError as e:
    print(f"Failed on: {e.context.get('key')}")
    # Stops at first error
```

## Validation Pipeline

Registration follows this validation sequence:

```
Input artifact
       ↓
1. Type validation (_internalize_artifact)
       ↓
2. Identifier extraction (_identifier_of)
       ↓
3. Identifier validation (_internalize_identifier)
       ↓
4. Cache check (return if valid)
       ↓
5. Full validation (strict/abstract/etc)
       ↓
6. Cache store
       ↓
7. Storage operation
       ↓
Success
```

Override hooks to customize:

```python
class CustomRegistry(FunctionalRegistry):
    @classmethod
    def _internalize_artifact(cls, value):
        # Custom validation on write
        if not callable(value):
            raise ValidationError(
                "Must be callable",
                ["Pass a function"],
                {"type": type(value).__name__}
            )
        return value
    
    @classmethod
    def _identifier_of(cls, item):
        # Extract custom identifier
        return f"{item.__module__}.{item.__name__}"
```

## Common Error Scenarios

### Scenario 1: Wrong Type

```python
try:
    MyRegistry.register_artifact("not a function")
except ValidationError as e:
    # Expected: function, Actual: str
    # Suggestion: Pass a function object, not a value
    pass
```

**Solution:**
```python
def my_function():
    pass

MyRegistry.register_artifact(my_function)  # ✓ Correct
```

### Scenario 2: Key Already Exists

```python
@MyRegistry.register_artifact
def my_loss(): pass

try:
    @MyRegistry.register_artifact
    def my_loss(): pass  # Same name
except RegistryError as e:
    # Key 'my_loss' already exists
    # Suggestion: Use _update_artifact to modify, or delete first
    pass
```

**Solution:**
```python
# Option 1: Update
MyRegistry._update_artifact("my_loss", new_function)

# Option 2: Unregister first
MyRegistry.unregister_identifier("my_loss")
MyRegistry.register_artifact(my_loss)
```

### Scenario 3: Missing Type Annotation (Strict Mode)

```python
class StrictFunctions(FunctionalRegistry, strict=True):
    pass

def untyped(x):  # No type hint
    return x + 1

try:
    StrictFunctions.register_artifact(untyped)
except ConformanceError as e:
    # Parameter 'x' missing type annotation
    # Suggestion: Annotate: x: <type>
    pass
```

**Solution:**
```python
def typed(x: int) -> int:
    return x + 1

StrictFunctions.register_artifact(typed)  # ✓ Correct
```

### Scenario 4: Server Connection Failed

```python
class RemoteRegistry(FunctionalRegistry, logic_namespace="my.namespace"):
    pass

try:
    RemoteRegistry.register_artifact(my_function)
except ValidationError as e:
    if "Cannot connect" in e.message:
        # Start the server first
        # Start server: python -m registry.__main__
        pass
```

**Solution:**
```bash
# Terminal 1: Start server
python -m registry.__main__ --host 0.0.0.0 --port 8001

# Terminal 2: Use registry
# Now connections work
```

## Debugging Tips

### 1. Print Full Exception

```python
try:
    artifact = MyRegistry.get_artifact("key")
except ValidationError as e:
    print(e)  # Prints formatted message with suggestions
```

### 2. Log Context

```python
import json

try:
    MyRegistry.register_artifact(invalid)
except ValidationError as e:
    print(json.dumps(e.context, indent=2))
    # Machine-readable format for logging
```

### 3. Validate Before Registering

```python
try:
    validated = MyRegistry.validate_artifact(candidate)
    # If no exception, it's valid
    MyRegistry.register_artifact(validated)
except ValidationError as e:
    print(f"Invalid: {e.suggestions}")
```

### 4. Check Registry State

```python
report = MyRegistry.validate_registry_state()

if report['validation_errors']:
    print("Found invalid artifacts:")
    for error in report['validation_errors']:
        print(f"  {error['key']}: {error['error']}")

print(f"Cache stats: {report['cache_stats']}")
```

### 5. Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("registry")
logger.setLevel(logging.DEBUG)

# Now all operations print debug info
```

## Exception Handling Patterns

### Pattern 1: Graceful Degradation

```python
from registry import RegistryError

try:
    loss_fn = Losses.get_artifact(user_input)
except RegistryError:
    loss_fn = Losses.get_artifact("mse")  # Fallback to default
```

### Pattern 2: Error Reporting

```python
from registry import ValidationError

try:
    Models.register_artifact(user_class)
except ValidationError as e:
    # Log for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.error(
        "Registration failed: %s",
        e.message,
        extra={"context": e.context}
    )
    # Show user-friendly message
    print(f"Could not register: {e.suggestions[0]}")
```

### Pattern 3: Batch Processing with Recovery

```python
successful = []
failed = []

for item in items:
    try:
        MyRegistry.register_artifact(item)
        successful.append(item)
    except ValidationError as e:
        failed.append((item, e))
        continue  # Don't stop

print(f"Registered {len(successful)}, failed {len(failed)}")
```

## Next Steps

- See [FACTORIZATION_PATTERN.md](FACTORIZATION_PATTERN.md) for recursive validation
- See [API_REFERENCE.md](API_REFERENCE.md) for method-level error documentation
