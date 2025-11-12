# Architecture & Design Patterns

## Layered Mixin Architecture

The registry system uses a composition of mixins to achieve separation of concerns:

```
┌─────────────────────────────────────────────┐
│  Concrete Registries                         │
│  (FunctionalRegistry, TypeRegistry, etc.)   │
└─────────────────────────────────────────────┘
                      △
                      │
┌─────────────────────────────────────────────┐
│  RegistryFactorizorMixin                    │
│  - Artifact instantiation                    │
│  - Pydantic scheme extraction               │
│  - Recursive factorization                   │
└─────────────────────────────────────────────┘
                      △
                      │
┌─────────────────────────────────────────────┐
│  MutableValidatorMixin                       │
│  - Write-side validation                     │
│  - Identifier/artifact coercion             │
│  - Batch operations                          │
└─────────────────────────────────────────────┘
                      △
                      │
┌─────────────────────────────────────────────┐
│  ImmutableValidatorMixin                     │
│  - Read-side validation                      │
│  - Caching of validation results             │
│  - Presence/absence assertions               │
└─────────────────────────────────────────────┘
                      △
                      │
┌─────────────────────────────────────────────┐
│  RegistryMutatorMixin                        │
│  - Write operations (_set_artifact, etc.)   │
│  - Batch updates                             │
│  - Clear operations                          │
└─────────────────────────────────────────────┘
                      △
                      │
┌─────────────────────────────────────────────┐
│  RegistryAccessorMixin                       │
│  - Read operations (_get_artifact, etc.)    │
│  - Presence checks                           │
│  - Iteration                                 │
└─────────────────────────────────────────────┘
                      △
                      │
┌─────────────────────────────────────────────┐
│  Storage Backend                             │
│  (ThreadSafeLocalStorage | RemoteStorageProxy)
└─────────────────────────────────────────────┘
```

## Core Design Principles

### 1. Separation of Concerns

Each mixin handles one responsibility:

- **Accessor**: Raw read operations on the mapping
- **Mutator**: Raw write operations on the mapping
- **Validator (Immutable)**: Validate before read, cache results
- **Validator (Mutable)**: Validate before write, batch error capture
- **Factorizor**: Instantiate artifacts with recursive dependency resolution

### 2. Error Model

Validation failures propagate rich context:

```python
ValidationError
├── CoercionError          # Value cannot be coerced to required type
├── ConformanceError       # Callable signature doesn't match
├── InheritanceError       # Class doesn't inherit from required base
└── RegistryError          # Mapping key not found / already exists
```

Each exception carries:
- `message`: Human-readable error text
- `suggestions`: List of actionable remediation steps
- `context`: Free-form dict with debugging metadata

### 3. Generic Type Parameters

Registries are generic over key and value types:

```python
class MyRegistry(MutableValidatorMixin[Hashable, Callable[[int], str]]):
    pass
```

This enables type checking and IDE autocomplete.

### 4. Pluggable Storage

The `_get_mapping()` method abstracts the storage backend:

```python
class LocalRegistry(FunctionalRegistry):
    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        cls._repository = ThreadSafeLocalStorage()  # In-process

class RemoteRegistry(FunctionalRegistry, logic_namespace="my.namespace"):
    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        cls._repository = RemoteStorageProxy(namespace="my.namespace")  # Network
```

Both implement `MutableMapping[KeyType, ValType]`.

## Storage Backends

### ThreadSafeLocalStorage

- **Location**: In-process memory
- **Concurrency**: RLock protects all operations
- **Persistence**: Lost on process termination
- **Latency**: < 1μs per operation
- **Suitable for**: Single-machine research workflows

```python
class MyRegistry(FunctionalRegistry):
    pass  # Uses ThreadSafeLocalStorage by default
```

### RemoteStorageProxy

- **Location**: Network HTTP server
- **Concurrency**: Handled by server (per-namespace locks)
- **Persistence**: In-memory (configurable by server)
- **Latency**: ~5-50ms per operation (network dependent)
- **Suitable for**: Multi-node research clusters

```python
class ClusterRegistry(FunctionalRegistry, logic_namespace="my.namespace"):
    pass  # Connects to server at REGISTRY_SERVER_HOST:REGISTRY_SERVER_PORT
```

Server startup:
```bash
python -m registry.__main__ --host 0.0.0.0 --port 8001
```

Server provides endpoints:
- `GET /health` — Server health check
- `GET /stats` — Storage statistics
- `POST /registry/<namespace>/set` — Write key-value
- `GET /registry/<namespace>/get/<key>` — Read value
- `DELETE /registry/<namespace>/delete/<key>` — Delete key
- `GET /registry/<namespace>/keys` — List all keys

## Validation Cache

The `ImmutableValidatorMixin` includes a process-local validation cache:

```
ValidationCache
├── Key: (id(artifact), type(artifact).__name__, operation)
├── Value: (result: bool, suggestions: List[str], timestamp)
├── TTL: 300s (configurable)
└── LRU Eviction: When size >= max_size (default 1000)
```

Access is guarded by `RLock` to prevent cross-thread corruption.

Configure globally:
```python
from registry.mixin.validator import configure_validation_cache

configure_validation_cache(max_size=500, ttl_seconds=600)
```

## Factorization System

The `RegistryFactorizorMixin` enables recursive instantiation:

```python
artifact = registry.factorize_artifact(
    type="MyClass",
    nested_param={  # Will be recursively factorized
        "_type": "NestedClass",
        "value": 42
    }
)
```

Steps:

1. **Lookup**: Retrieve registered artifact by type identifier
2. **Extract**: Infer Pydantic schema from signature (if not cached)
3. **Validate**: Coerce kwargs using schema
4. **Recursively factorize**: For each parameter with registered type, recursively instantiate
5. **Instantiate**: Call artifact (class or function) with validated kwargs

For file/network sources:

```python
# From JSON/YAML/TOML file
obj = registry.factorize_from_file("MyClass", "config.json")

# From RPC/HTTP endpoint
obj = registry.factorize_from_socket(
    "MyClass",
    socket_config={"url": "http://api.example.com/config"},
    protocol="http"
)
```

## Registry Inheritance Pattern

Concrete registries subclass the appropriate mixin:

```python
from registry import FunctionalRegistry

class LossFunctions(FunctionalRegistry, strict=False):
    """Registry of loss function implementations.
    
    Strict=False allows any callable; strict=True enforces signature.
    """
    pass

# Or with remote storage:
class DistributedLosses(FunctionalRegistry, logic_namespace="research.losses"):
    """Synced across network."""
    pass

# Or with both local and scheme extraction:
class FactorizedLosses(FunctionalRegistry, scheme_namespace="research.loss_schemes"):
    """Enables factorization from config."""
    pass
```

## Identifier Extraction

Each registry must implement `_identifier_of(artifact)`:

- **FunctionalRegistry**: Uses `get_func_name(func)`
- **TypeRegistry**: Uses `get_type_name(cls)`
- **SchemeRegistry**: Uses `artifact.__name__`

Override to customize:
```python
class MyRegistry(FunctionalRegistry):
    @classmethod
    def _identifier_of(cls, item):
        return f"{item.__module__}.{item.__name__}"
```

## Internalization vs Externalization

The validation mixin pattern distinguishes read/write transformations:

```python
@classmethod
def _internalize_artifact(cls, value: Any) -> ValType:
    """Validate and coerce on write."""
    return validated_value

@classmethod
def _externalize_artifact(cls, value: ValType) -> ValType:
    """Validate and coerce on read."""
    return validated_value
```

Example (type registry with strict mode):

```python
def _internalize_artifact(cls, value: Any) -> Type[Cls]:
    # Validate it's a class
    v = _validate_class(value)
    # Check inheritance
    if cls._abstract:
        v = _validate_class_hierarchy(v, abc_class=cls)
    # Check structure/protocol
    if cls._strict:
        protocol = get_protocol(cls)
        v = _validate_class_structure(v, exp_type=protocol)
    return super()._internalize_artifact(v)
```

## Thread Safety Model

The system uses RLock (reentrant lock) consistently:

- **LocalStorage**: One RLock protects entire mapping
- **RemoteStorage**: Server uses per-namespace locks
- **ValidationCache**: One RLock protects cache dict, entries are immutable tuples

This allows:
- Multiple concurrent readers
- Exclusive writers
- Safe re-entrancy (same thread can acquire lock multiple times)

## Module Organization

```
registry/
├── __init__.py                 # Public API exports
├── __main__.py                 # CLI entry point (remote server)
├── _version.py                 # Version string
├── engines.py                  # ConfigFileEngine, SocketEngine
├── fnc_registry.py             # FunctionalRegistry class
├── typ_registry.py             # TypeRegistry class
├── sch_registry.py             # SchemeRegistry class
├── storage.py                  # ThreadSafeLocalStorage, RemoteStorageProxy
├── utils.py                    # Exception hierarchy, helpers
└── mixin/
    ├── __init__.py             # Public mixin exports
    ├── accessor.py             # RegistryAccessorMixin
    ├── mutator.py              # RegistryMutatorMixin
    ├── validator.py            # ImmutableValidatorMixin, MutableValidatorMixin, cache
    └── factorizor.py           # RegistryFactorizorMixin
```

## Extension Points

To customize, override these methods:

### Identifier Extraction
```python
@classmethod
def _identifier_of(cls, item: ValType) -> KeyType:
    """Extract unique identifier from artifact."""
    pass
```

### Validation
```python
@classmethod
def _internalize_artifact(cls, value: Any) -> ValType:
    """Validate/coerce before writing."""
    pass

@classmethod
def _externalize_artifact(cls, value: ValType) -> ValType:
    """Validate/coerce after reading."""
    pass
```

### Storage Backend
```python
@classmethod
def _get_mapping(cls) -> MutableMapping[KeyType, ValType]:
    """Return storage backend (must implement MutableMapping)."""
    pass
```

## Performance Characteristics

| Operation | Local Storage | Remote Storage | Notes |
|-----------|---------------|-----------------|-------|
| get | O(1) ~1μs | O(1) network ~5-50ms | Hash lookup + serialization |
| set | O(1) ~1μs | O(1) network ~5-50ms | Hash insert + serialization |
| delete | O(1) ~1μs | O(1) network ~5-50ms | Hash remove + serialization |
| iter | O(n) ~100ns/item | O(n) network | Must fetch all keys |
| validate | Cached O(1) | Cached O(1) | TTL-based LRU cache |

For distributed workflows, consider batching operations to reduce round-trips.
