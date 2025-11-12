# Documentation Index & Quick Reference

## Quick Navigation

Start here based on your needs:

| I want to... | Go to... |
|---|---|
| **Get started quickly** | [GETTING_STARTED.md](GETTING_STARTED.md) |
| **Understand the design** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Look up a method** | [API_REFERENCE.md](API_REFERENCE.md) |
| **Learn about storage** | [STORAGE_BACKENDS.md](STORAGE_BACKENDS.md) |
| **Handle errors properly** | [VALIDATION_ERROR_HANDLING.md](VALIDATION_ERROR_HANDLING.md) |
| **Use configuration-driven instantiation** | [FACTORIZATION_PATTERN.md](FACTORIZATION_PATTERN.md) |
| **See practical examples** | [EXAMPLES.md](EXAMPLES.md) |
| **Install the package** | [README.md](README.md) |

## Documentation Structure

### Core Documentation

**[README.md](README.md)** — Start here
- Overview of the package
- Installation instructions
- Core concepts (three registry types)
- Quick start examples
- Feature summary

**[GETTING_STARTED.md](GETTING_STARTED.md)** — Step-by-step tutorials
- Installation & setup
- FunctionalRegistry tutorial
- TypeRegistry tutorial
- SchemeRegistry tutorial
- Remote storage walkthrough
- Batch operations
- Common patterns

**[API_REFERENCE.md](API_REFERENCE.md)** — Complete method reference
- Exception hierarchy
- Base mixins
- Each registry class
- Storage backends
- Engines
- Utility functions

### Advanced Topics

**[ARCHITECTURE.md](ARCHITECTURE.md)** — Design patterns & internals
- Layered mixin architecture
- Design principles
- Storage backend architecture
- Validation cache system
- Factorization system
- Registry inheritance pattern
- Thread safety model
- Extension points
- Performance characteristics

**[STORAGE_BACKENDS.md](STORAGE_BACKENDS.md)** — Local & remote storage
- ThreadSafeLocalStorage (in-process)
- RemoteStorageProxy (network-synced)
- When to use each
- Server setup & deployment
- Performance considerations
- Error handling
- Monitoring & cleanup

**[VALIDATION_ERROR_HANDLING.md](VALIDATION_ERROR_HANDLING.md)** — Errors & validation
- Exception hierarchy
- ValidationError, CoercionError, ConformanceError, InheritanceError, RegistryError
- Validation modes (optional, strict, abstract)
- Error context & debugging
- Validation caching
- Batch error handling
- Common scenarios
- Debugging tips

**[FACTORIZATION_PATTERN.md](FACTORIZATION_PATTERN.md)** — Configuration-driven instantiation
- What is factorization
- Why use it
- Basic usage
- Recursive resolution
- Configuration sources (dict, file, network)
- Validation during factorization
- Advanced patterns
- Error handling
- Performance tuning
- Integration with config files

**[EXAMPLES.md](EXAMPLES.md)** — Practical research examples
- Loss function registry
- Model architecture registry
- Optimizer configuration
- Kalman filter variants
- Experiment configuration from YAML
- Multi-modal data processing
- Batch grid search
- Remote registry for distributed training
- Validation pipeline
- Experiment tracking & logging
- Full workflow integration

## Core Concepts at a Glance

### Three Registry Types

| Type | Use Case | Stores | Example |
|------|----------|--------|---------|
| **FunctionalRegistry** | Algorithm implementations | Functions | Loss functions, metrics, optimizers |
| **TypeRegistry** | Architecture implementations | Classes | Neural network models, data loaders |
| **SchemeRegistry** | Config-driven instantiation | Callable→Pydantic schemes | Experiment configs, training setups |

### Two Storage Backends

| Backend | Location | Concurrency | Latency | Use When |
|---------|----------|-------------|---------|----------|
| **ThreadSafeLocalStorage** | In-process | RLock per registry | <1μs | Single machine, quick iteration |
| **RemoteStorageProxy** | Remote HTTP server | Server-side per-namespace | 5-50ms | Multi-node, distributed workflows |

### Exception Types

| Type | When | Cause |
|------|------|-------|
| **ValidationError** | Generic validation | Invalid input |
| **CoercionError** | Type mismatch | Can't convert type |
| **ConformanceError** | Signature mismatch | Function signature wrong |
| **InheritanceError** | Missing inheritance | Class doesn't inherit from base |
| **RegistryError** | Mapping operation | Key not found / already exists |

## Workflow Patterns

### Pattern 1: Development Workflow

```
1. Define registry class
   ↓
2. Register artifacts with decorators
   ↓
3. Retrieve with get_artifact()
   ↓
4. Use in code
```

See: [GETTING_STARTED.md](GETTING_STARTED.md) "Creating Your First Registry"

### Pattern 2: Configuration-Driven Workflow

```
1. Create YAML/JSON config
   ↓
2. Define SchemeRegistry
   ↓
3. Load config with factorize_from_file()
   ↓
4. Automatic recursive instantiation
```

See: [FACTORIZATION_PATTERN.md](FACTORIZATION_PATTERN.md) "Sources of Configurations"

### Pattern 3: Distributed Workflow

```
1. Start remote server
   ↓
2. Create registries with logic_namespace
   ↓
3. Register from any process
   ↓
4. All processes see same artifacts
```

See: [STORAGE_BACKENDS.md](STORAGE_BACKENDS.md) "RemoteStorageProxy"

### Pattern 4: Validation Workflow

```
1. Create StrictRegistry (strict=True)
   ↓
2. Define protocol/base class
   ↓
3. Register enforces conformance
   ↓
4. Validation errors caught early
```

See: [VALIDATION_ERROR_HANDLING.md](VALIDATION_ERROR_HANDLING.md) "Strict Validation"

## Decision Trees

### Which Registry Type?

```
Is it a function/algorithm?
  ├─ Yes → FunctionalRegistry
  └─ No
     Is it a class/architecture?
       ├─ Yes → TypeRegistry
       └─ No (need config-driven instantiation)
          → SchemeRegistry
```

### Which Storage Backend?

```
Is this distributed research?
  ├─ Yes → RemoteStorageProxy (with remote server)
  └─ No (single machine)
     → ThreadSafeLocalStorage (default)
```

### Strict Mode?

```
Need signature enforcement?
  ├─ Yes → strict=True
  └─ No → strict=False (default)

Need inheritance enforcement?
  ├─ Yes → abstract=True (TypeRegistry only)
  └─ No → abstract=False (default)
```

## Common Tasks Checklist

- [ ] **Install**: `pip install registry`
- [ ] **Create registry**: Subclass `FunctionalRegistry` or `TypeRegistry`
- [ ] **Register artifact**: Use `@Registry.register_artifact` decorator
- [ ] **Retrieve**: Call `Registry.get_artifact("key")`
- [ ] **List all**: Loop `Registry.iter_identifiers()`
- [ ] **Validate**: Catch `ValidationError`
- [ ] **Batch register**: Use `safe_register_batch()`
- [ ] **Use remote**: Set `logic_namespace="my.namespace"`
- [ ] **Load config**: Use `factorize_from_file()`
- [ ] **Debug error**: Print `e.message`, `e.suggestions`, `e.context`

## Performance Tuning

### For Speed

- Use local storage (default) for fast access
- Enable validation caching (automatic)
- Batch operations when possible
- Use `lru_cache` for repeated lookups

### For Distribution

- Use remote storage with `logic_namespace`
- Start server with sufficient resources
- Monitor server statistics at `/stats`
- Consider connection pooling for many clients

### For Memory

- Monitor registry size with `_len_mapping()`
- Clear unused registries with `clear_artifacts()`
- Use remote storage if registries too large

## Error Troubleshooting

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| "Key 'X' is not found" | Artifact not registered | Register it with `register_artifact()` |
| "Key must be hashable" | Dict/list as key | Use string or tuple key |
| "Cannot connect to registry server" | Server not running | Start: `python -m registry` |
| "Timeout" | Network slow | Increase timeout or reduce load |
| "Parameter 'X' missing type annotation" | Strict mode | Add type hints: `x: int` |
| "Not a subclass of Y" | Abstract mode | Inherit from base class |
| "Already exists" | Duplicate key | Use `unregister_identifier()` first |

## Version Information

Current version: **0.2.0**

See [README.md](README.md) for installation and requirements.

## Contributing

Report issues and suggest improvements via the repository.

See the source code comments and docstrings for implementation details.

## Next Steps

1. **Beginners**: Start with [GETTING_STARTED.md](GETTING_STARTED.md)
2. **Intermediate**: Read [ARCHITECTURE.md](ARCHITECTURE.md) for design patterns
3. **Advanced**: Explore [FACTORIZATION_PATTERN.md](FACTORIZATION_PATTERN.md) and [STORAGE_BACKENDS.md](STORAGE_BACKENDS.md)
4. **Research**: See [EXAMPLES.md](EXAMPLES.md) for ML/optimization applications
5. **Reference**: Use [API_REFERENCE.md](API_REFERENCE.md) as needed

## Quick Reference: Common Operations

### Create Registry
```python
from registry import FunctionalRegistry

class MyRegistry(FunctionalRegistry):
    pass
```

### Register Artifact
```python
@MyRegistry.register_artifact
def my_function():
    pass
```

### Get Artifact
```python
artifact = MyRegistry.get_artifact("my_function")
```

### List All
```python
for name in MyRegistry.iter_identifiers():
    print(name)
```

### Check Exists
```python
if MyRegistry.has_identifier("my_function"):
    print("Found")
```

### Unregister
```python
MyRegistry.unregister_identifier("my_function")
```

### Use Remote Storage
```python
class DistributedRegistry(FunctionalRegistry, logic_namespace="my.namespace"):
    pass
```

### Load from Config
```python
obj = MyRegistry.factorize_from_file("MyClass", "config.json")
```

### Handle Errors
```python
from registry import ValidationError

try:
    MyRegistry.register_artifact(invalid)
except ValidationError as e:
    print(e.message)
    print(e.suggestions)
```

### Batch Operation
```python
result = MyRegistry.safe_register_batch(items, skip_invalid=True)
print(f"Success: {len(result['successful'])}")
```

---

**Last updated**: 2024
**Documentation version**: 1.0
