# Storage Backends

## Overview

The registry system supports pluggable storage backends. Each registry can use local in-process storage or remote network-synchronized storage through a simple HTTP interface.

## ThreadSafeLocalStorage

**Location:** In-process memory  
**Concurrency:** Thread-safe with RLock  
**Persistence:** Lost on process exit  
**Latency:** Sub-microsecond  

### When to Use

- Single-machine research workflows
- Development and testing
- When quick iteration is more important than persistence
- Memory-constrained systems (all data fits in RAM)

### Default Behavior

All registries use local storage by default:

```python
from registry import FunctionalRegistry

class Losses(FunctionalRegistry):
    pass  # Uses ThreadSafeLocalStorage automatically

Losses.register_artifact(my_loss)
```

Behind the scenes:
```python
class Losses(FunctionalRegistry):
    _repository: ThreadSafeLocalStorage[str, Callable]
```

### Thread Safety Model

Local storage uses an RLock (reentrant lock) to ensure:

- Multiple threads can read simultaneously
- Writes are exclusive (block readers)
- Same thread can reacquire lock (reentrancy)

```python
# Thread 1
loss_fn = Losses.get_artifact("mse")  # Acquires lock

# Thread 2 blocks here until lock released
Losses.register_artifact(my_loss)

# Can safely hold lock across nested calls
```

### Memory Characteristics

All artifacts stay in memory. For large registries, monitor memory:

```python
import sys

count = Losses._len_mapping()
size_bytes = sum(sys.getsizeof(item) for item in Losses._iter_mapping())
print(f"Registered: {count} artifacts, ~{size_bytes / 1e6:.2f} MB")
```

### Serialization

Artifacts are stored as references (not serialized):

```python
@Losses.register_artifact
def my_loss(y_true, y_pred):
    return ...

# The actual function object is stored
stored_fn = Losses._get_artifact("my_loss")
print(stored_fn is my_loss)  # True
```

## RemoteStorageProxy

**Location:** Remote HTTP server  
**Concurrency:** Server-side per-namespace locks  
**Persistence:** In-memory (configurable by server)  
**Latency:** 5-50ms (network dependent)  

### When to Use

- Multi-node research clusters
- Sharing registries across processes
- Need for central registry server
- Distributed hyperparameter tuning
- Collaborative research teams

### Architecture

```
Process A               Process B               Process C
      │                       │                       │
      └──────────┬────────────┴───────────┬──────────┘
                 │                        │
            HTTP Proxy                HTTP Proxy
                 │                        │
                 └────────────┬───────────┘
                              │
                    ┌─────────────────────┐
                    │ Registry Server     │
                    │ (Flask + locks)     │
                    │                     │
                    │ namespace: storage  │
                    └─────────────────────┘
```

### Starting the Server

#### Basic

```bash
python -m registry.__main__
```

Runs on `localhost:8001` by default.

#### Custom Host/Port

```bash
python -m registry.__main__ --host 0.0.0.0 --port 5000
```

#### With Debug Logging

```bash
python -m registry.__main__ --debug
```

#### Environment Variables

```bash
export REGISTRY_SERVER_HOST=0.0.0.0
export REGISTRY_SERVER_PORT=5000
python -m registry.__main__
```

### Checking Server Health

```python
import requests

# Health check
response = requests.get("http://localhost:8001/health")
print(response.json())
# -> {'status': 'healthy', 'service': 'registry-server'}

# Statistics
response = requests.get("http://localhost:8001/stats")
stats = response.json()
print(f"Namespaces: {stats['namespaces']}")
print(f"Total entries: {stats['total_entries']}")
print(f"Uptime: {stats['created_at']}")
```

### Creating Remote Registries

```python
from registry import FunctionalRegistry

class DistributedLosses(FunctionalRegistry, logic_namespace="research.losses"):
    """Registry synced to remote server."""
    pass

class DistributedModels(FunctionalRegistry, logic_namespace="research.models"):
    """Different namespace for models."""
    pass
```

The `logic_namespace` parameter configures remote storage:

- Dot-delimited (e.g., `"research.losses"`)
- Must be alphanumeric + dots
- Different registries can share same namespace
- Recommended: use separate namespaces for different artifact types

### Registering Artifacts

```python
# Register from any process
@DistributedLosses.register_artifact
def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Other processes can retrieve it
from_another_process = DistributedLosses.get_artifact("mse")
```

### Serialization Over Network

Artifacts are serialized for transmission:

1. **JSON** (default, if possible)
   - Primitives: int, float, str, bool, None
   - Collections: dict, list
   - No function/object state

2. **Pickle** (fallback)
   - Arbitrary Python objects
   - Serializes function/class references
   - Requires same Python version

```python
# JSON-serializable config
class Config(SchemeRegistry, logic_namespace="research.config"):
    pass

config = {"learning_rate": 0.001, "momentum": 0.9}
# Sends over wire as: {"lr": 0.001, "momentum": 0.9}

# Non-serializable object (pickled)
@DistributedModels.register_artifact
class MyModel:
    """Complex model with internal state."""
    pass

# Sent via pickle + base64 encoding
```

### Namespace Organization

Namespaces segment the server's storage:

```
research.losses          → Loss function registry
research.models          → Model class registry
research.optim           → Optimizer registry
experiments.2024.exp01   → Specific experiment configs
```

Benefits:

- Clear logical separation
- Different access patterns/refresh rates
- Easy cleanup (clear entire namespace)
- Multi-team collaboration

```python
# Team A
class TeamALosses(FunctionalRegistry, logic_namespace="teamA.losses"):
    pass

# Team B
class TeamBLosses(FunctionalRegistry, logic_namespace="teamB.losses"):
    pass

# Both connect to same server, separate storage
```

### Performance Considerations

#### Request Batching

Minimize round-trips for bulk operations:

```python
# ❌ Inefficient (N requests)
for name, func in large_dict.items():
    Losses.register_artifact(func)

# ✅ Better (1 request)
Losses.safe_register_batch(large_dict, skip_invalid=True)
```

#### Server-Side Caching

The server maintains statistics but not application-level caches. Implement client-side caching:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_loss(name: str):
    return DistributedLosses.get_artifact(name)

# Repeated calls hit local cache
loss_fn = get_loss("mse")
loss_fn = get_loss("mse")  # No network call
```

#### Connection Pooling

Use requests sessions for efficiency:

```python
import requests
from registry import RemoteStorageProxy

# Create session with connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20
)
session.mount("http://", adapter)

# Registries automatically pool connections
```

### Error Handling

Network issues raise `ValidationError`:

```python
from registry import ValidationError

class DistributedLosses(FunctionalRegistry, logic_namespace="research.losses"):
    pass

try:
    loss_fn = DistributedLosses.get_artifact("mse")
except ValidationError as e:
    if "Cannot connect" in e.message:
        print("Server is down")
    elif "timeout" in e.message:
        print("Network is slow")
```

Common errors:

| Error | Cause | Fix |
|-------|-------|-----|
| Cannot connect | Server not running | Start server: `python -m registry` |
| Timeout | Network slow or server overloaded | Increase timeout or reduce load |
| Key not found | Artifact not registered | Register it first |
| Invalid namespace | Namespace format wrong | Use format: `my.namespace` |

### Timeout Configuration

Default timeout is 5 seconds. Adjust via:

```python
class SlowNetworkRegistry(FunctionalRegistry, logic_namespace="my.namespace"):
    pass

# Timeout is controlled via environment variable
import os
os.environ["REGISTRY_SERVER_TIMEOUT"] = "30.0"  # 30 seconds

# Or pass to RemoteStorageProxy directly
from registry.storage import RemoteStorageProxy

storage = RemoteStorageProxy(
    namespace="my.namespace",
    timeout=30.0
)
```

## Switching Between Backends

### From Local to Remote

```python
# Local development
class DevLosses(FunctionalRegistry):
    pass

# Production with remote storage
class ProdLosses(FunctionalRegistry, logic_namespace="prod.losses"):
    pass

# Same API, different backend
loss_fn = ProdLosses.get_artifact("mse")
```

### Hybrid Approach

Use local for frequent access, remote for shared state:

```python
class LocalCache(FunctionalRegistry):
    """Fast local access."""
    pass

class RemoteSource(FunctionalRegistry, logic_namespace="shared"):
    """Central authority."""
    pass

# Sync from remote to local
for name in RemoteSource.iter_identifiers():
    artifact = RemoteSource.get_artifact(name)
    LocalCache.register_artifact(artifact)
```

## Network Protocol Details

The remote storage uses HTTP with JSON/pickle encoding.

### Request Format

```http
POST /registry/my.namespace/set HTTP/1.1
Content-Type: application/json

{
  "key": {
    "type": "json",
    "data": "\"my_key\"",
    "encoding": "json"
  },
  "value": {
    "type": "json",
    "data": "{\"result\": 42}",
    "encoding": "json"
  }
}
```

### Response Format

```http
HTTP/1.1 200 OK
Content-Type: application/json

{"status": "success"}
```

### Serialization Details

**JSON:**
```python
{
  "type": "json",
  "data": "{...json string...}",
  "encoding": "json"
}
```

**Pickle (binary data):**
```python
{
  "type": "pickle",
  "data": "gASVCQAAAAAAAA...",  # base64 encoded
  "encoding": "base64"
}
```

## Deployment Recommendations

### Development

Use local storage:
```python
class DevRegistry(FunctionalRegistry):
    pass  # No logic_namespace
```

### Staging/Production

Use remote storage with persistent backend:

```python
class ProdRegistry(FunctionalRegistry, logic_namespace="prod.models"):
    pass  # Connects to central server
```

Server configuration:
```bash
python -m registry.__main__ \
  --host 0.0.0.0 \
  --port 8001
```

### High Availability

Run multiple server instances behind a load balancer:

```
Client 1 ──┐
Client 2 ──┤───→ Load Balancer ──→ Server 1 (same memory)
Client 3 ──┤                    ──→ Server 2 (different memory)
           ──→ Load Balancer ──→ Server 3 (different memory)
```

Note: Current implementation has per-server memory. For true HA, add persistent backend (Redis, PostgreSQL).

## Monitoring

### Server Statistics

```python
import requests
import time

def monitor_registry(url="http://localhost:8001"):
    while True:
        response = requests.get(f"{url}/stats")
        stats = response.json()
        
        print(f"Namespaces: {stats['namespaces']}")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Requests: {stats['total_requests']}")
        
        time.sleep(10)

monitor_registry()
```

### Client-Side Metrics

```python
from registry import FunctionalRegistry

class MonitoredRegistry(FunctionalRegistry, logic_namespace="my.namespace"):
    pass

import time

start = time.time()
MonitoredRegistry.register_artifact(my_function)
elapsed = time.time() - start

print(f"Registration took {elapsed*1000:.2f}ms")
```

## Cleanup

### Clear a Namespace

```python
class MyRegistry(FunctionalRegistry, logic_namespace="my.namespace"):
    pass

MyRegistry.clear_artifacts()  # Removes all from namespace
```

### Clear Remote Server

```bash
# Via HTTP API
curl -X DELETE http://localhost:8001/registry/my.namespace/clear
```

## Next Steps

- See [VALIDATION_ERROR_HANDLING.md](VALIDATION_ERROR_HANDLING.md) for error semantics
- See [FACTORIZATION_PATTERN.md](FACTORIZATION_PATTERN.md) for advanced usage
