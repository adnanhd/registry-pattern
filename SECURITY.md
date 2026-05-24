# Security policy

## Reporting a vulnerability

Please report security issues by email to **adnanharundogan@gmail.com**
with the subject prefix `[registry-pattern security]`. Do NOT open a
public GitHub issue.

You should expect an acknowledgement within 7 days. A fix and a
public disclosure date will be coordinated once the issue is
confirmed.

## Threat model

`registry-pattern` is an in-process dependency-injection / factory
container. It does not open network sockets, does not exec
subprocesses, and does not deserialise untrusted binary formats. The
threat boundary is "Python code in your process can register and
build objects."

### Surfaces worth knowing about

- **`build(cfg)`** instantiates whichever class the registry resolves
  `cfg["type"]` to, with whichever kwargs are in `cfg["data"]`. If
  config comes from a user-controlled file (YAML / JSON / TOML), an
  attacker who can edit the config can trigger any registered class's
  `__init__`. Treat config files like Python scripts.
- **`ConfigFileEngine`** loads YAML / JSON / TOML config from disk.
  Built on `pyyaml.safe_load` for YAML; no remote URL fetching.
- **`SyslogReporter`** opens a connection to the local syslog daemon.
  Linux/macOS only; no network exposure.
- **`HTTPDashboardReporter`** publishes telemetry to an HTTP endpoint
  you configure. Outgoing only.

### What does NOT execute code

- Schema derivation, validation, and type-guard checks
  (`Buildable[T]`, `serialize()`, marker validation).
- Reading the resolve cache.
- Registering a class via `@register_artifact` (decorator only --
  does not invoke the class).

## Supported versions

Only the latest minor release receives security patches. Pre-1.0
releases may carry breaking changes between minor versions.
