# Changelog

All notable changes to `registry-pattern` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Recursive factory `registry.build(cfg, ...)` -- envelope-shaped or
  explicit class call, validates kwargs against the target's Pydantic
  schema, recurses on nested envelopes, resolves `$ref` strings against
  ctx + sibling scope, invokes the target, runs post hooks, validates
  the resulting meta dict against a `meta_schema`, attaches `__meta__`
  to the result. One source-agnostic entry point.
- Three-tier API: `resolve(name, repo=...)` (lookup only), `validate(
  target, data, validator=...)` (lookup + schema check, no instantiate),
  `build(...)` (full pipeline). Same dotted-`repo` matching across all
  three.
- `registry.serialize(instance, serializator=...)` -- symmetric output
  side; mediums `python` / `yaml` / `json`.
- Hierarchical repo tree via `class X(TypeRegistry[T], repo="my.path"):
  ...`. Sub-registries inherit `post_init` / `pre_call` via Python
  inheritance (cooperative `super()`). `resolve` does prefix-or-exact
  match.
- Annotated marker system: `Annotated[T, SameDeviceAs("device"),
  Checksum("hash"), ...]` -- validate markers fire pre-invocation,
  compute markers populate the envelope's meta dict. Ships
  `SameDeviceAs`, `BoundTo`, `InputShapeMatches`, `Checksum`,
  `NumParams`, `Device`, `EffectiveLr`.
- Implicit `data_schema` (from constructor signature, rewrites arbitrary
  types to `Buildable[T]`) and `meta_schema` (from compute markers'
  return annotations) -- both overridable by setting class attributes
  on the registry.
- Meters (write to envelope meta): `FactoryMeter` base + `LifetimeMeter`,
  `CPUMeter`, `MemoryMeter`, `IOMeter`, `NetworkMeter`, `HeapMeter`,
  `RecursionMeter`. All stdlib (`resource`, `tracemalloc`, `/proc/self/
  io`, `/proc/self/net/dev`).
- Reporters (route events externally): `FactoryReporter` base +
  `JournalReporter` (syslog -> journald), `HTTPDashboardReporter` (JSON
  on localhost:port), `OpenTelemetryReporter` (spans + Histogram
  metrics; needs `[otel]` extra).
- Pipeline contract: meters fire BEFORE reporters at every stage so
  reporters always see populated meta.
- stdlib logging at INFO across `registry.factory`, `registry.typ_registry`,
  `registry.fnc_registry`, `registry.mixin.validator`. One `logging.
  basicConfig(level=INFO)` shows registry creation, artifact
  registration, build start/done events.
- `ValidatorRegistry` (internal): pluggable medium decoders --
  `python`, `yaml`, `json`, `argparse`, `pydantic`, `jsonargparse`,
  `noop`. `validator="yaml"` decodes a YAML string before validation.
- Tree-shaped sub-registrar pattern: same artifact registered in
  multiple repos with different post_init -- e.g. `ResNet50` in both
  `models.cnn` and `models.pretrained` with the latter enforcing a
  weight-hash check.

### Changed
- `is_build_cfg(value)` now requires BOTH `type` (str) AND `data` (dict)
  keys. Previously only `type` was checked, allowing collisions with
  constructor params named `type`.
- `TypeRegistry[T](abstract=True)` now checks `issubclass(artifact, T)`
  against the generic parameter, not against the registry class itself.
- `ContainerMixin.build_cfg` strict-by-default: schema validation
  failures raise instead of silently falling back to raw data; unknown
  kwarg keys raise instead of going to `meta._unused_data`. Opt out
  per-registry via `_strict_params = False` / `_strict_unused = False`
  ClassVars.
- Python floor bumped 3.8 -> 3.10 to match what CI tests.

### Removed
- `RemoteStorageProxy` + the Flask-based registry server CLI
  (`python -m registry server`). The pickle-over-HTTP RCE path is gone.
  Use callpyback's RPC stack for remote dispatch if needed.
- `--server` flags on `build`/`run` subcommands. The CLI surface is
  now `info` / `build` / `run`.
- `SchemeRegistry` (`registry/sch_registry.py`) -- superseded by
  `ValidatorRegistry` + auto-derived config schema.
- `ConfigRegistry` and `ObjectRegistry` (`registry/extra/`) -- unused.
- `ContainerMixin.configure_repos` and `ContainerMixin.get_repo`.
  Declare registries with `class X(TypeRegistry[T], repo="path"): ...`
  and use `registry.build` / `registry.resolve` instead. If you really
  need the legacy `build_cfg` flow, set `ContainerMixin._repos = {...}`
  directly.
- `logic_namespace` kwarg on `TypeRegistry` / `FunctionalRegistry`
  `__init_subclass__`. Use `repo=` instead.
- `[server]` and `[http]` optional dependency extras. `flask` and
  `requests` are no longer referenced by any code path.
- `ConfigFileEngine` and `SocketEngine` are no longer re-exported from
  `registry`; import from `registry.engines`.

### Fixed
- `_extract_params_model` resolves string annotations via
  `typing.get_type_hints` so `from __future__ import annotations` plus
  `List[Foo]` no longer raises "class not fully defined".
- `_is_json_native(tuple[float | Tensor, ...])` correctly returns
  False; previously the outer `tuple` origin made it appear JSON-safe
  and Pydantic later choked on `torch.Tensor`.
- `meta` written by hooks / meters / `pre_call` now survives the
  `meta_schema` `model_dump()` round-trip (derived schema gets
  `extra="allow"`).
- `cfg["meta"]` mutation propagates when the caller passes a dict
  instead of a `BuildCfg` instance.

### Internal
- `ValidatorRegistry` and `SerializerRegistry` are pipeline-internal
  and not re-exported from the package root. Import from
  `registry.validators` or `registry.factory` if you need to register
  custom mediums.
- 154 tests pass across the suite (Python 3.10-3.13, pydantic 2.x).
