# Contributing to registry-pattern

Thanks for considering a contribution. `registry-pattern` is beta
software; the public API is stable but the internals still move.
Issues, bug reports, and small PRs are all welcome.

## Quick start

```bash
git clone https://github.com/adnanhd/registry-pattern
cd registry-pattern
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs,torch,otel,yaml]"
pytest tests/ -q
```

If `pytest` is green, you have a working dev install.

## Reporting bugs

Open a GitHub issue with:

1. A minimal repro -- the smallest registry + `build()` call that
   shows the problem.
2. What you expected.
3. What you got (error message + traceback).
4. Python version + installed `registry-pattern` version.

## Proposing changes

Small fixes (typos, docs, one-file refactors): open a PR directly.

Larger changes (new public API, schema changes, behaviour changes
that would break existing consumers): open an issue first. Pre-1.0
we still want a sketch before code.

### PR checklist

- [ ] `pytest tests/` passes (full suite).
- [ ] If you touched the public API surface (top-level
  `registry/__init__.py` exports, factory pipeline, schema
  derivation), add or update a test.
- [ ] Docstrings on new public functions / classes -- include a
  one-line summary and at least one usage example for non-trivial
  helpers.
- [ ] Commit subjects: `category: subject` (`feat`, `fix`, `chore`,
  `refactor`, `style`, `docs`, `test`, `bench`). ~72 char cap.
- [ ] Update `CHANGELOG.md` under `[Unreleased]` if the change is
  user-visible.

## Code style

Conventional Python. `black .` for formatting, `ruff check registry/
tests/` for lint, `pyright` for type checks. CI runs all three plus
`flake8` (line-length 88, picks up `.flake8`).

## Tests

Tests live in `tests/`. Run the whole suite at logical checkpoints;
target specific modules during iteration:

```bash
pytest tests/test_factory.py -q
pytest tests/ -k schema -q
```

## Releases

Releases are cut by the maintainer:

1. Bump `registry/_version.py` and `pyproject.toml` to the new version.
2. Update `CHANGELOG.md`.
3. Tag the commit (`git tag v0.6.0 && git push --tags`).
4. The `release.yml` workflow builds the wheel + sdist and publishes
   to PyPI via trusted publishing.

## License

By contributing you agree that your contribution will be licensed
under the project's MIT license.
