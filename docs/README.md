# Registry Package Documentation

Complete documentation: Sphinx autodoc + markdown guides.

## Structure

```
docs/
├── autodoc/              Sphinx auto-generated API reference
├── guides/               Manual guides and tutorials
└── README.md             This file
```

## Quick Links

### Getting Started
- **[Getting Started Guide](guides/GETTING_STARTED.md)** — Step-by-step tutorials for all registry types

### Core Documentation
- **[Architecture & Design](guides/ARCHITECTURE.md)** — Design patterns, internals, mixin architecture
- **[API Reference](autodoc/build/html/index.html)** — Complete auto-generated API (build with: `make html` in autodoc/)
- **[Storage Backends](guides/STORAGE_BACKENDS.md)** — Local vs remote storage, deployment

### Advanced Topics
- **[Factorization Pattern](guides/FACTORIZATION_PATTERN.md)** — Configuration-driven instantiation
- **[Validation & Error Handling](guides/VALIDATION_ERROR_HANDLING.md)** — Error semantics, debugging
- **[Examples](guides/EXAMPLES.md)** — 10+ research examples (ML, optimization, signals)

### Reference
- **[Index & Quick Reference](guides/INDEX.md)** — Navigation guide, decision trees, quick lookup

---

## Build Sphinx Autodoc

To generate API reference HTML:

```bash
cd autodoc
pip install -r requirements.txt
make html
```

Generated documentation: `autodoc/build/html/index.html`

---

## Which Documentation to Use?

| Question | Go to |
|----------|-------|
| "How do I get started?" | GETTING_STARTED.md |
| "How does it work?" | ARCHITECTURE.md |
| "What method does X do?" | autodoc/build/html/ (API reference) |
| "How do I debug this error?" | VALIDATION_ERROR_HANDLING.md |
| "Show me a real example" | EXAMPLES.md |
| "What are all the options?" | INDEX.md |

---

**Start with:** [GETTING_STARTED.md](guides/GETTING_STARTED.md)
