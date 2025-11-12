# Setup Instructions

## Extract to Your Project

```bash
# Download everything from outputs
# Extract to your project's docs/ folder

your-project/
├── docs/
│   ├── autodoc/        ← Sphinx autodoc
│   ├── guides/         ← Markdown guides
│   └── README.md
├── registry/           ← Your source code
└── README.md
```

## Build Sphinx Autodoc API Reference

```bash
cd docs/autodoc
pip install -r requirements.txt
make html
```

Then open: `docs/autodoc/build/html/index.html`

## View Documentation

**Markdown guides:** Open `docs/guides/*.md` directly (or on GitHub)

**Sphinx API:** Build above, then open `docs/autodoc/build/html/index.html`

---

Start reading: `docs/guides/README.md`
