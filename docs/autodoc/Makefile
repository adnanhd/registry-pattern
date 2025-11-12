.PHONY: help clean html

help:
	@echo "Build Sphinx documentation:"
	@echo "  make html    - Build HTML documentation"
	@echo "  make clean   - Clean build directory"

clean:
	rm -rf build

html:
	sphinx-build -b html source build/html

.DEFAULT_GOAL := help
