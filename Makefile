.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard ./*.ipynb)

all: rs3 docs

rs3: $(SRC)
	nbdev-export
	touch rs3

sync:
	nbdev-update

docs_serve: docs
	nbdev-preview

docs: $(SRC)
	nbdev-docs
	touch docs

test:
	nbdev-test
	pytest

# Regenerate the library, README and docs, and run the tests -- run before pushing.
prepare:
	nbdev-prepare

release: pypi conda_release
	nbdev-bump-version

conda_release:
	nbdev-conda

# Prefer releasing by pushing a version tag, which runs .github/workflows/release.yaml:
# that checks the tag against rs3.__version__, verifies the model artifacts are in
# the wheel, and publishes via PyPI Trusted Publishing. This target is the manual
# fallback and performs none of those checks.
pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python -m build

clean:
	rm -rf dist
