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

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python -m build

clean:
	rm -rf dist
