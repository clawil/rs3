# Changelog

## 0.1.0 (unreleased)

Modernizes the dependency stack. **Predictions are unchanged**: every score this
release produces is bit-identical to 0.0.18 on the same inputs, verified against
recorded outputs from 0.0.18 running on its original 2021 dependency stack
(`tests/golden/`, `atol=0` for the model layer).

A minor rather than patch bump: the minimum Python version moves and
scikit-learn is no longer installed.

### Breaking

* Requires Python 3.10 or newer (was 3.7).
* **scikit-learn is no longer a dependency.** It was only ever needed to
  unpickle the models.
* The models ship as LightGBM text plus a JSON sidecar instead of pickles. The
  `.pkl` files are gone from the distribution.
* `load_seq_model()` and `load_target_model()` return a `BoosterModel` rather
  than an `LGBMRegressor` / `sklearn.pipeline.Pipeline`. It exposes
  `.predict(DataFrame)`, so calling code is unaffected, but code reaching into
  the estimator internals is not.

### Fixed

* **rs3 could not be installed from source at all.** `setup.py` imported
  `pkg_resources`, removed in setuptools 84, and with no `[build-system]` table
  the build frontend always provisioned an isolated environment with the newest
  setuptools — so the failure did not depend on the user's local setuptools.
* **Biopython 1.82+ silently changed the target scores.**
  `secondary_structure_fraction()` both fixed a `(Sheet, Turn, Helix)` /
  `(Helix, Turn, Sheet)` ordering bug and changed its amino acid groups. Since
  `targetfeat` consumed the tuple positionally, upgrading Biopython shifted
  target scores by a mean of 0.109 and a maximum of 0.445 — against a score
  standard deviation of 0.152, on 361 of 400 test guides, with no error. rs3 now
  computes those features itself, using the definition the models were trained
  on. See [MODELS.md](MODELS.md).
* `rs3.targetdata` imported `scipy.stats` at module scope and never used it.
  scipy was declared only in dev requirements, making this an undeclared runtime
  import.
* The UCSC conservation API is called over HTTPS rather than HTTP.
* Fixed an invalid escape sequence in `featurize_aa_seqs`, which Python has been
  escalating toward a `SyntaxError`.
* Install instructions: `README.md` had drifted from the `index.ipynb` it is
  generated from, and both were stale. They now describe the `libomp` error
  macOS users actually hit.

### Changed

* Version ceilings removed: `numpy<=1.26.4`, `lightgbm<=3.3.5` and
  `jupyter-client<=6.1.12` are gone, and remaining constraints are floors only.
  Verified on Python 3.10 (numpy 2.2, pandas 2.3) and 3.13 (numpy 2.5,
  pandas 3.0).
* Packaging moved from `setup.py` + `settings.ini` to `pyproject.toml`.
* nbdev v1 → 3; documentation moved from Jekyll to Quarto and is now built
  rather than committed as HTML.
* CI runs a Python 3.10–3.13 matrix and executes 40 notebook assertions per leg,
  plus the golden regression. Previously it tested only `index.ipynb`, which
  contains no assertions, on Python 3.9. Live-API tests run in a separate,
  non-blocking scheduled job because `rest.ensembl.org` returns intermittent
  503s.
* Devcontainer rebuilt on the standard Python image; `docker-compose.yml`
  removed.

### Added

* A golden-output regression suite (`tests/`, `tools/capture_baseline.py`)
  pinning predictions to what 0.0.18 produced.
* `tools/convert_models.py --verify`, which re-checks the shipped models against
  that baseline in any environment, without scikit-learn or the pickles.
* [MODELS.md](MODELS.md), documenting the artifact format and the two things
  that would silently produce wrong predictions if "corrected".

### Needs maintainer attention

* **GitHub Pages must be repointed** from the `docs/` folder to the `gh-pages`
  branch, or the published documentation will stop updating. See
  `.github/workflows/deploy.yaml`.
* **One assertion in `04_predict.ipynb` is flagged `#| notest` pending review.**
  It expects the Hsu2013 sequence score to correlate with GeCKOv2 activity more
  strongly than sequence + target does; measured on the committed fixtures the
  order is reversed by 0.006. This is not a regression — the same values come
  out of the original 2021 stack — but it is an unreviewed discrepancy between
  the notebook's claim and the committed test data.
* `sglearn`, a hard runtime dependency, was last released in 2021. It works with
  the current stack, and its own constraints are loose, but it is unmaintained.
