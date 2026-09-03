# Model artifacts

rs3 ships three trained models under `rs3/`, each as a LightGBM booster in
LightGBM's own text format plus a small JSON sidecar:

| artifact | features | trees | purpose |
| --- | --- | --- | --- |
| `RuleSet3.{txt,json}` | 632 | 2279 | sequence-only activity score (`rs3.seq`) |
| `target_model.{txt,json}` | 47 | 1702 | full target score (`rs3.predicttarg`) |
| `target_lite_model.{txt,json}` | 29 | 1779 | lite target score, no domain or conservation data |

The sidecar records `feature_names`, the `medians` used for imputation (target
models only), tree and feature counts, and the versions that produced the file.

## Why not pickles

Until v0.1.0 these shipped as `joblib` pickles, and the two target models were
`sklearn.pipeline.Pipeline` objects wrapping a `SimpleImputer(strategy='median')`
and an `LGBMRegressor`. That bound the package to its 2021 dependency stack:

* Those pickles were written by scikit-learn `1.0.dev0`. `SimpleImputer.transform`
  gained private attributes in scikit-learn 1.2, so unpickling and predicting on
  anything newer raises
  `AttributeError: 'SimpleImputer' object has no attribute '_fit_dtype'`.
  This is what `scikit-learn<=1.0.2` was pinning.
* LightGBM's scikit-learn wrapper needs scikit-learn installed merely to call
  `.predict()` — in 4.x it calls `super().get_params()`, which only exists when
  `sklearn.base.BaseEstimator` is available. So even a bare `LGBMRegressor`
  pickle could not predict without scikit-learn.

LightGBM's text format is designed to be read across versions, and the imputer
was a plain median fill whose values were already stored in the fitted object.
Exporting both removed scikit-learn from the dependency set entirely and let the
`numpy<=1.26.4` and `lightgbm<=3.3.5` ceilings come off.

The conversion is exact, not approximate. A booster reloaded from text and fed a
median-filled frame reproduces the original pipeline's predictions bit for bit
(`np.array_equal`, max difference `0.0`) on all three models.

## Feature order does not come from the model file

**This is the trap.** Neither model file can validate its own inputs:

* The **target boosters report `Column_0 .. Column_46`**. `SimpleImputer.transform`
  returns a bare array, so the column names were gone before `LGBMRegressor.fit`
  ever saw them. Their real order is the `feature_list` built in
  `merge_feature_dfs` (`rs3/targetfeat.py`) — 2 position features, then 16 domain
  and 2 conservation features for the full model, then 27 amino acid features.
* The **RuleSet3 booster reports LightGBM-sanitised names** — `GC_content` where
  the DataFrame column is `GC content`, differing in 8 of 632 positions. Close
  enough to look usable, and wrong.

So the sidecars carry the names as `featurize_context` and `merge_feature_dfs`
actually produce them, and `BoosterModel.predict` selects columns by name before
predicting. If you ever regenerate the sidecars, take the target models' feature
order from `merge_feature_dfs` and **not** from `booster.feature_name()`.

## Secondary structure features are deliberately out of date

`get_aa_secondary_structure` (`rs3/targetfeat.py`) computes the `Helix`, `Turn`
and `Sheet` fractions itself, using the amino acid groups Biopython used *before*
version 1.82:

    Helix: VIYFWL      Turn: NPGS      Sheet: EMAL

Biopython 1.82 changed `ProteinAnalysis.secondary_structure_fraction()` in two
ways at once: it corrected a bug where the method returned `(Sheet, Turn, Helix)`
while documenting `(Helix, Turn, Sheet)`, and it changed the groups themselves
(helix to `EMALK`, turn to `NPGSD`, sheet to `VIYFWLT`).

The target models were trained on the older definition, so reproducing it is what
keeps published scores reproducible. Delegating to Biopython's current
implementation shifts target scores by a mean of 0.109 and a maximum of 0.445 —
against a score standard deviation of 0.152, on 361 of 400 test guides, with no
error raised.

Note what this means: at the biological level those features are misnamed. The
column called `Helix` carries sheet propensity and vice versa. Fixing that
properly means retraining the target models, which is a modelling decision, not a
packaging one.

## Regenerating and verifying

Verification needs only the shipped artifacts — no scikit-learn, no pickles — so
run it anywhere, particularly on a new LightGBM major version:

```bash
python tools/convert_models.py --verify
```

It replays `tests/golden/`, the predictions rs3 0.0.18 produced on its original
2021 stack, and reports any model that does not reproduce them exactly.

Regenerating from the original pickles is not normally necessary. It needs the
legacy environment described in `requirements-legacy.txt` and the pickles
themselves, which were removed from the working tree once the exports were
verified but remain in git history:

```bash
git show 82411ac:rs3/RuleSet3.pkl          > rs3/RuleSet3.pkl
git show 82411ac:rs3/target_model.pkl      > rs3/target_model.pkl
git show 82411ac:rs3/target_lite_model.pkl > rs3/target_lite_model.pkl
python tools/convert_models.py
```

`tools/capture_baseline.py` regenerates `tests/golden/` itself. Only do that to
establish a deliberately new baseline — never to make a failing comparison pass.
