#!/usr/bin/env python
"""Convert the pickled rs3 models to LightGBM's text format plus a JSON sidecar.

Why: the shipped .pkl files bind rs3 to a 2021 dependency stack. Two of them are
`sklearn.pipeline.Pipeline` objects pickled by scikit-learn 1.0.dev0, and
`SimpleImputer.transform` gained private attributes in 1.2, so they raise
`AttributeError: 'SimpleImputer' object has no attribute '_fit_dtype'` on any
modern scikit-learn. Separately, lightgbm's sklearn wrapper needs scikit-learn
installed just to call `.predict()`. Pickles are why rs3 is pinned to
`scikit-learn<=1.0.2`, `lightgbm<=3.3.5` and `numpy<=1.26.4`.

LightGBM's own text format is designed to be read across versions, and the
imputer is a plain median fill whose values are already stored in the fitted
object. So each model becomes:

    <name>.txt     the LightGBM booster, in LightGBM's text format
    <name>.json    feature names, and the median vector for the target models

After this, rs3 needs no scikit-learn at all and nothing is version-coupled.

CONVERT (needs the legacy env from requirements-legacy.txt, which still has the
scikit-learn that wrote the pickles -- run from the repo root):

    ~/miniforge3/envs/rs3-legacy/bin/python tools/convert_models.py

The source .pkl files were removed from the working tree once the exported
artifacts were verified; they remain in git history. To re-run the conversion,
restore them first:

    git show 82411ac:rs3/RuleSet3.pkl          > rs3/RuleSet3.pkl
    git show 82411ac:rs3/target_model.pkl      > rs3/target_model.pkl
    git show 82411ac:rs3/target_lite_model.pkl > rs3/target_lite_model.pkl

Re-conversion should not normally be necessary: `--verify` is the standing check,
and it needs only the exported artifacts.

VERIFY (needs neither scikit-learn nor the pickles -- run it in every environment
you care about, especially one with lightgbm 4.x, to prove the text format reads
forward):

    python tools/convert_models.py --verify

Verification replays tests/golden/, so it checks the artifacts against what
rs3 0.0.18 actually predicted rather than against itself.
"""

import argparse
import datetime
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, 'tools')
for _p in (REPO, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PKG = os.path.join(REPO, 'rs3')
GOLDEN = os.path.join(REPO, 'tests', 'golden')

# (pickle, artifact stem, golden feature matrix, golden score file, score column)
MODELS = [
    ('RuleSet3.pkl', 'RuleSet3',
     'features_seq.csv', 'seq_predictions.csv', 'prediction'),
    ('target_model.pkl', 'target_model',
     'features_target_full.csv', 'target_full.csv', 'Target Score'),
    ('target_lite_model.pkl', 'target_lite_model',
     'features_target_lite.csv', 'target_lite.csv', 'Target Score Lite'),
]


def _shim(imputer):
    """Let a scikit-learn 1.0 SimpleImputer transform under scikit-learn >=1.2.

    Only reached if conversion is run outside the legacy env. The attributes are
    private and this is not shipped -- it exists so `--verify` can be compared
    against the original pipeline on a modern stack if ever needed.
    """
    imputer._fit_dtype = np.dtype('float64')
    imputer._fill_dtype = np.dtype('float64')
    if not hasattr(imputer, 'keep_empty_features'):
        imputer.keep_empty_features = False
    return imputer


def _seq_feature_names(booster):
    """The column order RuleSet3 expects, as `featurize_context` actually names them.

    Not `booster.feature_name()`: LightGBM sanitises names when it stores them, so
    the booster reports 'GC_content' where the DataFrame column is 'GC content' (8
    of the 632 columns differ this way). rs3 has to select columns from the
    featurized DataFrame, so the sidecar needs the DataFrame's spelling. The
    positional correspondence between the two is asserted below, which is what
    actually pins the ordering.
    """
    from rs3.seq import featurize_context

    probe = featurize_context(['GACGAAAGCGACGTCGGGCCCAGCGGCTGC'], sequence_tracr='Hsu2013')
    names = list(probe.columns)
    assert [c.replace(' ', '_') for c in names] == list(booster.feature_name()), \
        'featurize_context column order no longer matches the RuleSet3 booster'
    return names


def _target_feature_names(lite):
    """The feature order the target models are positionally bound to.

    Taken from `merge_feature_dfs` (rs3/targetfeat.py), which is the only place
    this order is defined. It must NOT be read off the booster: the imputer
    stripped column names before `LGBMRegressor.fit` saw them, so the target
    boosters report `Column_0..Column_N` and introspecting them would silently
    hand you the wrong thing.
    """
    from capture_baseline import _target_feature_matrix

    design_df = pd.read_csv(os.path.join(REPO, 'test_data', 'sgrna-designs.txt'), sep='\t')
    _, feature_cols = _target_feature_matrix(design_df, lite=lite)
    return feature_cols


def convert():
    import joblib
    import lightgbm as lgb
    import sklearn

    print('converting with scikit-learn {} / lightgbm {}'.format(
        sklearn.__version__, lgb.__version__))

    for pkl, stem, feat_csv, _score_csv, _score_col in MODELS:
        src = os.path.join(PKG, pkl)
        model = joblib.load(src)

        if hasattr(model, 'steps'):
            imputer, regressor = _shim(model.steps[0][1]), model.steps[1][1]
            assert imputer.strategy == 'median', \
                'expected a median imputer, got {!r}'.format(imputer.strategy)
            medians = imputer.statistics_.tolist()
            feature_names = _target_feature_names(lite='lite' in stem)
            assert len(feature_names) == len(medians), \
                '{}: {} feature names but {} imputer medians'.format(
                    stem, len(feature_names), len(medians))
            assert not np.isnan(imputer.statistics_).any(), \
                '{}: imputer medians contain NaN'.format(stem)
        else:
            # RuleSet3 was fit directly on the sglearn DataFrame, so its booster
            # kept (a sanitised form of) the real column names.
            imputer, regressor = None, model
            medians = None
            feature_names = _seq_feature_names(model.booster_)

        booster = regressor.booster_
        assert booster.num_feature() == len(feature_names), \
            '{}: booster has {} features, sidecar has {}'.format(
                stem, booster.num_feature(), len(feature_names))

        txt_path = os.path.join(PKG, stem + '.txt')
        booster.save_model(txt_path)

        sidecar = {
            'feature_names': feature_names,
            'medians': medians,
            'n_features': len(feature_names),
            'n_trees': booster.num_trees(),
            'source_pkl': pkl,
            'source_versions': {
                'scikit-learn': sklearn.__version__,
                'lightgbm': lgb.__version__,
                'numpy': np.__version__,
                'python': '.'.join(str(v) for v in sys.version_info[:3]),
            },
            'converted': datetime.date.today().isoformat(),
            'booster_feature_names': list(booster.feature_name()),
            'feature_name_source': ('rs3.targetfeat.merge_feature_dfs'
                                    if medians is not None else 'rs3.seq.featurize_context'),
            'imputation': ('median fill from `medians`, in `feature_names` order'
                           if medians is not None else None),
        }
        json_path = os.path.join(PKG, stem + '.json')
        with open(json_path, 'w') as fh:
            json.dump(sidecar, fh, indent=2)
            fh.write('\n')

        # Prove the exported artifacts reproduce the pickle before trusting them.
        feat = pd.read_csv(os.path.join(GOLDEN, feat_csv), float_precision='round_trip')
        X = feat[feature_names]
        reloaded = lgb.Booster(model_file=txt_path)
        if medians is None:
            reference = regressor.predict(X)
            candidate = reloaded.predict(X.values)
        else:
            reference = model.predict(X)
            candidate = reloaded.predict(
                X.fillna(pd.Series(medians, index=feature_names))[feature_names].values)
        assert np.array_equal(reference, candidate), \
            '{}: exported artifacts do not reproduce the pickle'.format(stem)

        print('  {:<24} {:>3} feat  {:>5} trees  {:>6.2f}MB pkl -> {:>6.2f}MB txt  exact'
              .format(stem, len(feature_names), booster.num_trees(),
                      os.path.getsize(src) / 1e6, os.path.getsize(txt_path) / 1e6))
    return 0


def verify():
    """Replay tests/golden/ through the exported artifacts only.

    Deliberately touches neither the pickles, scikit-learn, nor sglearn, so it
    runs anywhere -- which is the point: it is how the claim "lightgbm 4.x can
    read a model written by 3.3.5" gets measured instead of assumed.
    """
    import lightgbm as lgb

    print('verifying with lightgbm {} / numpy {} / pandas {} / python {}'.format(
        lgb.__version__, np.__version__, pd.__version__,
        '.'.join(str(v) for v in sys.version_info[:3])))

    failures = []
    for _pkl, stem, feat_csv, score_csv, score_col in MODELS:
        txt_path = os.path.join(PKG, stem + '.txt')
        with open(os.path.join(PKG, stem + '.json')) as fh:
            sidecar = json.load(fh)

        names, medians = sidecar['feature_names'], sidecar['medians']
        booster = lgb.Booster(model_file=txt_path)
        assert booster.num_feature() == len(names), \
            '{}: booster/sidecar feature count mismatch'.format(stem)

        feat = pd.read_csv(os.path.join(GOLDEN, feat_csv), float_precision='round_trip')
        X = feat[names]
        if medians is not None:
            X = X.fillna(pd.Series(medians, index=names))[names]
        actual = booster.predict(X.values)

        expected_df = pd.read_csv(os.path.join(GOLDEN, score_csv),
                                  float_precision='round_trip')
        if 'tracr' in expected_df.columns:
            # features_seq.csv was captured for the Chen2013 tracr.
            expected_df = expected_df[expected_df['tracr'] == 'Chen2013']
        expected = expected_df[score_col].to_numpy()

        diff = np.abs(actual - expected)
        exact = np.array_equal(actual, expected)
        status = 'exact' if exact else 'max diff {:.3e}'.format(diff.max())
        print('  {:<24} {:>3} feat  {:>5} trees  vs {:<22} {}'.format(
            stem, len(names), booster.num_trees(), score_csv, status))
        if not exact:
            failures.append((stem, diff.max()))

    if failures:
        print('\nFAILED -- exported artifacts do not reproduce the baseline:')
        for stem, worst in failures:
            print('  {}: max abs diff {:.3e}'.format(stem, worst))
        return 1
    print('\nAll models reproduce tests/golden/ exactly.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--verify', action='store_true',
                        help='check the exported artifacts against tests/golden/ '
                             'and exit; needs neither scikit-learn nor the pickles')
    args = parser.parse_args()
    warnings.simplefilter('ignore')
    return verify() if args.verify else convert()


if __name__ == '__main__':
    sys.exit(main())
