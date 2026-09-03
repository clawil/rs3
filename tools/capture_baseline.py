#!/usr/bin/env python
"""Capture golden prediction outputs for the rs3 regression suite.

Run this from the repo root against the LEGACY environment described in
requirements-legacy.txt (Python 3.10, scikit-learn 1.0.2, lightgbm 3.3.5,
numpy 1.26.4, pandas 1.5.3) to record what rs3 0.0.18 actually predicts:

    ~/miniforge3/envs/rs3-legacy/bin/python tools/capture_baseline.py

The resulting tests/golden/*.csv are the acceptance gate for the dependency
modernization: every later change must reproduce them bit-for-bit. Re-run this
script only to establish a NEW baseline, never to paper over a diff.

Everything here is network-free -- the precomputed parquet files under
test_data/target_data/ stand in for the Ensembl and UCSC API calls, so the
baseline is reproducible and cannot drift with upstream annotation releases.

Two kinds of golden file are written:

  target_*.csv / seq_*.csv    model OUTPUTS, via the public rs3 API
  features_*.csv              model INPUTS, via the same internal calls
                              rs3.predict.predict makes

Capturing both is deliberate. If a prediction ever drifts, the feature files
say immediately whether the cause is upstream (featurization changed) or
downstream (the model or its serialization changed) -- which is otherwise a
slow thing to bisect.
"""

import json
import os
import platform
import sys
import warnings

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Run against the working tree, not an installed copy -- the point is to capture
# what the code in this checkout does.
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TEST_DATA = os.path.join(REPO, 'test_data')
TARGET_DATA = os.path.join(TEST_DATA, 'target_data')
GOLDEN = os.path.join(REPO, 'tests', 'golden')

DESIGN_FILE = os.path.join(TEST_DATA, 'sgrna-designs.txt')
AA_SEQ_FILE = os.path.join(TARGET_DATA, 'aa_seqs.pq')
DOMAIN_FILE = os.path.join(TARGET_DATA, 'protein_domains.pq')
CONSERVATION_FILE = os.path.join(TARGET_DATA, 'conservation.pq')

TRACRS = ['Hsu2013', 'Chen2013']
ID_COLS = ['sgRNA Context Sequence', 'Target Cut Length', 'Target Transcript', 'Orientation']

# No float_format: pandas writes Python's repr, which is the shortest string that
# round-trips a float64 exactly. Explicit widths like '%.17g' are no more precise
# and just make the files bigger.
#
# The exactness is only recovered if the READER asks for it -- pandas' default C
# parser uses a fast approximate strtod and loses up to 1 ULP, which is enough to
# fail an atol=0 comparison. tests/conftest.py therefore reads these files with
# float_precision='round_trip'. Both halves are required; neither alone is exact.


def _write(df, name):
    """Write one golden file with exact float precision and a stable row order."""
    path = os.path.join(GOLDEN, name)
    df.to_csv(path, index=False)
    print('  wrote {:<28} {:>6} rows x {:>3} cols'.format(name, df.shape[0], df.shape[1]))
    return path


def capture_seq_predictions(design_df):
    """RuleSet3 sequence model, in isolation from the target models."""
    from rs3.seq import predict_seq

    frames = []
    for tracr in TRACRS:
        frames.append(pd.DataFrame({
            'sgRNA Context Sequence': design_df['sgRNA Context Sequence'],
            'tracr': tracr,
            'prediction': predict_seq(design_df['sgRNA Context Sequence'],
                                      sequence_tracr=tracr),
        }))
    return _write(pd.concat(frames, ignore_index=True), 'seq_predictions.csv')


def capture_seq_features(design_df):
    """The 632-column matrix fed to the sequence model."""
    from rs3.seq import featurize_context

    features = featurize_context(design_df['sgRNA Context Sequence'],
                                 sequence_tracr='Chen2013')
    return _write(features, 'features_seq.csv')


def _target_feature_matrix(design_df, lite):
    """Rebuild the model input matrix exactly as rs3.predict.predict does.

    Mirrors rs3/predict.py:71-121. Kept in step with that function on purpose --
    if the two ever diverge, the features_*.csv goldens stop describing what the
    model is actually fed, so any change to predict() must be reflected here.
    """
    from rs3 import targetfeat

    out_df = targetfeat.add_target_columns(design_df)
    transcript_bases = pd.Series(out_df['Transcript Base'].unique())

    aa_seq_df = pd.read_parquet(AA_SEQ_FILE, engine='pyarrow',
                                filters=[[('Transcript Base', 'in', transcript_bases)]])
    aa_subseq_df = targetfeat.get_aa_subseq_df(sg_designs=out_df, aa_seq_df=aa_seq_df,
                                               width=16, id_cols=ID_COLS)

    domain_feature_df = conservation_feature_df = None
    if not lite:
        domain_df = pd.read_parquet(DOMAIN_FILE, engine='pyarrow',
                                    filters=[[('Transcript Base', 'in', transcript_bases)]])
        domain_feature_df = targetfeat.get_protein_domain_features(
            out_df, domain_df, id_cols=ID_COLS, transcript_base_col='Transcript Base')

        conservation_df = pd.read_parquet(CONSERVATION_FILE, engine='pyarrow',
                                          filters=[[('Transcript Base', 'in', transcript_bases)]])
        conservation_feature_df = targetfeat.get_conservation_features(
            out_df, conservation_df, small_width=2, large_width=16,
            conservation_column='ranked_conservation', id_cols=ID_COLS)

    feature_df, feature_cols = targetfeat.merge_feature_dfs(
        out_df, aa_subseq_df=aa_subseq_df, domain_df=domain_feature_df,
        conservation_df=conservation_feature_df, id_cols=ID_COLS)
    return feature_df, feature_cols


def capture_target_features(design_df):
    """The 29- and 47-column matrices fed to the two target models."""
    paths = []
    for lite in (True, False):
        feature_df, feature_cols = _target_feature_matrix(design_df, lite=lite)
        label = 'lite' if lite else 'full'
        print('  {} target features: {} columns'.format(label, len(feature_cols)))
        paths.append(_write(feature_df[ID_COLS + feature_cols],
                            'features_target_{}.csv'.format(label)))
        # The column ORDER is what the models are positionally bound to, so record
        # it separately rather than trusting CSV header order to survive a reader.
        with open(os.path.join(GOLDEN, 'feature_cols_{}.json'.format(label)), 'w') as fh:
            json.dump(feature_cols, fh, indent=2)
    return paths


def capture_target_predictions(design_df):
    """Both target models end-to-end, through the public predict() API."""
    from rs3.predict import predict

    paths = []
    for lite in (True, False):
        kwargs = dict(tracr=TRACRS, target=True, lite=lite, aa_seq_file=AA_SEQ_FILE)
        if not lite:
            # NB: `conservatin_file` is misspelled in the public signature
            # (rs3/predict.py:38). It is API; do not "fix" it here.
            kwargs.update(domain_file=DOMAIN_FILE, conservatin_file=CONSERVATION_FILE)
        out = predict(design_df, **kwargs)
        score_cols = [c for c in out.columns if 'Score' in c]
        label = 'lite' if lite else 'full'
        paths.append(_write(out[ID_COLS + score_cols], 'target_{}.csv'.format(label)))
    return paths


def capture_aa_features():
    """Amino-acid featurization, which routes through Biopython's ProtParam.

    Exercised directly rather than through the pipeline so that a Biopython
    change shows up here instead of as an unexplained prediction diff.
    """
    from rs3 import targetfeat

    aa_seq_df = pd.read_parquet(AA_SEQ_FILE, engine='pyarrow')
    features = targetfeat.featurize_aa_seqs(aa_seq_df['seq'].str[:60])
    return _write(pd.concat([aa_seq_df[['Target Transcript']].reset_index(drop=True),
                             features.reset_index(drop=True)], axis=1),
                  'features_aa.csv')


def write_provenance():
    """Record which stack produced the baseline, so a mismatch is diagnosable."""
    import importlib

    versions = {'python': platform.python_version(), 'platform': platform.platform()}
    for pkg in ['numpy', 'pandas', 'sklearn', 'lightgbm', 'pyarrow', 'joblib',
                'Bio', 'sglearn', 'seqfold']:
        try:
            versions[pkg] = getattr(importlib.import_module(pkg), '__version__', 'unknown')
        except ImportError:
            versions[pkg] = 'not installed'
    path = os.path.join(GOLDEN, 'PROVENANCE.json')
    with open(path, 'w') as fh:
        json.dump(versions, fh, indent=2, sort_keys=True)
    print('  wrote PROVENANCE.json')
    return versions


def main():
    warnings.simplefilter('ignore')
    if not os.path.isdir(GOLDEN):
        os.makedirs(GOLDEN)

    design_df = pd.read_csv(DESIGN_FILE, sep='\t')
    print('designs: {} rows from {}'.format(design_df.shape[0],
                                            os.path.relpath(DESIGN_FILE, REPO)))

    print('\nsequence model:')
    capture_seq_features(design_df)
    capture_seq_predictions(design_df)

    print('\ntarget models:')
    capture_target_features(design_df)
    capture_target_predictions(design_df)

    print('\namino acid featurization:')
    capture_aa_features()

    print('\nprovenance:')
    versions = write_provenance()
    for k in sorted(versions):
        print('  {:<12} {}'.format(k, versions[k]))

    print('\nBaseline captured to tests/golden/. Verify with: pytest tests/ -v')
    return 0


if __name__ == '__main__':
    sys.exit(main())
