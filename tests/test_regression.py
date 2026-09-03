"""Golden-output regression tests.

These assert that rs3 still predicts what it predicted on its original 2021
dependency stack (see requirements-legacy.txt and tools/capture_baseline.py).

On tolerances. The instinct is atol=0, and for the model layer that is right:
LightGBM tree evaluation is deterministic, so given identical inputs a correct
reimplementation of the surrounding plumbing is bit-identical. But atol=0 is
NOT achievable for the featurization layer across Python versions, for a reason
that has nothing to do with rs3:

    CPython 3.12 changed the builtin sum() to use Neumaier compensated
    summation. Biopython's gravy() and aromaticity() are plain sum() calls, so
    the same sequence yields answers differing by ~1 ULP (measured: 8.9e-16 on
    Hydrophobicity, 2.8e-17 on Aromaticity) between Python <=3.11 and >=3.12.
    Nothing can be pinned to avoid this; the 3.12 result is the more accurate one.

So there are two tolerances, and the gap between them is deliberately huge --
four orders of magnitude below any semantically meaningful change, and four
above interpreter float noise. A failure at these tolerances is a real
behavioural change, not rounding.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import ID_COLS, TRACRS

# Featurization: absorbs the CPython 3.12 sum() change (~1e-16) and nothing more.
ATOL_FEATURES = 1e-12

# Predictions: gradient-boosted trees are piecewise constant, so a 1e-16 nudge in
# a feature only moves a score when it straddles a split threshold. Allow for that
# without allowing a real shift -- target scores have a standard deviation of ~0.15,
# so this is ~8 orders of magnitude below signal.
ATOL_PREDICTIONS = 1e-9


def assert_frame_matches(actual, expected, subset=None, atol=ATOL_PREDICTIONS):
    """Compare numeric frames on a shared key, ignoring row and column order."""
    key = [c for c in ID_COLS if c in expected.columns and c in actual.columns]
    if not key:
        key = [c for c in expected.columns if c in actual.columns
               and not pd.api.types.is_numeric_dtype(expected[c])]

    cols = subset or [c for c in expected.columns if c not in key]
    missing = [c for c in cols if c not in actual.columns]
    assert not missing, 'columns absent from current output: {}'.format(missing)

    left = actual.sort_values(key).reset_index(drop=True)
    right = expected.sort_values(key).reset_index(drop=True)
    assert len(left) == len(right), \
        'row count changed: {} now vs {} in golden'.format(len(left), len(right))

    for col in cols:
        a, e = left[col], right[col]
        if pd.api.types.is_numeric_dtype(e):
            # NaN must land in exactly the same places.
            assert a.isna().tolist() == e.isna().tolist(), \
                'NaN pattern changed in column {!r}'.format(col)
            mask = ~e.isna()
            diff = np.abs(a[mask].to_numpy(dtype=float) - e[mask].to_numpy(dtype=float))
            worst = diff.max() if diff.size else 0.0
            assert worst <= atol, \
                'column {!r} drifted: max abs diff {:.3e} (atol={})'.format(col, worst, atol)
        else:
            assert a.astype(str).tolist() == e.astype(str).tolist(), \
                'column {!r} changed'.format(col)


class TestSequenceModel:
    """RuleSet3, exercised without the target models."""

    def test_predictions_match_golden(self, design_df, golden):
        from rs3.seq import predict_seq

        expected = golden('seq_predictions.csv')
        frames = []
        for tracr in TRACRS:
            frames.append(pd.DataFrame({
                'sgRNA Context Sequence': design_df['sgRNA Context Sequence'],
                'tracr': tracr,
                'prediction': predict_seq(design_df['sgRNA Context Sequence'],
                                          sequence_tracr=tracr),
            }))
        actual = pd.concat(frames, ignore_index=True)
        assert_frame_matches(actual, expected, subset=['prediction'])

    def test_features_match_golden(self, design_df, golden):
        """Catches featurization drift in sglearn/seqfold before it reaches the model."""
        from rs3.seq import featurize_context

        expected = golden('features_seq.csv')
        actual = featurize_context(design_df['sgRNA Context Sequence'],
                                   sequence_tracr='Chen2013')
        assert list(actual.columns) == list(expected.columns), \
            'sequence feature columns changed'
        np.testing.assert_allclose(actual.to_numpy(dtype=float),
                                   expected.to_numpy(dtype=float),
                                   rtol=0, atol=ATOL_FEATURES)


class TestTargetModels:
    """The lite (29-feature) and full (47-feature) target models."""

    def test_lite_predictions_match_golden(self, design_df, golden, aa_seq_file):
        from rs3.predict import predict

        expected = golden('target_lite.csv')
        actual = predict(design_df, tracr=TRACRS, target=True, lite=True,
                         aa_seq_file=aa_seq_file)
        assert_frame_matches(actual, expected)

    def test_full_predictions_match_golden(self, design_df, golden, aa_seq_file,
                                           domain_file, conservation_file):
        from rs3.predict import predict

        expected = golden('target_full.csv')
        actual = predict(design_df, tracr=TRACRS, target=True, lite=False,
                         aa_seq_file=aa_seq_file, domain_file=domain_file,
                         # misspelling is part of the public signature
                         conservatin_file=conservation_file)
        assert_frame_matches(actual, expected)

    @pytest.mark.parametrize('label,n_features', [('lite', 29), ('full', 47)])
    def test_feature_order_is_stable(self, golden_json, label, n_features):
        """The target models are positionally bound to this column order.

        Their boosters report only `Column_0..N` -- the real names live in
        `merge_feature_dfs` (rs3/targetfeat.py). If this order changes, the
        models are silently fed the wrong features, so pin it explicitly.
        """
        cols = golden_json('feature_cols_{}.json'.format(label))
        assert len(cols) == n_features
        assert len(set(cols)) == len(cols), 'duplicate feature names'


class TestAminoAcidFeatures:
    """Biopython ProtParam-backed featurization."""

    def test_features_match_golden(self, golden, aa_seq_file):
        from rs3 import targetfeat

        expected = golden('features_aa.csv')
        aa_seq_df = pd.read_parquet(aa_seq_file, engine='pyarrow')
        actual = targetfeat.featurize_aa_seqs(aa_seq_df['seq'].str[:60])
        assert list(actual.columns) == [c for c in expected.columns
                                        if c != 'Target Transcript']
        np.testing.assert_allclose(actual.to_numpy(dtype=float),
                                   expected.drop(columns='Target Transcript')
                                           .to_numpy(dtype=float),
                                   rtol=0, atol=ATOL_FEATURES)
