"""Shared fixtures for the rs3 test suite.

The golden files under tests/golden/ were captured from rs3 0.0.18 running on
its original dependency stack (see requirements-legacy.txt). They are the
reference for every later change: predictions must reproduce them exactly.
"""

import json
import os

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')
TEST_DATA = os.path.join(REPO, 'test_data')
TARGET_DATA = os.path.join(TEST_DATA, 'target_data')

ID_COLS = ['sgRNA Context Sequence', 'Target Cut Length', 'Target Transcript', 'Orientation']
TRACRS = ['Hsu2013', 'Chen2013']


@pytest.fixture(scope='session')
def repo_root():
    return REPO


@pytest.fixture(scope='session')
def design_df():
    """The 400 CRISPick designs the baseline was captured from."""
    return pd.read_csv(os.path.join(TEST_DATA, 'sgrna-designs.txt'), sep='\t')


@pytest.fixture(scope='session')
def aa_seq_file():
    return os.path.join(TARGET_DATA, 'aa_seqs.pq')


@pytest.fixture(scope='session')
def domain_file():
    return os.path.join(TARGET_DATA, 'protein_domains.pq')


@pytest.fixture(scope='session')
def conservation_file():
    return os.path.join(TARGET_DATA, 'conservation.pq')


@pytest.fixture(scope='session')
def golden():
    """Load a golden CSV by name, e.g. golden('target_lite.csv')."""
    def _load(name):
        path = os.path.join(GOLDEN, name)
        if not os.path.exists(path):
            pytest.fail(
                'Missing golden file {}. Regenerate with:\n'
                '    python tools/capture_baseline.py\n'
                'against the environment in requirements-legacy.txt.'.format(name))
        # float_precision='round_trip' is required for exact comparison: pandas'
        # default C parser uses a fast approximate strtod and loses up to 1 ULP,
        # which alone is enough to fail the atol=0 assertions.
        return pd.read_csv(path, float_precision='round_trip')
    return _load


@pytest.fixture(scope='session')
def golden_json():
    """Load a golden JSON sidecar by name, e.g. golden_json('feature_cols_lite.json')."""
    def _load(name):
        with open(os.path.join(GOLDEN, name)) as fh:
            return json.load(fh)
    return _load
