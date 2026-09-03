"""Import and model-loading smoke tests.

Deliberately written against the public API rather than against filenames, so
these keep passing across the model-serialization change: they assert that
`load_seq_model` and `load_target_model` return something predictable, not that
any particular artifact exists on disk.
"""

import importlib

import pytest

MODULES = ['rs3', 'rs3.seq', 'rs3.predict', 'rs3.predicttarg',
           'rs3.targetdata', 'rs3.targetfeat']


@pytest.mark.parametrize('name', MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_version_is_declared():
    import rs3

    assert isinstance(rs3.__version__, str)
    assert rs3.__version__.count('.') >= 2, rs3.__version__


def test_sequence_model_loads():
    from rs3.seq import load_seq_model

    model = load_seq_model()
    assert model is not None
    assert hasattr(model, 'predict') or isinstance(model, tuple)


@pytest.mark.parametrize('lite', [True, False])
def test_target_model_loads(lite):
    from rs3.predicttarg import load_target_model

    model = load_target_model(lite=lite)
    assert model is not None
    assert hasattr(model, 'predict') or isinstance(model, tuple)
