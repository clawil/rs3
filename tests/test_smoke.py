"""Import and model-loading smoke tests.

Deliberately written against the public API rather than against filenames, so
these keep passing across the model-serialization change: they assert that
`load_seq_model` and `load_target_model` return something predictable, not that
any particular artifact exists on disk.
"""

import importlib
import os

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


@pytest.mark.parametrize('forbidden', ['sklearn', 'scipy'])
def test_library_does_not_import(forbidden):
    """rs3's own modules must not import these.

    scikit-learn left with the pickled models. scipy was never a declared
    runtime dependency -- it sat in dev_requirements while rs3/targetdata.py
    imported `stats` at module scope and never used it, which is an undeclared
    runtime import waiting to break an install.

    Checked against the source rather than sys.modules, because sys.modules
    would be measuring the wrong thing: lightgbm imports scipy itself, so scipy
    is present in any working environment regardless of what rs3 does. The
    claim being tested is about rs3's dependencies, not the transitive closure.
    """
    import ast
    import glob

    pkg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rs3')
    offenders = []
    for path in sorted(glob.glob(os.path.join(pkg, '*.py'))):
        with open(path) as fh:
            tree = ast.parse(fh.read(), path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            if any(n == forbidden or n.startswith(forbidden + '.') for n in names):
                offenders.append('{}:{}'.format(os.path.basename(path), node.lineno))
    assert not offenders, 'rs3 imports {}: {}'.format(forbidden, ', '.join(offenders))
