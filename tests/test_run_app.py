"""Launcher environment checks (no server is started)."""
import sys

import run_app


def test_environment_check_passes_in_supported_env():
    try:
        import sklearn
    except ImportError:  # pragma: no cover
        return
    problems = run_app.check_environment()
    if sys.version_info >= (3, 11) and sklearn.__version__ == run_app.EXPECTED_SKLEARN:
        assert problems == []
    else:
        assert problems and all(isinstance(p, str) for p in problems)


def test_old_python_is_reported(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    assert any("Python 3.11+" in problem for problem in run_app.check_environment())


def test_missing_package_is_reported(monkeypatch):
    real_import = run_app.importlib.import_module

    def fake(name):
        if name == "fastapi":
            raise ImportError(name)
        return real_import(name)

    monkeypatch.setattr(run_app.importlib, "import_module", fake)
    assert any("fastapi" in problem and "pip install" in problem for problem in run_app.check_environment())
