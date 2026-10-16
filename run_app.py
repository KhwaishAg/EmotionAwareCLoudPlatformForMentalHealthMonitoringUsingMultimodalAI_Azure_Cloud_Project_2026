"""Start the Emotion-Aware assessment app locally.

    python run_app.py            # serves http://127.0.0.1:8000 and opens the browser
    python run_app.py --port 9000 --no-browser

Checks the environment first (Python version, scikit-learn version the committed
models need, required packages) and explains how to fix problems instead of
failing later with an obscure error. The server listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import threading
import webbrowser

EXPECTED_SKLEARN = "1.8.0"
REQUIRED_PACKAGES = ("fastapi", "uvicorn", "pydantic", "numpy", "pandas", "sklearn", "joblib", "librosa", "scipy")


def check_environment() -> list[str]:
    problems = []
    if sys.version_info < (3, 11):
        problems.append(f"Python 3.11+ is required (found {sys.version.split()[0]}); scikit-learn {EXPECTED_SKLEARN} "
                        "needs it and the committed models were saved with that version.")
    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        problems.append("Missing packages: " + ", ".join(missing) + ". Run: pip install -r requirements.txt")
    else:
        import sklearn

        if sklearn.__version__ != EXPECTED_SKLEARN:
            problems.append(f"scikit-learn {sklearn.__version__} is installed but the committed models need "
                            f"{EXPECTED_SKLEARN}. Run: pip install scikit-learn=={EXPECTED_SKLEARN}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    problems = check_environment()
    if problems:
        print("Cannot start:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1

    import uvicorn

    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving the assessment app at {url}  (Ctrl+C to stop). Non-clinical demonstration only.")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("src.backend.assessment.api:app", host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
