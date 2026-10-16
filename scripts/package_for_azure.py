"""Build the deployment bundle for Azure App Service (FREE F1 tier). Local only: no network, no Azure.

    python scripts/package_for_azure.py            # writes dist/azure_app.zip

The bundle contains only what the running app needs: the application code, the three
trained model folders, and a slimmed requirements.txt (no training/plotting/Azure SDK
packages) that App Service installs during deployment. It deliberately EXCLUDES raw data,
audio, the virtual environment, tests, docs, Azure templates and anything secret.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]

#: Python version configured on the web app. scikit-learn 1.8.0 (needed by the committed models) requires >= 3.11.
PYTHON_VERSION = "3.12"

CODE_DIRS = ["src"]
#: Never shipped (data, caches, cloud templates, anything that could hold private or secret material).
EXCLUDED_DIR_NAMES = {"__pycache__", "azure", ".git", "venv", ".venv", "data", "tests", "docs", "results", "dist"}
EXCLUDED_SUFFIXES = {".pyc", ".wav", ".mp3", ".flac", ".csv", ".env", ".key", ".pem", ".template"}
MODEL_FILES = {
    "models/questionnaire": {"selected_model.json", "questionnaire_feature_names.json", "explainability_metadata.json",
                             "training_metadata.json", "random_forest.joblib", "logistic_regression.joblib"},
    "models/behavioral": {"behavioral_feature_names.json", "explainability_metadata.json", "training_metadata.json",
                          "logistic_regression.joblib"},
    "models/voice_emotion": {"selected_model.json", "voice_explainability_metadata.json", "voice_training_report.json",
                             "voice_emotion_random_forest.joblib"},
}
#: Packages only needed for training/plotting or tests; not shipped to the free tier (disk is only ~1 GB).
DEPLOY_EXCLUDED_REQUIREMENTS = ("matplotlib", "pytest", "azure", "pyyaml", "httpx")
MAX_BUNDLE_MB = 150


def deploy_requirements(text: str) -> str:
    """requirements.txt for the server: the runtime pins minus training/test/Azure-only packages."""
    kept: List[str] = []
    for line in text.splitlines():
        spec = line.split("#")[0].strip()
        if spec and not spec.lower().startswith(DEPLOY_EXCLUDED_REQUIREMENTS):
            kept.append(spec)
    return "\n".join(kept) + "\n"


def _code_files() -> Iterable[Path]:
    for folder in CODE_DIRS:
        for path in sorted((ROOT / folder).rglob("*")):
            relative = path.relative_to(ROOT)
            if not path.is_file() or EXCLUDED_DIR_NAMES & set(relative.parts) or path.suffix in EXCLUDED_SUFFIXES:
                continue
            if "backend" in relative.parts and "questionnaire" in relative.parts:
                continue  # the standalone questionnaire API is not the deployed app
            if "ai_model" in relative.parts and relative.name in {"audit_dataset.py", "inspect_multiselect_options.py",
                                                                    "inspect_response_categories.py"}:
                continue
            yield path


def _model_files() -> Iterable[Path]:
    for folder, names in MODEL_FILES.items():
        for name in sorted(names):
            path = ROOT / folder / name
            if not path.is_file():
                raise FileNotFoundError(f"Required model file is missing: {folder}/{name}")
            yield path


def build_bundle(output: Path) -> dict:
    """Create the zip and return a summary; raises if a forbidden file would be shipped."""
    members = {}
    for path in list(_code_files()) + list(_model_files()):
        members[path.relative_to(ROOT).as_posix()] = path
    forbidden = [name for name in members if re.search(r"(^|/)(\.env|.*secret.*|.*\.(wav|csv|pem|key))$", name, re.I)]
    if forbidden:
        raise ValueError(f"Refusing to bundle sensitive files: {forbidden}")
    requirements = deploy_requirements((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, name)
        archive.writestr("requirements.txt", requirements)
    size_mb = output.stat().st_size / 1e6
    if size_mb > MAX_BUNDLE_MB:
        raise ValueError(f"Bundle is {size_mb:.0f} MB (limit {MAX_BUNDLE_MB} MB for the free tier).")
    return {"output": str(output), "files": len(members) + 1, "size_mb": round(size_mb, 1), "requirements": requirements.split()}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "azure_app.zip")
    args = parser.parse_args(argv)
    try:
        summary = build_bundle(args.output)
    except (FileNotFoundError, ValueError) as error:
        print(f"Packaging failed: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {summary['output']}: {summary['files']} files, {summary['size_mb']} MB (local only; nothing uploaded).")
    print("Server will install: " + ", ".join(summary["requirements"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
