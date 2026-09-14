"""Azure ML command adapter for the existing questionnaire training workflow.

This module deliberately delegates methodology to
``src.ai_model.questionnaire.train_models``. It stages an approved Azure ML input
into temporary storage and redirects only that module's in-memory artifact paths,
leaving repository-managed data and existing local artifacts untouched.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_model.questionnaire import train_models


EXPECTED_INPUT_FILE = "questionnaire_with_target.csv"
REQUIRED_MODEL_ARTIFACTS = (
    "logistic_regression.joblib",
    "random_forest.joblib",
    "questionnaire_feature_names.json",
    "training_metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing questionnaire training implementation from an Azure ML data input."
    )
    parser.add_argument(
        "--data",
        required=True,
        help=("Azure ML mounted/downloaded file or directory containing the approved "
              f"{EXPECTED_INPUT_FILE} input file."),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Azure ML output directory for generated model and training-report artifacts.",
    )
    return parser.parse_args()


def resolve_input_file(data_path: Path) -> Path:
    """Return the approved target-defined training CSV from an Azure ML input path."""
    if not data_path.exists():
        raise FileNotFoundError(f"Azure ML data path does not exist: {data_path}")
    input_file = data_path / EXPECTED_INPUT_FILE if data_path.is_dir() else data_path
    if input_file.name != EXPECTED_INPUT_FILE:
        raise FileNotFoundError(
            f"Expected input file named {EXPECTED_INPUT_FILE}; received {input_file.name}."
        )
    if not input_file.is_file():
        raise FileNotFoundError(f"Expected input file was not found: {input_file}")
    return input_file


def validate_artifacts(models_dir: Path) -> None:
    missing = [name for name in REQUIRED_MODEL_ARTIFACTS if not (models_dir / name).is_file()]
    if missing:
        raise RuntimeError("Expected model artifacts were not produced: " + ", ".join(missing))


def run_training(data_file: Path, output_dir: Path) -> None:
    """Stage input and delegate to the existing, single-source training implementation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="azureml_questionnaire_") as temporary_directory:
        work_dir = Path(temporary_directory)
        staged_input = work_dir / EXPECTED_INPUT_FILE
        shutil.copy2(data_file, staged_input)
        staged_models = work_dir / "models"
        staged_results = work_dir / "results"

        # These paths are changed only in this process, not in train_models.py or its files.
        train_models.MODELS_DIR = staged_models
        train_models.RESULTS_DIR = staged_results
        print("Input validation: PASS")
        print("Training start: delegating to existing questionnaire train_models.run_training")
        train_models.run_training(input_csv_path=staged_input)
        validate_artifacts(staged_models)

        shutil.copytree(staged_models, output_dir / "models", dirs_exist_ok=True)
        shutil.copytree(staged_results, output_dir / "results", dirs_exist_ok=True)


def main() -> int:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        data_file = resolve_input_file(data_path)
        print(f"Received data path: {data_path}")
        print(f"Output path: {output_dir}")
        run_training(data_file, output_dir)
        print("Training completion: PASS")
        print(f"Model artifacts: {output_dir / 'models'}")
        print(f"Training report artifacts: {output_dir / 'results'}")
        return 0
    except Exception as error:
        print(f"Azure questionnaire training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
