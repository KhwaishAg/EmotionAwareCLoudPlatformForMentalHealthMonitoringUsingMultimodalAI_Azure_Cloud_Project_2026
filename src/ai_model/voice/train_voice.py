"""Reproducible training and held-out evaluation for the voice stress-risk model.

This module trains a voice classifier ONLY from a labeled dataset you supply.
It never generates data. As of this commit the repository contains no audio
recordings and no labels, so no model has been trained (see
``docs/voice_model_training.md`` for the dataset required).

Pipeline
    1. Load a metadata CSV (filename, participant id, label) and extract the
       41 combined acoustic features per recording with the existing
       preprocessing/feature code - or load a pre-extracted feature CSV.
    2. Validate: labels must be exactly ``low`` / ``moderate`` / ``high`` (the
       classes the fusion contract requires), every class needs enough
       distinct participants, and participant ids are mandatory.
    3. Split by PARTICIPANT (no speaker appears in both train and held-out
       test), stratified by label, with a fixed seed.
    4. Cross-validate on the training participants (grouped folds) and fit the
       final model on the training split only.
    5. Evaluate once on the held-out participants (accuracy, macro-F1,
       per-class precision/recall, confusion matrix).
    6. Write artifacts + a training report that records the dataset kind,
       sizes, seed and library versions.

Usage
    python -m src.ai_model.voice.train_voice \\
        --metadata data/raw/audio/metadata.csv --dataset-kind real_student

Non-clinical: metrics describe performance on the supplied dataset only and are
not evidence of clinical validity or real-world accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import MODEL_CONFIG, PATH_CONFIG
from .dataset import DatasetError, load_voice_dataset
from .evaluation import evaluate_voice_model
from .features import COMBINED_FEATURE_NAMES
from .model import BaseVoiceClassifier, VoiceLogisticRegression, VoiceRandomForest
from .predict import TRAINING_REPORT_FILENAME

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: The only classes the voice -> fusion contract accepts.
RISK_LABELS: Tuple[str, ...] = ("low", "moderate", "high")

#: What the training data is. ``synthetic_demo`` models are never served as real predictors.
DATASET_KINDS: Tuple[str, ...] = ("real_student", "public_acted", "synthetic_demo")

MODEL_TYPES = {
    "random_forest": ("random_forest", VoiceRandomForest),
    "logistic_regression": ("logistic_regression", VoiceLogisticRegression),
}

DEFAULT_MIN_PARTICIPANTS_PER_CLASS = 10

DISCLAIMER = (
    "Metrics describe performance on the supplied dataset only. They are not evidence of clinical validity "
    "or of real-world accuracy. Non-clinical research tool."
)


class TrainingDataError(ValueError):
    """The supplied dataset cannot support training a trustworthy voice model."""


@dataclass(frozen=True)
class TrainingSettings:
    dataset_kind: str
    model_type: str = "random_forest"
    seed: int = MODEL_CONFIG.random_seed
    test_size: float = MODEL_CONFIG.test_size
    min_participants_per_class: int = DEFAULT_MIN_PARTICIPANTS_PER_CLASS
    label_column: str = MODEL_CONFIG.label_column
    participant_column: str = MODEL_CONFIG.participant_id_column
    #: Class labels the model is trained on. Defaults to the fusion contract's stress-risk classes;
    #: other tasks (e.g. emotion recognition) pass their own and are never fed to fusion.
    labels: Tuple[str, ...] = RISK_LABELS
    #: What the model predicts; recorded in the report and checked by the inference loaders.
    task: str = "stress_risk"
    model_prefix: str = "voice"


# ---------------------------------------------------------------------------
# Validation and splitting
# ---------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame, settings: TrainingSettings) -> pd.DataFrame:
    """Return a cleaned copy of ``df`` or raise :class:`TrainingDataError`."""
    if settings.dataset_kind not in DATASET_KINDS:
        raise TrainingDataError(f"dataset_kind must be one of {DATASET_KINDS}.")
    if settings.model_type not in MODEL_TYPES:
        raise TrainingDataError(f"model_type must be one of {tuple(MODEL_TYPES)}.")
    if not 0.05 <= settings.test_size <= 0.5:
        raise TrainingDataError("test_size must be between 0.05 and 0.5.")
    needed = [settings.label_column, settings.participant_column, *COMBINED_FEATURE_NAMES]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise TrainingDataError(
            f"Dataset is missing required columns: {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}"
        )
    data = df[needed].copy()
    if data[settings.participant_column].isna().any():
        raise TrainingDataError("Every recording needs a participant id (needed for a leak-free split).")
    data[settings.participant_column] = data[settings.participant_column].astype(str)
    labels = data[settings.label_column].astype(str).str.strip().str.lower()
    unknown = sorted(set(labels) - set(settings.labels))
    if unknown:
        raise TrainingDataError(
            f"Unsupported labels {unknown}. This task requires exactly {list(settings.labels)}; "
            "do not remap labels from a different construct (for example acted emotions) onto them."
        )
    data[settings.label_column] = labels
    features = data[list(COMBINED_FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise TrainingDataError("Feature columns contain missing, non-numeric or non-finite values.")
    data[list(COMBINED_FEATURE_NAMES)] = features

    per_class = data.groupby(settings.label_column)[settings.participant_column].nunique()
    short = {label: int(per_class.get(label, 0)) for label in settings.labels
             if per_class.get(label, 0) < settings.min_participants_per_class}
    if short:
        raise TrainingDataError(
            f"Not enough distinct participants per class (need >= {settings.min_participants_per_class}); "
            f"found {short}. Collect more labeled participants rather than lowering the threshold."
        )
    return data.reset_index(drop=True)


def participant_split(data: pd.DataFrame, settings: TrainingSettings) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified, participant-disjoint train/test indices (deterministic for a seed)."""
    n_splits = max(2, int(round(1.0 / settings.test_size)))
    labels = data[settings.label_column].to_numpy()
    groups = data[settings.participant_column].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=settings.seed)
    for train_idx, test_idx in splitter.split(np.zeros(len(data)), labels, groups):
        if set(labels[train_idx]) == set(settings.labels) and set(labels[test_idx]) == set(settings.labels):
            if set(groups[train_idx]) & set(groups[test_idx]):  # defensive: must never happen
                raise TrainingDataError("Participant leakage detected between train and test split.")
            return train_idx, test_idx
    raise TrainingDataError(
        "Could not build a participant-disjoint split containing all three classes in both train and test; "
        "more participants are needed."
    )


def cross_validate(data: pd.DataFrame, settings: TrainingSettings) -> Optional[Dict[str, Any]]:
    """Grouped CV macro-F1 on the training portion; ``None`` if too few participants."""
    labels = data[settings.label_column].to_numpy()
    groups = data[settings.participant_column].to_numpy()
    per_class = min(len(set(groups[labels == label])) for label in settings.labels)
    n_splits = min(5, per_class)
    if n_splits < 2:
        return None
    _, factory = MODEL_TYPES[settings.model_type]
    scores: List[float] = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=settings.seed)
    features = data[list(COMBINED_FEATURE_NAMES)]
    for fit_idx, val_idx in splitter.split(features, labels, groups):
        model = factory(random_state=settings.seed, expected_feature_names=list(COMBINED_FEATURE_NAMES))
        model.fit(features.iloc[fit_idx], labels[fit_idx])
        predicted = model.predict(features.iloc[val_idx])
        scores.append(float(f1_score(labels[val_idx], predicted, labels=list(settings.labels),
                                     average="macro", zero_division=0)))
    return {"folds": n_splits, "macro_f1_mean": float(np.mean(scores)), "macro_f1_std": float(np.std(scores)),
            "macro_f1_per_fold": scores}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _explainability(model: BaseVoiceClassifier, limit: int = 10) -> Dict[str, Any]:
    names = list(COMBINED_FEATURE_NAMES)
    if hasattr(model, "feature_importances_dict"):
        scores = dict(model.feature_importances_dict)
        method = "random_forest_impurity_importance"
    else:
        scores = dict(zip(names, np.abs(model.coef_).mean(axis=0).tolist()))
        method = "mean_absolute_logistic_coefficient"
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    return {"method": method, "top_features": [{"feature": name, "importance": float(value)} for name, value in top],
            "note": "Model-associated importance only; not causal evidence."}


def _fingerprint(data: pd.DataFrame, settings: TrainingSettings) -> str:
    columns = [settings.label_column, *COMBINED_FEATURE_NAMES]
    hashed = pd.util.hash_pandas_object(data[columns], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def train_voice_model(df: pd.DataFrame, settings: TrainingSettings) -> Tuple[BaseVoiceClassifier, Dict[str, Any], Dict[str, Any]]:
    """Validate, split, cross-validate, fit and evaluate. Returns (model, report, explainability).

    Nothing is written to disk here.
    """
    data = validate_dataset(df, settings)
    train_idx, test_idx = participant_split(data, settings)
    train, test = data.iloc[train_idx].reset_index(drop=True), data.iloc[test_idx].reset_index(drop=True)
    features = list(COMBINED_FEATURE_NAMES)

    cross_validation = cross_validate(train, settings)

    suffix, factory = MODEL_TYPES[settings.model_type]
    model_name = f"{settings.model_prefix}_{suffix}"
    model = factory(random_state=settings.seed, expected_feature_names=features)
    model.fit(train[features], train[settings.label_column].to_numpy())
    metrics = evaluate_voice_model(
        model, test[features], test[settings.label_column].to_numpy(), average="macro", include_confusion_matrix=True
    )

    def counts(frame: pd.DataFrame) -> Dict[str, Any]:
        return {
            "recordings": int(len(frame)),
            "participants": int(frame[settings.participant_column].nunique()),
            "recordings_per_class": {label: int((frame[settings.label_column] == label).sum()) for label in settings.labels},
            "participants_per_class": {label: int(frame.loc[frame[settings.label_column] == label, settings.participant_column].nunique())
                                       for label in settings.labels},
        }

    report = {
        "task": settings.task,
        "dataset_kind": settings.dataset_kind,
        "synthetic_data": settings.dataset_kind == "synthetic_demo",
        "model_name": model_name,
        "model_type": settings.model_type,
        "classes": list(settings.labels),
        "feature_count": len(features),
        "dataset_fingerprint_sha256": _fingerprint(data, settings),
        "random_seed": settings.seed,
        "split": {"method": "stratified_group_kfold_by_participant", "test_size_requested": settings.test_size,
                  "train": counts(train), "held_out_test": counts(test)},
        "cross_validation_on_train": cross_validation,
        "held_out_metrics": metrics.to_dict(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "disclaimer": DISCLAIMER,
    }
    return model, report, _explainability(model)


def save_artifacts(model: BaseVoiceClassifier, report: Dict[str, Any], explainability: Dict[str, Any],
                   output_dir: Path, report_dir: Optional[Path] = None, overwrite: bool = False) -> Dict[str, Path]:
    """Write model, selector, explainability and report. Refuses to clobber unless ``overwrite``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{report['model_name']}.joblib"
    paths = {
        "model": model_path,
        "selector": output_dir / "selected_model.json",
        "explainability": output_dir / "voice_explainability_metadata.json",
        "report": output_dir / TRAINING_REPORT_FILENAME,
    }
    existing = [path.name for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifacts ({', '.join(existing)}); pass --overwrite.")
    model.save(model_path, metadata={"dataset_kind": report["dataset_kind"], "classes": report["classes"],
                                     "random_seed": report["random_seed"]})
    paths["selector"].write_text(json.dumps({"selected_model": report["model_name"],
                                             "dataset_kind": report["dataset_kind"]}, indent=2), encoding="utf-8")
    paths["explainability"].write_text(json.dumps(explainability, indent=2), encoding="utf-8")
    paths["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report_dir is not None:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        copy = Path(report_dir) / "evaluation_report.json"
        copy.write_text(json.dumps(report, indent=2), encoding="utf-8")
        paths["report_copy"] = copy
    return paths


def load_training_frame(metadata: Optional[Path], audio_dir: Optional[Path], features_csv: Optional[Path],
                        settings: TrainingSettings, filename_column: Optional[str] = None) -> pd.DataFrame:
    """Build the (participant, 41 features, label) frame from audio metadata or a feature CSV."""
    if features_csv is not None:
        return pd.read_csv(features_csv)
    if metadata is None:
        raise TrainingDataError("Provide --metadata (with audio) or --features-csv.")
    return load_voice_dataset(metadata, audio_dir=audio_dir, label_column=settings.label_column,
                              participant_id_column=settings.participant_column, filename_column=filename_column)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / PATH_CONFIG.raw_audio_dir / PATH_CONFIG.metadata_filename,
                        help="CSV with filename, participant id and label columns.")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Directory containing the audio files.")
    parser.add_argument("--features-csv", type=Path, default=None,
                        help="Pre-extracted features CSV (from save_voice_dataset); skips audio extraction.")
    parser.add_argument("--dataset-kind", required=True, choices=DATASET_KINDS,
                        help="What the data is. synthetic_demo models are refused at inference time.")
    parser.add_argument("--model-type", default="random_forest", choices=tuple(MODEL_TYPES))
    parser.add_argument("--seed", type=int, default=MODEL_CONFIG.random_seed)
    parser.add_argument("--test-size", type=float, default=MODEL_CONFIG.test_size)
    parser.add_argument("--min-participants-per-class", type=int, default=DEFAULT_MIN_PARTICIPANTS_PER_CLASS)
    parser.add_argument("--label-column", default=MODEL_CONFIG.label_column)
    parser.add_argument("--participant-column", default=MODEL_CONFIG.participant_id_column)
    parser.add_argument("--filename-column", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / PATH_CONFIG.models_dir)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "results" / "voice")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = TrainingSettings(
        dataset_kind=args.dataset_kind, model_type=args.model_type, seed=args.seed, test_size=args.test_size,
        min_participants_per_class=args.min_participants_per_class, label_column=args.label_column,
        participant_column=args.participant_column,
    )
    if settings.dataset_kind in ("synthetic_demo", "public_acted"):
        print(f"WARNING: {settings.dataset_kind} data - the stress-risk voice model trained from it will be refused "
              "at inference time (it is not defensible stress ground truth).", file=sys.stderr)
    try:
        frame = load_training_frame(args.metadata if args.features_csv is None else None, args.audio_dir,
                                    args.features_csv, settings, args.filename_column)
        model, report, explainability = train_voice_model(frame, settings)
        paths = save_artifacts(model, report, explainability, args.output_dir, args.report_dir, args.overwrite)
    except (TrainingDataError, DatasetError, FileNotFoundError, FileExistsError) as error:
        print(f"Training aborted: {error}", file=sys.stderr)
        return 2
    held_out = report["held_out_metrics"]
    print(f"Trained {report['model_name']} on {report['split']['train']['participants']} participants; "
          f"held-out ({report['split']['held_out_test']['participants']} participants): "
          f"accuracy={held_out['accuracy']:.3f} macro_f1={held_out['macro_f1']:.3f}")
    print("Artifacts: " + ", ".join(str(path) for path in paths.values()))
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
