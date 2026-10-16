"""Train the experimental acted-emotion model on CREMA-D (speaker-independent evaluation).

CREMA-D filenames encode the actor and the acted emotion:
``<ActorID>_<Sentence>_<Emotion>_<Level>.wav`` (for example ``1001_DFA_ANG_XX.wav``).

This trains a SEPARATE emotion-recognition model. It does not create, imply or
replace a stress-risk model: emotion classes are never mapped to low / moderate /
high, and the output is never fused. See ``emotion.py`` and
``docs/voice_model_training.md``.

Usage
    python -m src.ai_model.voice.train_emotion --crema-dir data/raw/audio/crema-d

Dataset: Cao et al. (2014), "CREMA-D: Crowd-sourced Emotional Multimodal Actors
Dataset", IEEE Trans. Affective Computing; released under the Open Database License.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from joblib import Parallel, delayed

from .emotion import EMOTION_AUDIO_CONFIG, EMOTION_LABELS, EMOTION_MODELS_DIR, EMOTION_TASK
from .features import COMBINED_FEATURE_NAMES
from .predict import extract_voice_features_from_audio
from .train_voice import (
    MODEL_TYPES,
    TrainingDataError,
    TrainingSettings,
    save_artifacts,
    train_voice_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: CREMA-D emotion codes -> labels.
CREMA_EMOTIONS = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}

DATASET_INFO = {
    "name": "CREMA-D",
    "citation": "Cao et al. 2014, CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset, IEEE Trans. Affective Computing",
    "license": "Open Database License (ODbL)",
    "nature": "acted emotional speech by 91 professional actors reading 12 fixed sentences",
    "limitations": [
        "Acted speech, not spontaneous speech and not students.",
        "Labels are the intended acted emotion, not measured stress or mental state.",
        "The six emotion classes are never mapped to stress-risk labels.",
    ],
}


def parse_crema_filename(name: str) -> Optional[Dict[str, str]]:
    parts = Path(name).stem.split("_")
    if len(parts) < 3 or parts[2] not in CREMA_EMOTIONS or not parts[0].isdigit():
        return None
    return {"filename": name, "participant_id": parts[0], "sentence": parts[1], "target": CREMA_EMOTIONS[parts[2]]}


def build_metadata(crema_dir: Path, limit_per_actor: Optional[int] = None) -> pd.DataFrame:
    """Metadata (filename, actor id, emotion) from the AudioWAV directory; unknown files are skipped."""
    wav_dir = crema_dir / "AudioWAV" if (crema_dir / "AudioWAV").is_dir() else crema_dir
    if not wav_dir.is_dir():
        raise TrainingDataError(f"CREMA-D directory not found: {crema_dir}")
    rows = [row for row in (parse_crema_filename(path.name) for path in sorted(wav_dir.glob("*.wav"))) if row]
    if not rows:
        raise TrainingDataError(f"No CREMA-D .wav files with the expected names were found in {wav_dir}")
    frame = pd.DataFrame(rows)
    frame["path"] = [str(wav_dir / name) for name in frame["filename"]]
    if limit_per_actor:
        frame = frame.groupby("participant_id", group_keys=False).head(limit_per_actor).reset_index(drop=True)
    return frame


def _extract_one(path: str) -> Optional[Dict[str, float]]:
    warnings.filterwarnings("ignore")
    try:
        return extract_voice_features_from_audio(path, audio_config=EMOTION_AUDIO_CONFIG)
    except Exception:  # a single bad clip must not abort a multi-thousand-file run
        return None


def extract_feature_table(metadata: pd.DataFrame, n_jobs: int = -1) -> tuple[pd.DataFrame, List[str]]:
    """41 acoustic features per clip using the project's own preprocessing/extraction (parallel)."""
    results = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(_extract_one)(path) for path in metadata["path"])
    ok = [index for index, features in enumerate(results) if features is not None]
    failed = [metadata["filename"].iloc[index] for index, features in enumerate(results) if features is None]
    table = pd.DataFrame([results[index] for index in ok], columns=list(COMBINED_FEATURE_NAMES))
    table.insert(0, "participant_id", metadata["participant_id"].iloc[ok].to_numpy())
    table["target"] = metadata["target"].iloc[ok].to_numpy()
    return table.reset_index(drop=True), failed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--crema-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "audio" / "crema-d")
    parser.add_argument("--features-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "crema_d_features.csv",
                        help="Feature cache; reused if it exists (delete it to re-extract).")
    parser.add_argument("--limit-per-actor", type=int, default=None, help="Use only N clips per actor (quick runs).")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--model-type", default="random_forest", choices=tuple(MODEL_TYPES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=EMOTION_MODELS_DIR)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "results" / "voice_emotion")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    settings = TrainingSettings(
        dataset_kind="public_acted", model_type=args.model_type, seed=args.seed, test_size=args.test_size,
        min_participants_per_class=20, labels=EMOTION_LABELS, task=EMOTION_TASK, model_prefix="voice_emotion",
    )
    try:
        failed: List[str] = []
        if args.features_csv.is_file():
            print(f"Reusing cached features: {args.features_csv}")
            table = pd.read_csv(args.features_csv, dtype={"participant_id": str})
        else:
            metadata = build_metadata(args.crema_dir, args.limit_per_actor)
            print(f"Extracting features from {len(metadata)} clips ({metadata['participant_id'].nunique()} actors)...")
            table, failed = extract_feature_table(metadata, args.n_jobs)
            if args.limit_per_actor is None:  # only cache a complete extraction
                args.features_csv.parent.mkdir(parents=True, exist_ok=True)
                table.to_csv(args.features_csv, index=False)
        model, report, explainability = train_voice_model(table, settings)
        report["dataset"] = {**DATASET_INFO, "clips_used": int(len(table)), "clips_failed_extraction": len(failed),
                             "audio_config": {"min_duration_seconds": EMOTION_AUDIO_CONFIG.min_duration_seconds}}
        paths = save_artifacts(model, report, explainability, args.output_dir, args.report_dir, args.overwrite)
    except (TrainingDataError, FileNotFoundError, FileExistsError) as error:
        print(f"Training aborted: {error}", file=sys.stderr)
        return 2
    held = report["held_out_metrics"]
    print(f"Trained {report['model_name']}: held-out actors={report['split']['held_out_test']['participants']} "
          f"accuracy={held['accuracy']:.3f} macro_f1={held['macro_f1']:.3f} (chance = {1 / len(EMOTION_LABELS):.3f})")
    print("Artifacts: " + ", ".join(str(path) for path in paths.values()))
    print("Acted emotion recognition only - not a stress or mental-health measure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
