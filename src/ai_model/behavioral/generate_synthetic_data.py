"""Generate a synthetic labeled dataset for the behavioral stress-risk modality.

IMPORTANT: This data is SYNTHETIC, generated for development/demo purposes
because no real behavioral dataset was collected for this project in the
time available. It is clearly labeled as such in every output artifact.
Near-perfect model scores on this data reflect clean synthetic class
separation, not real-world predictive performance -- report this honestly.

Run:
    python generate_synthetic_data.py
Produces:
    data/processed/behavioral_synthetic.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .behavioral_schema import REQUIRED_BEHAVIORAL_FIELDS
except ImportError:  # pragma: no cover - direct execution support
    from behavioral_schema import REQUIRED_BEHAVIORAL_FIELDS

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "behavioral_synthetic.csv"
RANDOM_SEED = 42
N_SAMPLES = 900  # 300 per class, balanced


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def generate(n_per_class: int = 300, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    # LOW risk profile: good sleep, consistent study, social, low screen time, regular routine
    low_sleep = _clip(rng.normal(7.5, 0.8, n_per_class), 4, 11)
    low_consistency = _clip(rng.normal(4.2, 0.6, n_per_class), 1, 5)
    low_social = _clip(rng.normal(4.0, 0.7, n_per_class), 1, 5)
    low_screen = _clip(rng.normal(3.5, 1.2, n_per_class), 0, 14)
    low_irregular = _clip(rng.normal(1.8, 0.6, n_per_class), 1, 5)
    for i in range(n_per_class):
        rows.append((low_sleep[i], low_consistency[i], low_social[i], low_screen[i], low_irregular[i], "low"))

    # MODERATE risk profile: middling everything
    mod_sleep = _clip(rng.normal(6.2, 0.7, n_per_class), 4, 10)
    mod_consistency = _clip(rng.normal(3.0, 0.6, n_per_class), 1, 5)
    mod_social = _clip(rng.normal(2.8, 0.7, n_per_class), 1, 5)
    mod_screen = _clip(rng.normal(6.0, 1.3, n_per_class), 0, 16)
    mod_irregular = _clip(rng.normal(3.0, 0.6, n_per_class), 1, 5)
    for i in range(n_per_class):
        rows.append((mod_sleep[i], mod_consistency[i], mod_social[i], mod_screen[i], mod_irregular[i], "moderate"))

    # HIGH risk profile: poor sleep, inconsistent study, isolated, high screen time, irregular routine
    high_sleep = _clip(rng.normal(4.8, 0.9, n_per_class), 2, 8)
    high_consistency = _clip(rng.normal(1.8, 0.6, n_per_class), 1, 5)
    high_social = _clip(rng.normal(1.6, 0.6, n_per_class), 1, 5)
    high_screen = _clip(rng.normal(9.5, 1.6, n_per_class), 2, 16)
    high_irregular = _clip(rng.normal(4.3, 0.6, n_per_class), 1, 5)
    for i in range(n_per_class):
        rows.append((high_sleep[i], high_consistency[i], high_social[i], high_screen[i], high_irregular[i], "high"))

    df = pd.DataFrame(rows, columns=list(REQUIRED_BEHAVIORAL_FIELDS) + ["stress_level"])
    # Shuffle so class order isn't trivially learnable by row position.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    dataset = generate(n_per_class=N_SAMPLES // 3)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print("SYNTHETIC BEHAVIORAL DATASET GENERATED (NOT REAL DATA)")
    print(f"Rows: {len(dataset)}  |  Classes: {dataset['stress_level'].value_counts().to_dict()}")
    print(f"Output: {OUTPUT_PATH}")
