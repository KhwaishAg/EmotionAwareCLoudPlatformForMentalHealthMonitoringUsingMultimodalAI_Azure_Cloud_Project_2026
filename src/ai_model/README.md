# AI Models Architecture & Pipelines

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This document specifies the architecture, feature engineering pipelines, model baselines, and inference interfaces for the three modality models and the multimodal fusion layer in the **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

---

## Modality Architecture Overview

```
+---------------------------------------------------------------------------------------+
|                                  STUDENT INPUT CHANNELS                               |
|            [Questionnaire Survey]     [Short Audio: 30-60s]     [Behavioral Logs]     |
+----------------------+--------------------------+-------------------------+-----------+
                       |                          |                         |
                       v                          v                         v
           +-----------------------+  +-----------------------+  +----------------------+
           | Questionnaire Pipeline|  | Voice Modality        |  | Behavioral Pipeline  |
           | (Khwaish)             |  | Pipeline (Nandini)    |  | (Netal)              |
           +-----------------------+  +-----------------------+  +----------------------+
           | - Survey encoding     |  | - Audio preprocessing |  | - Log aggregation    |
           | - Subscale scoring    |  | - Acoustic extraction |  | - Temporal stats     |
           | - Questionnaire Model |  | - Voice ML Model      |  | - Behavioral Model   |
           +-----------+-----------+  +-----------+-----------+  +----------+-----------+
                       |                          |                         |
                       +--------------------+     |     +-------------------+
                                            |     |     |
                                            v     v     v
                             +--------------------------------------+
                             |       MULTIMODAL FUSION LAYER        |
                             |  - Late Decision-Level Fusion        |
                             |  - Cross-Modal Consistency Check     |
                             |  - Confidence / Uncertainty Estimate |
                             +------------------+-------------------+
                                                |
                                                v
                             +--------------------------------------+
                             |     UNIFIED ASSESSMENT OUTPUT        |
                             |  - Estimated Risk Level              |
                             |  - Modality Breakdown                |
                             |  - Explainability & Recommendations  |
                             +--------------------------------------+
```

---

## 1. Voice Modality Pipeline

*Primary Responsibility: Nandini (`feature/Nandini`)*

### A. End-to-End Processing Pipeline

The voice processing pipeline transforms raw spoken audio into a calibrated risk indicator through the following deterministic stages:

```
Raw Audio Recording (.wav)
   |
   v
Audio Loading (SoundFile / Librosa)
   |
   v
Mono Conversion (averaging multi-channel input to single-channel)
   |
   v
16 kHz Resampling (standardizing sample rate)
   |
   v
Amplitude Normalization (peak / RMS normalization to counteract volume differences)
   |
   v
Silence Handling (trimming leading/trailing silence, segmenting unvoiced pauses)
   |
   v
Acoustic Feature Extraction (frame-level spectral, prosodic, and temporal metrics)
   |
   v
Statistical Aggregation -> Fixed-Length Feature Vector (mean, std, min, max per feature)
   |
   v
Standard Feature Scaling (fitted on training partition)
   |
   v
Voice Machine Learning Model (Logistic Regression / Random Forest)
   |
   v
Voice Assessment Output (Risk Score, Predicted Class, Class Probabilities)
```

### B. Acoustic Features Extracted

1. **Mel-Frequency Cepstral Coefficients (MFCCs):**
   - 13 to 20 static coefficients representing the spectral envelope.
   - First-order ($\Delta$) and second-order ($\Delta\Delta$) delta coefficients to capture temporal trajectories.
2. **Fundamental Frequency / Pitch ($F_0$):**
   - Pitch mean, median, standard deviation, and range computed using autocorrelation or normalized cross-correlation.
3. **Energy & Intensity:**
   - Root Mean Square (RMS) energy mean and standard deviation, capturing vocal loudness dynamics.
4. **Spectral Descriptors:**
   - **Spectral Centroid:** Mean frequency weighted by energy (spectral brightness).
   - **Spectral Bandwidth:** Energy spread around the centroid.
   - **Spectral Rolloff:** 85th percentile frequency of energy distribution.
   - **Zero-Crossing Rate (ZCR):** Rate of signal sign changes, aiding voiced/unvoiced discrimination.
5. **Temporal & Pause Metrics:**
   - Total voiced duration, unvoiced duration, and pause-to-speech ratio.
6. **Exploratory Micro-Perturbation Features:**
   - Jitter (pitch perturbation), Shimmer (amplitude perturbation), and Harmonics-to-Noise Ratio (HNR).
   - *Note: These will be evaluated if supported reliably by the dataset SNR and audio toolkit.*

### C. Machine Learning Modeling Strategy

> [!IMPORTANT]
> **Baseline First — Deep Learning is NOT Mandatory:**
> The voice modality focuses initially on lightweight, explainable, classical machine learning models trained on fixed-length acoustic feature vectors.
>
> Deep learning is not required for the initial implementation.

- **Initial Baseline Models:**
  - **Logistic Regression Baseline:** Calibrated linear baseline providing directly interpretable coefficients and class probabilities.
  - **Random Forest Classifier:** Robust non-linear ensemble capable of capturing feature interactions with built-in Gini/permutation feature importance.
- **Future Exploratory Architectures (Optional):**
  - Gradient Boosted Trees (XGBoost / LightGBM)
  - Pretrained acoustic representations (e.g., wav2vec 2.0 / HuBERT embeddings)
  - 1D/2D CNNs on mel-spectrograms

### D. Conceptual Inference Interface

The voice model will expose a clean, decoupled Python interface for backend invocation:

```python
# Conceptual Signature (Planned)
def predict_voice(audio_file_path: str) -> dict:
    """
    Extracts acoustic features from an audio file and returns the voice risk assessment.
    
    Args:
        audio_file_path: Path to the input WAV audio file.
        
    Returns:
        dict: Standardized assessment dictionary containing:
            - "risk_score": float (estimated risk indicator, e.g. 0.0 to 1.0)
            - "predicted_class": str (predicted category)
            - "class_probabilities": dict (mapping class names to probabilities)
            - "top_features": list of dicts (key feature values for explainability)
    """
    pass
```

*Note: The exact label taxonomy is not yet finalized. No hardcoded label mappings will be established until the project label schema is approved.*

### E. Data Partitioning & Leakage Prevention Protocol
When multiple audio recordings exist from the same participant:
- **Participant-Level Splitting:** Data splitting into train, validation, and test sets **must occur strictly at the participant ID level** (GroupKFold / GroupShuffleSplit).
- Recordings from participant `P001` must never appear in both train and test splits simultaneously, preventing identity leakage and artificially inflated performance.

---

## 2. Questionnaire Model Pipeline *(Placeholder)*

*Primary Responsibility: Khwaish (`feature/Khwaish`)*

- **Input:** Tabular/JSON survey responses.
- **Pipeline:** Survey encoding -> scale validation -> feature weighting -> classifier/regressor.
- **Output:** Questionnaire risk score, class probabilities, contributing response items.
- **Status:** *To be specified and developed by Khwaish on `feature/Khwaish`.*

---

## 3. Behavioral Model Pipeline *(Placeholder)*

*Primary Responsibility: Netal (`feature/Netal`)*

- **Input:** Tabular/JSON behavioral logs (study patterns, sleep regularity, deadlines).
- **Pipeline:** Temporal aggregation -> lifestyle regularity metrics -> behavioral classifier.
- **Output:** Behavioral risk score, class probabilities, key behavioral flags.
- **Status:** *To be specified and developed by Netal on `feature/Netal`.*

---

## 4. Multimodal Fusion Layer *(Planned)*

*Primary Responsibility: Netal (`feature/Netal`) & Collaborative Team Effort*

- **Mechanism:** Late decision-level fusion synthesizing probability outputs from all three modality classifiers.
- **Key Outputs:**
  - Overall estimated risk score
  - Modality consistency metric (detecting alignment vs. discordance)
  - Assessment confidence score
  - Unified explainability summary
- **Status:** *Architecture in research phase.*
