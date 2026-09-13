# Architecture Overview

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This document outlines the system architecture for the **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

---

## High-Level System Architecture

The system is structured as a modular, three-tier pipeline:

1. **Modality Ingestion & Preprocessing:** Receives student input across three distinct channels (Questionnaire, Voice, Behavioral).
2. **Specialized Modality Modeling:** Independent evaluation of each modality using tailored feature extraction and classification techniques.
3. **Multimodal Fusion & Assessment Generation:** Synthesis of individual modality outputs into a unified assessment featuring cross-modal consistency, confidence scoring, explainable factors, and non-clinical recommendations.

```
+-------------------------------------------------------------------------------+
|                               STUDENT INTERFACE                               |
|   - Questionnaire Submission   - Voice Recording (30-60s)   - Behavioral Logs |
+---------------------------------------+---------------------------------------+
                                        |
       +--------------------------------+-------------------------------+
       |                                |                               |
       v                                v                               v
+--------------+               +------------------+            +----------------+
| Questionnaire|               | Voice Modality   |            | Behavioral     |
| Modality     |               | (Nandini)        |            | Modality       |
| (Khwaish)    |               |                  |            | (Netal)        |
+--------------+               +------------------+            +----------------+
| Scoring &    |               | Audio Loading    |            | Log Parsing &  |
| Encoding     |               | Resample (16kHz) |            | Aggregation    |
|              |               | Normalization    |            |                |
|              |               | Silence Trim     |            |                |
|              |               | Feature Extract  |            |                |
+-------+------+               +--------+---------+            +-------+--------+
        |                               |                              |
        v                               v                              v
+--------------+               +------------------+            +----------------+
| Questionnaire|               | Voice Model      |            | Behavioral     |
| Model        |               | (e.g., Random    |            | Model          |
|              |               | Forest / LR)     |            |                |
+-------+------+               +--------+---------+            +-------+--------+
        |                               |                              |
        | [Risk Score / Probs]          | [Risk Score / Probs]         | [Risk Score / Probs]
        +-----------------------+-------+------------------------------+
                                |
                                v
             +--------------------------------------+
             |       MULTIMODAL FUSION LAYER        |
             |       (Planned - Not Finalized)      |
             |   - Decision-level / Late Fusion     |
             |   - Cross-Modal Consistency Check    |
             |   - Confidence & Uncertainty Scoring |
             +------------------+-------------------+
                                |
                                v
             +--------------------------------------+
             |          OVERALL ASSESSMENT          |
             |  - Overall Estimated Stress/Risk     |
             |  - Modality Breakdown Scores         |
             |  - Consistency / Disagreement Flag   |
             |  - Assessment Confidence             |
             |  - Contributing Factors (SHAP/Feats) |
             |  - Non-Clinical Recommendations      |
             +--------------------------------------+
```

---

## Component Details

### 1. Modality Processing Pipelines

#### A. Questionnaire Modality *(Responsibility: Khwaish - `feature/Khwaish`)*
- **Input:** Standardized student self-report responses regarding academic workload, perceived stress, and emotional state.
- **Processing:** Numerical encoding, subscale aggregation, and normalization.
- **Model:** Classification / regression model producing an estimated subjective stress score and class probabilities.
- **Status:** *Planned.*

#### B. Voice Modality *(Responsibility: Nandini - `feature/Nandini`)*
- **Input:** Short audio recording (30–60 seconds, WAV format, 16 kHz, mono) answering a standardized academic prompt.
- **Processing:**
  - Audio normalization and silence filtering.
  - Acoustic feature extraction: Mel-Frequency Cepstral Coefficients (MFCCs), fundamental frequency (pitch/F0), energy (RMS), spectral characteristics (centroid, rolloff, bandwidth), and zero-crossing rate.
  - Formation of a fixed-length acoustic feature vector.
- **Model:** Classical ML baselines (Logistic Regression, Random Forest) with planned exploration of advanced approaches.
- **Output:** Voice risk score, predicted class, and class probability distribution.
- **Status:** *Planned / in active design.*

#### C. Behavioral Modality *(Responsibility: Netal - `feature/Netal`)*
- **Input:** Contextual lifestyle and academic indicators (study duration, sleep schedule regularity, assignment deadlines, breaks).
- **Processing:** Time-window aggregation, trend calculation, and feature vector assembly.
- **Model:** Behavioral classifier generating a behavioral risk indicator.
- **Status:** *Planned.*

---

### 2. Multimodal Fusion *(Planned - Architecture Under Research)*

Multimodal fusion combines the three independent modality outputs. The exact fusion algorithm is currently **not finalized** and will be selected following empirical experimentation.

Candidate approaches under consideration include:
- **Decision-Level (Late) Fusion:** Weighted averaging, rule-based ensemble, or meta-classifier operating on modality probability distributions.
- **Hybrid Fusion:** Combining extracted intermediate features with decision scores.

Key requirements for the fusion layer:
1. **Modality Consistency / Disagreement Analysis:**
   - Detects concordance (e.g., all three modalities point to elevated stress).
   - Flags discordance (e.g., student reports low stress on questionnaire, but voice and behavioral indicators reveal significant strain).
2. **Confidence Estimation:**
   - Derives a composite confidence score based on individual model certainty and cross-modal agreement.
3. **Explainability:**
   - Identifies which modality (and which specific features) contributed most significantly to the final assessment.

---

### 3. Supportive Recommendation Engine *(Planned)*

Based on the overall assessment, modality breakdown, and identified contributing factors, the system generates tailored, non-clinical recommendations:
- **Academic Pacing:** Suggestions for scheduling and task breakdown when academic behavior is a dominant stressor.
- **Rest & Routine:** Sleep hygiene and active break suggestions when behavioral irregularities are elevated.
- **Grounding Exercises:** Brief breathing or mindfulness prompts when acoustic/expressive tension is observed.
- **Campus Resources:** Direct links to institutional counseling, tutoring, and peer support centers.

---

## Data Flow Summary

| Step | Component | Input | Output | Status |
|---|---|---|---|---|
| 1 | Ingestion Layer | User responses, 30-60s WAV audio, lifestyle metrics | Cleaned, validated modality payloads | Planned |
| 2 | Questionnaire Pipeline | Encoded survey answers | Subjective stress score & probabilities | Planned |
| 3 | Voice Pipeline | 16 kHz mono WAV file | Acoustic feature vector -> Voice risk score & probabilities | Planned |
| 4 | Behavioral Pipeline | Tabular behavioral metrics | Behavioral risk score & probabilities | Planned |
| 5 | Fusion Layer | 3x Modality probability vectors | Fused risk score, consistency metric, confidence value | Planned |
| 6 | Explainability & Advice | Fused outputs + feature importances | Explainable breakdown + non-clinical supportive advice | Planned |

---

## Architectural Principles

1. **Non-Invasive by Design:** Excludes video, facial recognition, and physiological wearable sensors to minimize user burden and maximize privacy.
2. **Explainability Over Black-Box:** Prioritizes interpretable features and clear modality breakdowns over opaque predictions.
3. **Graceful Handling of Disagreement:** Treats cross-modal disagreement as meaningful data rather than simple noise.
4. **Strict Non-Clinical Framing:** All components report well-being and stress indicators without medical or psychiatric diagnostic claims.
