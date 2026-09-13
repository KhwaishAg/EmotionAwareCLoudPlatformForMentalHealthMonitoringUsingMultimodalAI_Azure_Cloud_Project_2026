# Experimental Results & Evaluation Framework

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This directory documents the evaluation framework, metric definitions, experiment logging protocols, and expected result schemas for the **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

> [!NOTE]
> **No Synthetic Results:**
> This repository does not contain fabricated result numbers or simulated benchmarks. All actual performance figures, confusion matrices, and model comparison tables will be logged here as models are trained and validated on experimental datasets.

---

## Evaluation Metrics

To rigorously assess individual modality models and multimodal fusion architectures, the following standard classification and regression metrics will be utilized:

| Metric | Definition & Purpose | Key Applicability |
|---|---|---|
| **Accuracy** | Overall proportion of correct predictions across all classes | General performance indicator; evaluated alongside balanced metrics |
| **Precision** | Ratio of true positive predictions to total predicted positives | Critical for minimizing false alarms in elevated-risk detection |
| **Recall (Sensitivity)** | Ratio of true positive predictions to all actual positive cases | Critical for ensuring students experiencing stress indicators are not missed |
| **F1-Score** | Harmonic mean of Precision and Recall | Balanced metric for per-class performance |
| **Macro F1-Score** | Unweighted average of F1-scores across all classes | Essential metric when evaluating potentially imbalanced stress/risk classes |
| **Confusion Matrix** | Tabular layout showing true vs. predicted class distributions | Pinpoints specific misclassification patterns (e.g., low vs. moderate stress) |
| **ROC-AUC / PR-AUC** | Area under the Receiver Operating Characteristic / Precision-Recall curve | Evaluates probabilistic discrimination across varying decision thresholds |

---

## Expected Result Schemas

### 1. Per-Modality Model Results

Each independent modality model (Questionnaire, Voice, Behavioral) produces a standardized output structure for every evaluation instance:

```json
{
  "modality": "voice",
  "participant_id": "P001",
  "predicted_class": "moderate_risk",
  "risk_score": 0.62,
  "class_probabilities": {
    "low_risk": 0.18,
    "moderate_risk": 0.62,
    "high_risk": 0.20
  },
  "top_contributing_features": [
    {"feature": "pitch_std", "value": 34.2, "relative_importance": 0.28},
    {"feature": "spectral_centroid_mean", "value": 2150.4, "relative_importance": 0.22},
    {"feature": "mfcc_2_mean", "value": -14.8, "relative_importance": 0.18}
  ]
}
```

*Note: Class labels shown above are illustrative placeholders; exact final label taxonomy will be established during experimental design.*

### 2. Multimodal Fusion Results

The multimodal fusion layer synthesizes the three modality outputs into a unified assessment payload:

```json
{
  "assessment_id": "ASM-2026-0001",
  "participant_id": "P001",
  "timestamp": "2026-09-13T12:00:00Z",
  "overall_estimated_risk": "moderate_risk",
  "fused_risk_score": 0.58,
  "modality_scores": {
    "questionnaire_risk_score": 0.45,
    "voice_risk_score": 0.62,
    "behavioral_risk_score": 0.67
  },
  "modality_consistency": {
    "status": "partial_concordance",
    "agreement_score": 0.72,
    "flagged_discrepancy": "Questionnaire self-report indicates lower stress than voice and behavioral markers"
  },
  "confidence": {
    "composite_confidence": 0.81,
    "rating": "high"
  },
  "explainability": {
    "primary_driver_modality": "behavioral",
    "contributing_factors": [
      "Irregular sleep patterns over previous 3 days",
      "Elevated vocal pitch variability during study task reflection"
    ]
  },
  "recommendations": [
    "Consider establishing a consistent sleep-wake window",
    "Explore scheduled study intervals (Pomodoro method) for upcoming deadlines",
    "Access campus peer wellness coaching resources"
  ]
}
```

---

## Experiment Logging Protocol

When conducting experiments, team members will structure logs under `results/` following this layout:

```
results/
|-- README.md                       # This evaluation framework guide
|-- voice_modality/                 # Member 2 (Nandini) experimental logs
|   |-- baseline_logistic_regression/
|   +-- baseline_random_forest/
|-- questionnaire_modality/         # Member 1 (Khwaish) experimental logs
|-- behavioral_modality/            # Member 3 (Netal) experimental logs
+-- multimodal_fusion/              # Joint fusion benchmarking logs
```

Each experimental subfolder will contain:
- `metrics.json` — Quantified performance metrics on validation/test splits.
- `confusion_matrix.png` — Visualized confusion matrix.
- `experiment_config.json` — Hyperparameters, feature configurations, and data split metadata.
