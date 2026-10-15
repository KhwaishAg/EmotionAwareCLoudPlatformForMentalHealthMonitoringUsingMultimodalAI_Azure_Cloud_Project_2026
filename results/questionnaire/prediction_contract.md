# Questionnaire prediction fusion contract

Member 3 can call `predict_questionnaire(response)` with a Python dictionary containing raw `Q1` through `Q20` responses. `Timestamp` and `Q21` through `Q25` are accepted; `Q21`, `Q22`, `Q23`, and `Q25` may be omitted. The function returns a JSON-serializable dictionary and does not read the training CSV.

The response is transformed in this order: semantic ordinal encoding; Q10/Q11 multi-hot encoding; Q1/Q2/Q25 categorical encoding; derived feature engineering; then selection of the exact 75 saved predictor names. The saved sklearn pipeline performs its own fitted imputation and preprocessing. It is never refit during inference.

`Q4` may be supplied for schema compatibility, but neither `Q4` nor `Q4_encoded` is a predictor. `stress_level` and `stress_level_encoded` are also excluded, preventing target leakage.

```json
{
  "modality": "questionnaire",
  "risk_score": 0.0,
  "risk_level": "low",
  "confidence": 0.0,
  "available_features": 0,
  "total_features": 75,
  "key_factors": [{"feature": "string", "contribution": 0.0, "direction": "model_associated"}],
  "data_quality": {"missing_features": 0, "completeness": 0.0}
}
```

`risk_score = 0 × prob_low + 0.5 × prob_moderate + 1 × prob_high`; it is a continuous project-level stress-risk representation, not a clinical probability. `risk_level` is the classifier prediction, not a threshold applied to the score. Confidence follows Step 13: `0.5 × max_probability + 0.5 × (max_probability − second_probability)`. It is a model-confidence heuristic, not clinical certainty.

`available_features` counts non-missing selected predictors before pipeline imputation; `completeness = available_features / total_features`. This measures input data quality, not confidence. Up to five key factors use saved model-association importance normalized to 0–1. Their directions are only semantic annotations where justified; feature importance does not establish causation. This component is non-clinical and is not medical advice or diagnosis.
