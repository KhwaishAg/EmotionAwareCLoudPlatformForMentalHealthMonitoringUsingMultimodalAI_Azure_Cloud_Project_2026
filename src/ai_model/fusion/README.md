# Multimodal fusion engine

The fusion engine combines modality-level model outputs, never raw questionnaire, voice, or behavioral features. The three accepted modalities are `questionnaire`, `voice`, and `behavioral`; each uses the common prediction contract.

Call `fuse_modalities(questionnaire=None, voice=None, behavioral=None)`. A present result is validated strictly. A missing result (`None`) is ignored, not treated as zero risk. A modality is usable only with positive available and total features and positive completeness.

Each input has the same modality-independent shape:

```json
{
  "modality": "voice",
  "risk_score": 0.72,
  "risk_level": "high",
  "confidence": 0.80,
  "available_features": 8,
  "total_features": 10,
  "key_factors": [{"feature": "stress_related_prosody", "contribution": 0.8, "direction": "model_associated"}],
  "data_quality": {"missing_features": 2, "completeness": 0.8}
}
```

Each usable modality receives the dynamic weight `confidence × completeness`, normalized across usable modalities. Fused risk is the normalized weighted average of modality risk scores. Project thresholds classify it as low (<0.34), moderate (0.34–<0.67), or high (>=0.67); these are not clinical thresholds.

Agreement is `1 - (maximum risk score - minimum risk score)` for two or more modalities, and 1.0 for one modality. Final confidence is weighted confidence multiplied by `0.5 + 0.5 × agreement`. It is a project-level model-confidence indicator, not clinical certainty, diagnostic confidence, or a probability of a disorder.

Factors are weighted by their modality normalized weight and aggregated by matching feature names; they are model-associated indicators only and do not establish causation. The engine returns at most five. Overall data quality sums usable feature counts before calculating completeness.

With one modality, its risk score and confidence are preserved and agreement is high, but the result is a single-modality assessment. With no usable modalities (or zero aggregate weight), risk score and level are `null` and status explains that fusion is unavailable.

```python
result = fuse_modalities(questionnaire=questionnaire_result, voice=voice_result)
print(result["risk_score"], result["normalized_weights"])
```

For example, output retains the shared prediction fields and adds fusion evidence: `{"modality": "multimodal_fusion", "risk_score": 0.69, "risk_level": "high", "modalities_used": ["questionnaire", "voice"], "agreement": {"score": 0.92, "level": "high"}}`.

The executable integration tests include explicitly labeled mock voice and behavioral contracts only to validate interface behavior. They are not model outputs, training data, research results, medical advice, or clinical validation.
