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

Agreement is `1 - (maximum risk score - minimum risk score)` for two or more modalities, and 1.0 for one modality (reported with level `not_applicable`, since one source cannot agree or disagree). Final confidence is weighted confidence multiplied by `0.5 + 0.5 × agreement`. It is a project-level model-confidence indicator, not clinical certainty, diagnostic confidence, or a probability of a disorder.

Factors are weighted by their modality normalized weight and aggregated by matching feature names; they are model-associated indicators only and do not establish causation. The engine returns at most five. Overall data quality sums usable feature counts before calculating completeness.

With one modality, its risk score and confidence are preserved and agreement level is `not_applicable`, but the result is a single-modality assessment. With no usable modalities (or zero aggregate weight), risk score and level are `null` and status explains that fusion is unavailable.

```python
result = fuse_modalities(questionnaire=questionnaire_result, voice=voice_result)
print(result["risk_score"], result["normalized_weights"])
```

For example, output retains the shared prediction fields and adds fusion evidence: `{"modality": "multimodal_fusion", "risk_score": 0.69, "risk_level": "high", "modalities_used": ["questionnaire", "voice"], "agreement": {"score": 0.92, "level": "high", "sources": 2}}`.

The executable integration tests include explicitly labeled mock voice and behavioral contracts only to validate interface behavior. They are not model outputs, training data, research results, medical advice, or clinical validation.

## Unavailable-modality policy and voice

* The assessment API passes `None` for any modality that was not provided, has no trained model, was invalid, or failed; the per-modality reason is returned in `modality_status`. Nothing is zero-filled or guessed.
* Voice produces this contract through `src.ai_model.voice.predict.predict_voice_from_features` (extracted features) or `predict_voice_contract_from_audio` (WAV file). Both raise `VoiceModelNotTrainedError` until a model exists in `models/voice/` (`selected_model.json` + `<name>.joblib`, or `voice_model.joblib`). The voice model must predict exactly the classes `low`, `moderate`, `high`.
* The older `VoiceFusionPayload` (`src/ai_model/voice/fusion_interface.py`, class probabilities) is a different, voice-internal shape and is **not** accepted by `fuse_modalities`.
* Tests: `tests/test_fusion_engine.py`, `tests/test_voice_contract.py`, `tests/test_assessment_api.py`.

## Cross-modal consistency and possible masked stress

`fuse_modalities` also returns `consistency` (`calculate_consistency`), which only applies with two or more usable sources:

| Field | Meaning |
|---|---|
| `applicable` / `status` | `false` with `no_sources` or `single_source`; otherwise `consistent` (gap < 0.20), `moderate_difference` (< 0.40) or `conflict` (>= 0.40), where gap = largest minus smallest source risk score |
| `pattern` | `masked_stress_possible`: the questionnaire (self-report) is low/moderate and the average of the behavioral/voice sources is at least 0.30 higher. `self_report_higher`: the reverse. Otherwise `none` |
| `sources` | each source's score and level, for display |

Thresholds are project-defined, not clinical, and a pattern is a prompt to reflect on, never a finding. It cannot appear from one source. In practice the behavioral model is trained on synthetic data and voice has no stress model, so the flag is only as meaningful as those inputs.

## Personal explanations (`explanations.py`)

`key_factors` are model-wide importances (the same for every user), so on their own they cannot say what matters for one person. `explanations.py` adds, without changing any model output:

* `describe_feature`: a plain-language label (and related themes) for every questionnaire, daily-habit and voice feature, e.g. `Q9_encoded` -> "Stress affecting your sleep".
* `personal_insights`: adverse areas found in the person's OWN answers (sleep, workload, focus, screen time, routine ...), with their answers quoted back.
* `build_recommendations`: ranks those areas by severity (70%) and the model's reliance on that area (30%, from the fused `key_factors`), attaches general self-care advice and the reason, and adds a masked-stress prompt when flagged. Sleep advice therefore appears only when sleep answers are adverse.

The API returns them as `insights` and `recommendations`, and adds a `label` to each key factor. Tests: `tests/test_explanations.py`, `tests/test_fusion_engine.py`.
