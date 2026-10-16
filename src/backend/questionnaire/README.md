# Questionnaire prediction API

This FastAPI adapter exposes the validated questionnaire inference component
without retraining, changing preprocessing, logging responses, or connecting
to Azure services.

> **Status:** Locally implemented and validated. 4/4 unit tests pass.
> Azure deployment is not yet configured for this HTTP service; the Azure
> inference path uses `score.py` via a managed online endpoint instead.

---

## Running locally

Install the API dependencies from this directory, plus the project ML dependencies from the repository root:

```powershell
# From the repository root
pip install fastapi==0.135.3 uvicorn==0.43.0
pip install pandas==2.3.3 numpy==2.4.2 scikit-learn==1.8.0 joblib==1.5.3 matplotlib==3.10.8

uvicorn src.backend.questionnaire.api:app --reload
```

---

## Endpoint

### `POST /predict/questionnaire`

**Content-Type:** `application/json`

**Request body:**
```json
{
  "responses": {
    "Q1":  "4th Year",
    "Q2":  "Regular academic semester; Working on academic projects",
    "Q3":  "4–6 hours",
    "Q4":  "4 — Very",
    "Q5":  "Often",
    "Q6":  "Sometimes",
    "Q7":  "Very difficult",
    "Q8":  "Often",
    "Q9":  "Sometimes",
    "Q10": "Academic workload; Projects",
    "Q11": "Anxious; Tired",
    "Q12": "4",
    "Q13": "3",
    "Q14": "15–30 minutes",
    "Q15": "Sometimes",
    "Q16": "Within an hour",
    "Q17": "Often",
    "Q18": "Sometimes",
    "Q19": "Rarely",
    "Q20": "Sometimes",
    "Q21": "Academic workload",
    "Q22": "Overwhelmed",
    "Q23": "Rest",
    "Q24": "Yes",
    "Q25": "Academic workload"
  }
}
```

**Q1–Q20 are required. Q21–Q25 and Timestamp are optional.**

**Successful response (`200 OK`):**
```json
{
  "modality":           "questionnaire",
  "risk_score":         0.72,
  "risk_level":         "high",
  "confidence":         0.41,
  "available_features": 76,
  "total_features":     76,
  "key_factors": [
    {"feature": "stress_impact_dimension", "contribution": 1.0,  "direction": "increases_risk"},
    {"feature": "overwhelm_dimension",     "contribution": 0.81, "direction": "increases_risk"},
    {"feature": "Q9_encoded",             "contribution": 0.64, "direction": "increases_risk"},
    {"feature": "Q12_encoded",            "contribution": 0.52, "direction": "model_associated"},
    {"feature": "stress_behavior_dimension","contribution": 0.44, "direction": "increases_risk"}
  ],
  "data_quality": {
    "missing_features": 0,
    "completeness":     1.0
  }
}
```

---

## Error responses

| Condition | HTTP status |
|---|---|
| Extra/unknown field in request body | `422 Unprocessable Entity` |
| Missing required questionnaire fields (Q1–Q20) | `422 Unprocessable Entity` |
| Model artifact files not found | `503 Service Unavailable` |
| Unexpected internal error | `500 Internal Server Error` |

Stack traces are never returned in error responses.

---

## Non-clinical disclaimer

- `risk_level` and `risk_score` are project-defined categories (low / moderate / high) derived from Q4 self-report.
- `confidence` is a model heuristic (not clinical certainty).
- `key_factors` list model feature associations; they are not causal explanations.
- Q4 is accepted in the request body as a respondent data field but is **never used as a model predictor**.
- This is a non-clinical assessment tool; it does not constitute a medical diagnosis.

---

## Running the unit tests

```powershell
# From the repository root
python -m pytest src/backend/questionnaire/test_api.py -v
```

Expected output: 4 passed.

The test fixtures are development-only example inputs. No participant-level questionnaire data is used in tests.
