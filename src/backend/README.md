# Backend Architecture & API Specifications

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This document specifies the planned backend architecture, endpoint definitions, and assessment orchestration workflows for the **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

> [!IMPORTANT]
> **Implementation Status: Planned**
> The endpoints, schemas, and services described herein are architectural specifications. No backend source code, server scripts, or live endpoints have been implemented in this documentation pass.

---

## Planned Backend Responsibilities

The backend service serves as the central orchestration hub connecting student client interfaces, individual modality inference models, the multimodal fusion engine, and the recommendation generation service.

Key responsibilities include:
1. **Multi-Channel Data Ingestion:**
   - Ingest structured questionnaire submissions.
   - Ingest tabular / time-windowed behavioral metrics.
   - Ingest uploaded audio speech recordings (`.wav`, 16 kHz mono).
2. **Modality Model Execution:**
   - Invoke the Questionnaire Model on validated survey responses.
   - Invoke the Voice Model feature extraction and classification pipeline on uploaded audio.
   - Invoke the Behavioral Model on aggregated lifestyle indicators.
3. **Multimodal Fusion Orchestration:**
   - Collect intermediate predictions, probabilities, and confidence metrics from each modality model.
   - Pass modality outputs to the Late Fusion Layer.
4. **Assessment Synthesis & Delivery:**
   - Compute cross-modal consistency (concordance vs. discordance).
   - Calculate overall estimated stress/risk indicators and composite confidence scores.
   - Compile explainability breakdowns (primary contributing factors).
   - Generate tailored, non-clinical supportive recommendations.

---

## Planned API Specifications (Conceptual)

The planned backend will expose RESTful endpoints designed for asynchronous or synchronous assessment workflows.

### 1. Voice Modality Ingestion & Analysis *(Planned)*

```http
POST /api/v1/assessment/voice
Content-Type: multipart/form-data
```

#### Request Parameters:
- `participant_id` (string, required): Anonymous participant identifier (e.g., `"P001"`).
- `audio_file` (binary, required): Recorded audio file (`.wav` format, 30–60 seconds, 16 kHz).

#### Conceptual Response Schema (Planned):
```json
{
  "status": "success",
  "modality": "voice",
  "participant_id": "P001",
  "voice_risk_score": 0.61,
  "predicted_class": "moderate_risk",
  "class_probabilities": {
    "low_risk": 0.20,
    "moderate_risk": 0.61,
    "high_risk": 0.19
  },
  "extracted_features_summary": {
    "pitch_mean_hz": 192.4,
    "pitch_std_hz": 31.8,
    "energy_rms_mean": 0.042,
    "spectral_centroid_hz": 2100.5,
    "speech_duration_seconds": 42.6
  }
}
```

---

### 2. Full Multimodal Assessment Submission *(Planned)*

```http
POST /api/v1/assessment/full
Content-Type: multipart/form-data
```

#### Request Payload:
- `participant_id`: `"P001"`
- `questionnaire_payload`: JSON string of survey responses
- `behavioral_payload`: JSON string of lifestyle/study log metrics
- `voice_audio`: Uploaded `.wav` file

#### Conceptual Response Schema (Planned):
```json
{
  "assessment_id": "ASM-2026-9021",
  "participant_id": "P001",
  "timestamp": "2026-09-13T12:00:00Z",
  "overall_assessment": {
    "estimated_risk_level": "moderate_risk",
    "fused_risk_score": 0.58,
    "confidence_score": 0.82
  },
  "modality_breakdown": {
    "questionnaire": {"score": 0.45, "class": "low_risk"},
    "voice": {"score": 0.61, "class": "moderate_risk"},
    "behavioral": {"score": 0.68, "class": "moderate_risk"}
  },
  "consistency_analysis": {
    "consistency_status": "discordance_detected",
    "details": "Self-reported questionnaire stress is low, whereas voice acoustic markers and behavioral habits indicate moderate strain."
  },
  "explainability": {
    "primary_driver": "behavioral",
    "top_contributing_factors": [
      "Inconsistent sleep schedules over the last 72 hours",
      "Elevated vocal pitch variability during study workload reflection"
    ]
  },
  "supportive_recommendations": [
    "Prioritize establishing a consistent sleep-wake schedule this week",
    "Try a 10-minute structured study break between major reading tasks",
    "Campus academic support services are available for exam planning assistance"
  ]
}
```

---

## Architecture & Technology Considerations

- **Framework Candidates:** FastAPI / Flask (Python-based) to enable native, in-process execution of Python ML models and scientific audio libraries (Librosa, SoundFile).
- **Asynchronous Task Processing:** Optional background queue (e.g., Celery / Redis) for longer audio processing tasks if request volume demands decoupling.
- **Security:** Request validation using Pydantic models, rate-limiting, and strict CORS policies.
