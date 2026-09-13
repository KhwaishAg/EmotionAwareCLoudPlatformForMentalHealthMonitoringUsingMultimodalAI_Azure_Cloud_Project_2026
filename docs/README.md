# Documentation Index & Governance Guide

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

Welcome to the project documentation index for the **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

This document serves as the central directory for all technical, scientific, and architectural documentation across the repository, alongside project governance guidelines and a standardized terminology glossary.

---

## Documentation Navigation Index

| Section | Location | Description |
|---|---|---|
| **Root Overview** | [`README.md`](../README.md) | Official project title, scope, 3-modality overview, team mapping, and novelty |
| **System Architecture** | [`architecture/README.md`](../architecture/README.md) | Pipeline diagrams, modality processing, fusion design, and data flow |
| **Dataset Specifications** | [`dataset/README.md`](../dataset/README.md) | Audio recording specs, prompt design, public datasets, and privacy policies |
| **Literature Survey** | [`literature_survey/README.md`](../literature_survey/README.md) | State-of-the-art review across speech emotion/stress, questionnaires, and fusion |
| **Academic References** | [`references/README.md`](../references/README.md) | Bibliography, citation standards, and verified research foundations |
| **Experimental Results** | [`results/README.md`](../results/README.md) | Evaluation metrics, benchmark protocols, and performance tracking formats |
| **Source Code Overview** | [`src/README.md`](../src/README.md) | Directory structure and responsibilities for all code components |
| **AI Models** | [`src/ai_model/README.md`](../src/ai_model/README.md) | Voice feature extraction pipeline, ML classifiers, and fusion architecture |
| **Azure Cloud Services** | [`src/azure/README.md`](../src/azure/README.md) | Planned cloud storage, model management, and secrets policies |
| **Backend Service** | [`src/backend/README.md`](../src/backend/README.md) | Planned REST API endpoints and assessment orchestration logic |
| **Frontend UI** | [`src/frontend/README.md`](../src/frontend/README.md) | Planned audio recording UI, questionnaire interface, and results dashboard |

---

## Terminology Glossary

To ensure clarity and scientific rigor across all codebase documentation and communications, the following definitions are established:

### Core Concepts

- **Stress/Risk Indicator:** An estimated, quantitative metric reflecting the probability or intensity of academic-related stress or well-being vulnerability, derived from computational analysis.
- **Questionnaire Indicator:** A subjective, self-reported metric derived from standardized survey responses addressing perceived workload, stress, and mood.
- **Voice-Derived Indicator:** An expressive, acoustic metric extracted from short audio recordings (e.g., changes in fundamental frequency, energy variability, spectral distribution).
- **Behavioral Indicator:** A contextual metric calculated from daily routine patterns (e.g., study habits, sleep duration/regularity, deadline density).
- **Multimodal Fusion:** The computational integration of outputs or features from multiple independent modalities (questionnaire, voice, behavioral) into a cohesive assessment.
- **Modality Consistency (Concordance):** The degree to which independent modality outputs align in their stress assessment (e.g., all three modalities estimate elevated stress).
- **Modality Disagreement (Discordance):** Divergence between modality indicators (e.g., low questionnaire stress paired with high acoustic or behavioral stress), serving as a vital informative indicator for masked or unacknowledged tension.
- **Confidence:** A statistical or heuristic measure reflecting the certainty of the fused assessment, informed by model output probabilities and cross-modal agreement.
- **Explainability:** The transparent attribution of the final assessment to specific modalities and underlying feature contributions (e.g., elevated vocal pitch variation + erratic sleep schedules).
- **Early-Intervention Support:** Proactive, non-clinical guidance, self-care prompts, and institutional resource referrals delivered before stress escalates to severe distress.

---

## Ethical Guidelines & Language Policy

### Non-Clinical Terminology Policy

Because this system is an academic research platform and **not a certified medical diagnostic device**, all contributors must strictly adhere to the following language standards in documentation, code comments, and UI text:

```
+-------------------------------------------+-------------------------------------------+
| PREFERRED TERMINOLOGY (USE)               | PROHIBITED TERMINOLOGY (DO NOT USE)       |
+-------------------------------------------+-------------------------------------------+
| "stress/risk indicators"                  | "diagnoses depression"                    |
| "estimated risk / stress level"           | "diagnoses anxiety"                       |
| "well-being indicators"                   | "diagnoses mental illness"                |
| "voice-derived indicators"                | "clinical detection of disorder"          |
| "behavioral stress indicators"            | "medical diagnostic tool"                 |
| "multimodal assessment"                   | "replaces psychologist / doctor"          |
| "early-intervention support"              | "prescribes treatment"                    |
| "supportive recommendations"              | "medical prognosis"                       |
+-------------------------------------------+-------------------------------------------+
```

### Participant Ethics & Data Protection

1. **Voluntary Participation:** Engagement with the platform is entirely voluntary, and participants may withdraw without penalty.
2. **Strict Anonymization:** Data collection uses pseudorandom participant codes (`P001`, `P002`, ...). Directly identifying records must never be stored alongside acoustic or behavioral logs.
3. **No Private Data in Git:** Source control repositories must remain entirely free of participant audio, completed surveys, and personally identifiable records.
4. **Transparent Communication:** The user interface must prominently inform students that the assessment is non-clinical and intended solely for supportive self-reflection.
