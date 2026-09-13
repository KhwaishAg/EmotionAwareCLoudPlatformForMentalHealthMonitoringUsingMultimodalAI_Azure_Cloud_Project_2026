# Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

## Project Overview

**Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI** is an academic research and engineering project aimed at developing a **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

The platform is designed to provide a low-burden, explainable, and multi-perspective assessment of student stress and well-being indicators by integrating three complementary, non-invasive modalities:

1. **Questionnaire** — Subjective / self-reported perspective (academic stress, perceived pressure, emotional self-assessment)
2. **Voice** — Expressive / acoustic indicators derived from short standardized voice recordings
3. **Behavioral Information** — Contextual, lifestyle, and academic behavioral indicators (study patterns, sleep regularity, workload trends)

Outputs from these three modalities are synthesized through **multimodal fusion** to generate an estimated risk/stress level along with cross-modal consistency analysis, confidence estimates, factor explainability, and personalized supportive recommendations.

---

## Problem Statement

College students frequently experience acute and chronic stress stemming from academic workload, examinations, deadlines, and lifestyle adjustments. Conventional assessment methods face notable limitations:

- **Single-modality bias:** Relying solely on questionnaires introduces subjective reporting bias, social desirability bias, or retrospective recall errors.
- **High user burden:** Complex diagnostic interviews or invasive wearable sensor protocols can create friction, leading to low adherence.
- **Lack of explainability:** Black-box prediction outputs that state a binary outcome (e.g., "You are stressed") offer little actionable insight and foster distrust.
- **Inability to capture discordance:** When a student reports feeling "fine" on a questionnaire but exhibits significant acoustic or behavioral stress indicators, single-modality systems fail to detect the underlying tension.

This project addresses these challenges by combining three non-invasive modalities into an integrated, explainable assessment framework.

---

## Current Project Scope

The implementation scope is explicitly focused and bounded:

- **Included Modalities:** Exactly three modalities:
  1. Questionnaire
  2. Voice (short audio recordings)
  3. Behavioral information (academic & lifestyle context)
- **Target Audience:** College / university students.
- **Output Objectives:** Estimated stress/risk level, modality breakdown, modality consistency/disagreement, confidence metric, contributing factors (explainability), and supportive early-intervention recommendations.

### Out of Scope (Explicit Exclusions)

The following components are **NOT** part of the current implementation:
- Facial image or video analysis
- Facial emotion recognition (FER) / facial expression analysis
- Wearable or physiological sensors (e.g., smartwatches, ECG, PPG, galvanic skin response)
- Mandatory biometric hardware
- Separate natural-language text processing modality (beyond structured questionnaire responses)

---

## The Three Modalities

```
+-------------------+---------------------------------------------------------+
| Modality          | Description & Perspective                               |
+-------------------+---------------------------------------------------------+
| 1. Questionnaire  | Subjective / Self-Reported Perspective                  |
|                   | Captures perceived stress, academic load evaluation,     |
|                   | and self-reported emotional states via structured items.|
+-------------------+---------------------------------------------------------+
| 2. Voice          | Acoustic / Expressive Indicator Perspective              |
|                   | Captures acoustic properties (pitch, energy, spectral   |
|                   | dynamics, MFCCs) from a short 30-60s spoken response    |
|                   | to a standardized prompt.                               |
+-------------------+---------------------------------------------------------+
| 3. Behavioral     | Contextual / Lifestyle Perspective                      |
|                   | Captures academic habits, study duration, sleep trends,  |
|                   | deadline proximity, and routine regularity indicators.   |
+-------------------+---------------------------------------------------------+
```

---

## High-Level Architecture

The system follows a modular pipeline where each modality is analyzed by a specialized model before multimodal fusion and assessment generation:

```
Student
   |
   +---------------- Questionnaire
   |                       |
   |                       v
   |                 Questionnaire Model
   |
   +---------------- Voice (Short Recording)
   |                       |
   |                       v
   |                    Voice Model
   |
   +---------------- Behavioral Information
                           |
                           v
                    Behavioral Model
                           |
                           v
                    Multimodal Fusion
                           |
                           v
                    Overall Assessment
                           |
             +-------------+-------------+
             |             |             |
        Consistency    Confidence   Explainability
                                           |
                                           v
                               Personalized Supportive
                                   Recommendations
```

---

## Team Ownership & Branch Strategy

The project is developed by a 3-member team with clear modular responsibilities across dedicated feature branches:

| Member | Branch | Primary Responsibility | Scope Details |
|---|---|---|---|
| **Khwaish** (Member 1) | `feature/Khwaish` | **Questionnaire Modality** | Questionnaire design, data representation, questionnaire ML model |
| **Nandini** (Member 2) | `feature/Nandini` | **Voice Modality** | Audio preprocessing pipeline, acoustic feature extraction, voice ML model |
| **Netal** (Member 3) | `feature/Netal` | **Behavioral Modality** & Integration | Behavioral feature engineering, behavioral model, multimodal fusion, backend & deployment integration |

### Git Branch Structure

- `main` — Production-ready, stable releases.
- `develop` — Shared integration branch.
- `feature/Khwaish` — Questionnaire modality development.
- `feature/Nandini` — Voice modality development.
- `feature/Netal` — Behavioral modality, fusion, and integration development.

*Note: All current work for Member 2 remains strictly on `feature/Nandini`.*

---

## Planned Novelty

The project investigates several novel system-level characteristics:

1. **Low-Burden Multimodal Assessment:** Combines short audio (30–60 s), lightweight questionnaire items, and basic behavioral indicators without requiring wearable hardware or invasive monitoring.
2. **Tri-Perspective Synthesis:** Merges subjective self-report, objective acoustic signals, and contextual behavioral patterns for robust risk estimation.
3. **Cross-Modal Consistency & Disagreement Analysis:** Explicitly quantifies whether modalities agree (e.g., all indicate mild tension) or conflict (e.g., low self-reported stress but high vocal/behavioral tension), highlighting masked stress.
4. **Confidence-Aware Assessment:** Reports uncertainty/confidence scores alongside estimates so students and advisors understand output reliability.
5. **Factor Explainability:** Breaks down contributing factors per modality rather than delivering an opaque score.
6. **Non-Clinical Early-Intervention Support:** Maps assessment profiles to constructive, non-clinical recommendations (e.g., study pacing, sleep hygiene, breathing exercises, campus wellness resources).

*(Note: These novelty concepts represent planned research goals and will be iteratively implemented and validated.)*

---

## Non-Clinical Disclaimer

> [!IMPORTANT]
> **This platform is an academic research and early-intervention support system, NOT a medical diagnostic tool.**
>
> - The system does **not** diagnose depression, anxiety, or any psychiatric disorder.
> - The system does **not** provide medical or clinical advice.
> - The system does **not** replace licensed psychologists, psychiatrists, counselors, or medical professionals.
> - All outputs are framed strictly as **stress/risk indicators**, **estimated well-being indicators**, and **supportive recommendations** for proactive student self-care.

---

## Repository Directory Guide

```
EmotionAwareCloudPlatform/
|-- architecture/       # System architecture documentation and pipeline diagrams
|-- dataset/            # Data schemas, voice recording protocols, and ethical guidelines
|-- docs/               # Project documentation index, glossary, and ethical policies
|-- literature_survey/  # Literature reviews across questionnaire, voice, and behavioral modalities
|-- presentation/       # Presentation decks, milestone reviews, and demo assets
|-- references/         # Academic references, citation standards, and bibliography
|-- results/            # Experimental results, evaluation metrics, and benchmarking logs
|-- src/                # Planned source code
|   |-- ai_model/       # Modality models (Questionnaire, Voice, Behavioral) & fusion logic
|   |-- azure/          # Cloud infrastructure and deployment specifications
|   |-- backend/        # Planned REST API and assessment orchestration service
|   +-- frontend/       # Planned student assessment and feedback dashboard UI
+-- README.md           # Root project overview and architecture documentation (this file)
```

---

## Current Development Status

- **Phase:** Documentation, architectural alignment, and modality protocol definition.
- **Implementation Status:** Code implementation is pending. Modality models, pipelines, backend APIs, and frontend interfaces are in the design and documentation stage across respective feature branches.
