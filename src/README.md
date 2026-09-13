# Source Code Structure & Architecture Guide

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This document outlines the planned source code structure, component responsibilities, and modular development guidelines for the **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

> [!IMPORTANT]
> **Implementation Status: Pending**
> All directories under `src/` are currently in the architectural specification and documentation stage. Source code implementation, ML model weights, API services, and user interfaces will be developed incrementally across dedicated feature branches.

---

## Planned Directory Structure

```
src/
|-- ai_model/     # Machine learning pipelines for all three modalities and fusion layer
|-- azure/        # Cloud architecture, storage configuration, and deployment specifications
|-- backend/      # REST API service, orchestration logic, and recommendation engine
+-- frontend/     # Student assessment UI, audio recording interface, and results dashboard
```

---

## Directory Responsibilities

### 1. `src/ai_model/`
- **Role:** Houses data preprocessing routines, feature extraction pipelines, model training scripts, and inference estimators for each modality.
- **Components:**
  - **Voice Pipeline (Nandini):** Audio normalization, silence trimming, acoustic feature extraction (MFCCs, pitch, spectral characteristics), feature scaling, and classical ML classification models.
  - **Questionnaire Pipeline (Khwaish):** Survey response encoding, psychometric feature scoring, and risk classification.
  - **Behavioral Pipeline (Netal):** Time-series/tabular behavioral aggregation, feature engineering, and lifestyle risk classification.
  - **Multimodal Fusion Layer (Netal / Collaborative):** Late-fusion synthesis, cross-modal consistency analysis, uncertainty/confidence calculation, and factor attribution.
- **Reference:** See [`src/ai_model/README.md`](ai_model/README.md) for detailed technical specifications.

### 2. `src/azure/`
- **Role:** Defines cloud infrastructure requirements, storage configurations, model management strategies, and deployment guidelines on Microsoft Azure.
- **Key Services:**
  - Azure Blob Storage (secure audio and tabular data storage)
  - Azure Machine Learning (experiment tracking, model registry)
  - Azure App Service (containerized API hosting)
- **Reference:** See [`src/azure/README.md`](azure/README.md) for architecture, secrets management, and local-vs-cloud workflow.

### 3. `src/backend/`
- **Role:** Web backend and API service orchestrating data ingestion, modality model inference, fusion computation, and recommendation delivery.
- **Responsibilities:**
  - RESTful endpoints for questionnaire payload submission, behavioral log upload, and audio file ingestion.
  - Asynchronous / modular invocation of modality inference engines.
  - Synthesis of multimodal results and delivery of non-clinical supportive feedback.
- **Reference:** See [`src/backend/README.md`](backend/README.md) for endpoint designs and data flow.

### 4. `src/frontend/`
- **Role:** Client-side interface providing students with a low-burden, accessible portal for assessment and feedback.
- **Responsibilities:**
  - Standardized voice recording interface (30–60 s capture with real-time waveform and prompt display).
  - Clean, accessible questionnaire response forms.
  - Behavioral data entry / logging view.
  - Explainable results dashboard visualizing estimated risk, modality breakdown, confidence, and supportive self-care recommendations.
- **Reference:** See [`src/frontend/README.md`](frontend/README.md) for UI components and wireframe layouts.

---

## Modular Development Guidelines

To ensure smooth multi-member collaboration without merge conflicts:
- Each team member develops primarily within their assigned module in `src/ai_model/` on their respective feature branch (`feature/Khwaish`, `feature/Nandini`, `feature/Netal`).
- Modality models must expose standard, decoupled inference signatures (e.g., accepting a file path or feature dict, returning a standardized prediction dictionary) so the backend and fusion layers can integrate seamlessly.
