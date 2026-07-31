# Contributing & Team Task Division

## Emotion-Aware Cloud Platform for Mental Health Monitoring using Multimodal AI

**Course:** BITE412L – Cloud Computing  
**Instructor:** Dr. Priya V

---

# Project Flow

```text
Simulated Device Data (WESAD + MELD)
                │
                ▼
         Azure IoT Hub
                │
                ▼
       Azure Functions
      (Feature Extraction)
                │
                ▼
      Azure Blob Storage
                │
                ▼
 Azure Machine Learning
 (Attention Fusion Model)
                │
                ▼
      Azure Cosmos DB
   (Results + Trend History)
         │            │
         ▼            ▼
Power BI Dashboard   Trend Detection
                           │
                           ▼
                     Service Bus
                           │
                           ▼
                  Notification Hubs
                           │
                           ▼
                     Clinician Alert

Cross-cutting Services
• Microsoft Entra ID
• Azure Key Vault
• Azure Monitor
```

---

# Team Responsibilities

## Student A — AI Model & Intelligence

### Responsibility

- Preprocess WESAD and MELD datasets
- Build single-modality baseline models
- Train attention-based fusion model
- Evaluate modality dropout
- Deploy model to Azure ML Endpoint

### Objectives

- Objective 2
- Objective 3

### Report Sections

- Model Methodology
- Dataset Details
- Novelty Summary (Fusion)

### Branch

```text
feature/student-a-model
```

---

## Student B — Pipeline & Infrastructure

### Responsibility

- Simulate IoT device
- Configure Azure IoT Hub
- Azure Functions
- Service Bus
- Cosmos DB
- Event Grid

### Objectives

- Objective 1
- Objective 4

### Report Sections

- Azure Services
- Serverless Architecture

### Branch

```text
feature/student-b-pipeline
```

---

## Student C — Security & Dashboard

### Responsibility

- Microsoft Entra ID
- Azure Key Vault
- Blob Encryption
- Service Bus
- Notification Hubs
- Power BI Dashboard

### Objectives

- Objective 5
- Objective 6

### Report Sections

- Security
- Privacy
- Dashboard

### Branch

```text
feature/student-c-security-dashboard
```

---

# Shared Responsibilities

| Activity | Distribution |
|-----------|--------------|
| Literature Survey | Nandini: 1–5, Netal: 6–10, Khwaish: 11–15 |
| Research Gap | Same split |
| Git Commits | 20–30 each |
| Pull Requests | Minimum 2 each |
| Documentation | Everyone |
| Final Report | Everyone |
| Presentation | Everyone |

---

# Git Workflow

```text
main
 │
 ▼
develop
 ├──────────────┬──────────────┐
 ▼              ▼              ▼
feature/     feature/      feature/
student-a    student-b     student-c
```

Workflow

1. Clone repository
2. Checkout your feature branch
3. Develop your module
4. Commit frequently
5. Push your branch
6. Create Pull Request → develop
7. Review & Merge
8. Merge develop → main
9. Tag release

---

# Repository Structure

```text
EmotionAwarePlatform_Azure_Cloud_Project_2026
│
├── docs
├── literature_survey
├── architecture
├── dataset
├── src
│   ├── ai_model
│   ├── azure
│   │   ├── ingestion
│   │   ├── processing
│   │   └── security
│   └── dashboard
├── results
├── presentation
└── references
```