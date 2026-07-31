# Emotion-Aware Cloud Platform for Mental Health Monitoring using Multimodal AI

BITE412L — Cloud Computing, Phase I
Course Instructor: Dr. Priya V

---

## Team Members

| Name | Role / Novelty Angle Owned |
|---|---|
| Nandini Goyal | AI Model & Intelligence — Attention-based fusion + modality-dropout robustness |
| Netal Agarwal | Pipeline & Infrastructure — Serverless, event-driven Azure architecture |
| Khwaish Agarwala | Security & Dashboard — Edge-cloud privacy-preserving design |

---

## Problem Statement

Mental health conditions such as chronic stress, anxiety, and depression often remain undetected until they significantly affect an individual's wellbeing. Traditional diagnosis relies heavily on periodic self-reported questionnaires, which are subjective, infrequent, and prone to underreporting due to stigma or lack of self-awareness. Existing emotion recognition systems typically rely on a single data modality (facial expression, voice, or text) and are developed/tested only in controlled lab environments, leaving a significant gap between research prototypes and deployable, cloud-scale healthcare systems.

This project proposes an emotion-aware cloud platform that fuses speech, text, and physiological signals from wearable devices to detect emotional states with greater accuracy and robustness — including graceful handling of a missing modality — deployed on a fully serverless Azure architecture with a privacy-conscious edge-to-cloud pipeline.

---

## Objectives

1. Develop a multimodal AI fusion pipeline combining speech, text, and physiological (heart rate + EDA) signals using an attention-based late fusion model, classifying at least 4 emotional/stress states.
2. Achieve at least 85% classification accuracy on the fused model, benchmarked against single-modality baselines.
3. Maintain robust performance under modality dropout, with no more than a 10–15% accuracy drop when any single modality is unavailable.
4. Reduce end-to-end processing latency to under 5 seconds per micro-batch using Azure serverless compute.
5. Ensure privacy-preserving, secure data handling via edge feature extraction, Microsoft Entra ID authentication, and Azure Key Vault secrets management.
6. Provide a clinician-facing Power BI dashboard visualizing individual and trend-level emotional states over time, with no hosted web application required.

Full detail: [`docs/objectives.md`](docs/objectives.md)

---

## Proposed Architecture / Framework

```
Simulated device data (WESAD + MELD)
        |
        v
Azure IoT Hub --ingest--> Azure Functions --extract features--> Azure Blob Storage
                                                                       |
                                                                       v
                                                    Azure ML (attention-based fusion model)
                                                                       |
                                                                       v
                                                    Azure Cosmos DB (results + trend history)
                                                                       |
                                              +------------------------+------------------------+
                                              v                                                   v
                                    Trend/risk detection                              Power BI dashboard
                                              |
                                              v
                          Service Bus -> Notification Hubs -> Clinician alert

Cross-cutting: Microsoft Entra ID (auth) | Azure Key Vault (secrets) | Azure Monitor (logging)
```

Full architecture diagrams: [`architecture/`](architecture/)

---

## Technology Stack

**Cloud Platform:** Microsoft Azure (Azure for Students subscription)

**Ingestion & Messaging:** Azure IoT Hub, Azure Event Hubs/Event Grid, Azure Service Bus

**Compute:** Azure Functions (serverless), Azure Logic Apps (optional orchestration)

**AI / ML:** Azure Machine Learning, Azure AI Speech, Azure AI Language, Python (PyTorch/TensorFlow, scikit-learn, pandas, numpy)

**Storage:** Azure Blob Storage, Azure Cosmos DB, Azure SQL Database

**Security:** Microsoft Entra ID, Azure Key Vault

**Monitoring:** Azure Monitor, Application Insights

**Visualization:** Power BI Desktop

**Dev Tools:** GitHub, VS Code, Azure CLI

Full detail with free-tier breakdown: [`docs/azure_services.md`](docs/azure_services.md)

---

## Dataset Details

| Dataset | Modality | Access |
|---|---|---|
| **WESAD** (Wearable Stress and Affect Detection) | Physiological (heart rate, EDA) | Freely downloadable — UCI ML Repository |
| **MELD** (Multimodal EmotionLines Dataset) | Speech + Text | Freely downloadable — no license request required |

Full details (source, size, records, features, license, preprocessing): [`dataset/dataset_details.md`](dataset/dataset_details.md)

---

## Repository Structure

```
EmotionAwarePlatform_Azure_Cloud_Project_2026/
│
├── README.md                       ← this file
├── docs/                           ← objectives, novelty summary, azure services planning
├── literature_survey/              ← 15-paper survey + individual research gap analysis
├── architecture/                   ← both mandatory architecture diagrams
├── dataset/                        ← dataset details (WESAD + MELD)
├── src/
│   ├── ai_model/                   ← Nandini: fusion model, baselines, training scripts
│   ├── azure/
│   │   ├── ingestion/              ← Netal: IoT Hub setup, simulated device script
│   │   ├── processing/             ← Netal: Azure Functions, feature extraction
│   │   └── security/               ← Khwaish: Entra ID, Key Vault, encryption config
│   └── dashboard/                  ← Khwaish: Power BI dashboard files, alert logic
├── results/                        ← model evaluation results, latency benchmarks
├── presentation/                   ← final presentation deck
└── references/                     ← citation list, reference materials
```

---

## Work Distribution

See [`docs/WORK_DISTRIBUTION.md`](docs/WORK_DISTRIBUTION.md) for the full team task division, individual responsibilities, objective ownership, and Git branch workflow.

---

## Git Workflow

```
main
 │
 develop
 ┌───────────┼───────────┐
 │           │           │
feature/    feature/    feature/
student-a-  student-b-  student-c-
model       pipeline    security-dashboard
```

No one commits directly to `main`. All work merges via Pull Request into `develop`, reviewed by the team, then merged to `main` and tagged `v1.0-Phase1` once stable.

---

## Status

Phase I — Planning and architecture design complete. Implementation in progress.
