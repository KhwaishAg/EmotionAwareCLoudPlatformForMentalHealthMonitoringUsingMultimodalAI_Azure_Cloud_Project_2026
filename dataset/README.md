# Dataset Specifications & Data Governance

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This document outlines the data requirements, collection protocols, ethical standards, and governance procedures for the three modalities in the student stress/risk assessment platform.

---

## 1. Voice Modality Dataset Specifications

*Primary Responsibility: Nandini (`feature/Nandini`)*

### Overview
The voice modality analyzes acoustic and expressive indicators extracted from short, standardized speech samples provided by student participants.

### Audio Recording Specifications
- **File Format:** WAV (uncompressed PCM audio)
- **Audio Channels:** Mono (single channel)
- **Sampling Rate:** 16,000 Hz (16 kHz)
- **Bit Depth:** 16-bit
- **Recording Duration:** Approximately 30–60 seconds per sample

### Standardized Elicitation Prompt
To ensure consistency across participants and minimize variability caused by task differences, all participants respond to the following standardized prompt:

> *"Please describe your recent academic workload, your typical college day, and how you have been feeling about your studies."*

This prompt directs participants toward academic and routine-related experiences, evoking natural communicative speech relevant to academic well-being.

### Participant Anonymization Protocol
- Every recording must be indexed using an **anonymous participant identifier** (e.g., `P001`, `P002`, `P003`).
- **Strict Prohibition of PII:** Audio metadata, filenames, and transcripts must never contain:
  - Student names
  - Email addresses
  - Student ID numbers or phone numbers
  - Any personally identifying institutional information

### Candidate Public Speech Datasets for Research & Experimentation

For preliminary feature engineering, acoustic baseline benchmarking, and validation of the audio preprocessing pipeline, the following public speech-emotion datasets may be explored:

| Dataset | Language | Sample Size | Primary Focus | Use in This Project |
|---|---|---|---|---|
| **RAVDESS** | English | 24 actors (1440 audio files) | Emotional speech and song | Acoustic feature extraction validation |
| **SAVEE** | English | 4 male actors (480 audio files) | Emotion classification | Baseline pipeline testing |
| **TESS** | English | 2 female actors (2800 audio files) | Emotion classification | Acoustic variability testing |
| **EmoDB** | German | 10 actors (535 audio files) | Acted emotion speech | Cross-dataset feature exploration |

> [!IMPORTANT]
> **Domain Limitation Note:**
> These public datasets contain acted emotional speech collected under controlled conditions. **They do not directly represent natural student academic stress.**
>
> While they provide valuable benchmarks for verifying acoustic extraction algorithms (e.g., pitch tracking, MFCC computation, energy contour extraction), models trained on acted emotions cannot be assumed equivalent to the final student academic stress/risk target.

### Target Labels
- **Status:** *Not finalized.*
- The exact label taxonomy (e.g., binary low/elevated risk vs. continuous risk score) is under active research and will be established in alignment with the team and faculty advisors.
- No synthetic or fabricated datasets or labels are permitted.

---

## 2. Questionnaire Modality Dataset *(Placeholder)*

*Primary Responsibility: Khwaish (`feature/Khwaish`)*

- **Purpose:** Capture subjective self-reported stress, perceived academic load, and psychological well-being indicators.
- **Instrument:** Standardized survey instrument (e.g., validated academic stress scales or perceived stress indices).
- **Data Structure:** Tabular records linked to anonymous participant IDs (`P001`, `P002`, etc.).
- **Status:** *To be specified and documented by Khwaish on `feature/Khwaish`.*

---

## 3. Behavioral Modality Dataset *(Placeholder)*

*Primary Responsibility: Netal (`feature/Netal`)*

- **Purpose:** Capture contextual, lifestyle, and academic behavioral indicators.
- **Attributes Under Consideration:** Self-reported study hours, sleep duration/regularity, break frequency, and proximity to major academic deadlines.
- **Data Structure:** Tabular time-series or aggregated vectors indexed by anonymous participant ID.
- **Status:** *To be specified and documented by Netal on `feature/Netal`.*

---

## Ethical Standards, Privacy & Data Governance

### 1. Informed Consent
All data collection involving human participants requires prior informed consent:
- Participants must be informed of the research scope, types of data recorded, and the voluntary nature of participation.
- Participants must be clearly informed that the system is an academic research platform and **not a diagnostic medical system**.
- Participants retain the right to withdraw at any point.

### 2. Participant Privacy & Anonymity
- All recordings and records are decoupled from identifying information at the point of ingestion.
- Mapping keys between student identities and participant IDs (if any exist for longitudinal tracking) must be stored in secure, encrypted storage separate from any project repository.

### 3. Repository Cleanliness Policy
> [!CAUTION]
> **NO PRIVATE RECORDINGS IN GIT:**
> Raw audio recordings, participant voice samples, filled questionnaires, and behavioral logs containing participant data must **never be committed to the GitHub repository**.
>
> Datasets must reside exclusively in designated secure cloud storage (e.g., access-controlled Azure Blob Storage) or secure local scratch directories excluded via `.gitignore`.
