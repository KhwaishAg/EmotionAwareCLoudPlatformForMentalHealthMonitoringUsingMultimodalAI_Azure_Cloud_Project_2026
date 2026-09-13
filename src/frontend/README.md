# Frontend UI Architecture & Interface Design

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This document details the planned user interface, interaction flows, recording components, and feedback dashboards for the **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

> [!IMPORTANT]
> **Implementation Status: Planned**
> All UI wireframes, client-side workflows, and component specifications described in this document are architectural designs. No frontend source code (HTML, CSS, JavaScript, or frameworks) has been created during this documentation pass.

---

## Planned User Experience (UX) Flow

The student-facing interface is designed to be **accessible, low-burden, and transparent**:

```
+-------------------------------------------------------------------------------+
| 1. Landing & Informed Consent                                                 |
|    - Explains research purpose & non-clinical nature of assessment            |
|    - Transparent privacy terms: anonymous ID, voluntary participation        |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| 2. Multi-Channel Assessment Workflow                                          |
|                                                                               |
|   Step A: Questionnaire            Step B: Voice Recording   Step C: Behavior |
|   - Concise survey items           - Standardized prompt     - Study hours    |
|   - Workload & stress rating       - In-browser recorder     - Sleep pattern  |
|                                    - 30-60s timer & waveform - Deadlines      |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| 3. Assessment Results Dashboard                                               |
|    - Overall Estimated Stress/Risk Level                                      |
|    - Tri-Modality Breakdown (Questionnaire / Voice / Behavioral)              |
|    - Cross-Modal Consistency / Disagreement Indicator                         |
|    - Assessment Confidence Metric                                             |
|    - Explainable Factor Attribution (What drove the assessment?)              |
|    - Personalized Supportive Non-Clinical Recommendations                     |
+-------------------------------------------------------------------------------+
```

---

## Core UI Components

### 1. Voice Recording Interface *(Key Voice Modality View)*
The voice interface allows students to record and submit a standardized audio sample directly through the browser:

- **Standardized Prompt Display:** Prominently displays the prompt:
  > *"Please describe your recent academic workload, your typical college day, and how you have been feeling about your studies."*
- **Recording Controls:** Clean Start / Pause / Stop / Re-record buttons with a live duration counter (30–60 seconds recommended).
- **Audio Feedback:** Real-time visual audio waveform or volume meter confirming microphone input.
- **Playback & Verification:** Audio player allowing the student to review their recording before confirming submission.
- **Privacy Notice:** Explicit reminder displayed directly below the recorder: *"Recordings are processed anonymously and stripped of personal identifiers."*

### 2. Questionnaire Interface
- Clean, uncluttered Likert-scale items designed to minimize cognitive fatigue.
- Progress bar showing step completion.

### 3. Behavioral Data Ingestion View
- Lightweight sliders and selectors for entering typical daily study duration, sleep consistency, and upcoming academic deadlines.

### 4. Explainable Assessment Dashboard
The dashboard avoids opaque binary outputs and presents a multi-dimensional synthesis:
- **Estimated Risk Gauge:** Visual spectrum (e.g., Low / Mild / Moderate / Elevated Indicators) with non-clinical labeling.
- **Modality Breakdown Cards:** Individual status for each of the three modalities.
- **Consistency Alert:** Highlights when modalities align or diverge (e.g., *"Your survey indicated low stress, but speech acoustic and behavioral markers suggest mild fatigue"*).
- **Confidence Rating:** Indicates whether the assessment has high, medium, or lower certainty.
- **Factor Attribution Section:** Plain-language bullet points explaining key contributing factors (e.g., *"Decreased sleep regularity"*, *"Elevated vocal dynamic variation"*).
- **Supportive Recommendations Card:** Actionable, self-directed wellness ideas and links to campus resources.

---

## Non-Clinical UI Framing Standards

To adhere to research ethics, the interface must strictly enforce the following UI messaging rules:
1. **Prominent Banner:** Every assessment page must include a notice:
   > *"This platform provides self-care and well-being indicators for academic stress reflection. It is not a clinical diagnostic tool and does not provide medical advice."*
2. **Supportive CTA:** A dedicated "Need to Talk?" button linking directly to campus counseling hotlines and peer wellness centers.
