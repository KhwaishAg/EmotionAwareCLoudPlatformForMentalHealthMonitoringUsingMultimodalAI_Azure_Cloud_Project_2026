# Voice Audio Pipeline — Recording Protocol & Standardized Prompt

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This document defines the standardized voice recording protocol, elicitation prompt, audio specifications, and participant privacy guidelines for the **Voice Modality** of the Multimodal Student Stress/Risk Assessment system.

---

## 1. Standardized Recording Prompt

All participants respond to the following standardized prompt:

> **"Please describe your recent academic workload, your typical college day, and how you have been feeling about your studies."**

### Purpose

The prompt is designed to:
- Elicit natural, conversational speech about academic experiences and emotional state.
- Provide a reasonably consistent speech sample across all participants, enabling comparable extraction of voice-derived acoustic indicators (pitch dynamics, energy patterns, spectral characteristics).
- Avoid artificially induced emotional expression — participants speak in their natural voice about their genuine academic context.

The prompt is intentionally open-ended to allow natural variation in speech duration, fluency, and expressive content while maintaining topical consistency.

---

## 2. Target Recording Duration

- **Recommended duration:** Approximately **30–60 seconds**.
- **Minimum acceptable duration:** 5 seconds (to allow meaningful acoustic feature extraction).
- **Maximum acceptable duration:** 120 seconds (to prevent excessively long processing times).

Participants should be encouraged to speak naturally for the recommended duration. They do not need to fill exactly 60 seconds — the goal is a natural, unhurried response.

---

## 3. Preferred Audio Format & Specifications

| Parameter | Specification | Rationale |
|---|---|---|
| **File format** | WAV (uncompressed PCM) | Lossless audio preserves acoustic detail for reliable feature extraction |
| **Channels** | Mono (single channel) | Standardizes input; eliminates stereo spatial artifacts |
| **Sample rate** | 16,000 Hz (16 kHz) | Sufficient for speech bandwidth (~8 kHz Nyquist); matches common speech processing conventions |
| **Bit depth** | 16-bit | Standard resolution for speech recordings |

If a recording is captured at a different sample rate or in stereo, the preprocessing pipeline will automatically resample to 16 kHz mono. However, source recordings at or above 16 kHz are preferred to avoid upsampling artifacts.

---

## 4. Recording Guidelines for Participants

When collecting voice samples, the following guidelines should be communicated to participants:

1. **Environment:** Record in a reasonably quiet environment. Avoid locations with loud background noise (cafeteria, busy street, construction). A quiet room, library study space, or dormitory room with the door closed is preferred.

2. **Natural speaking voice:** Speak in your normal, everyday voice. Do not whisper, shout, or intentionally alter your voice.

3. **Speak clearly:** Maintain a comfortable speaking pace. There is no need to rush or speak unnaturally slowly.

4. **Do not perform emotions:** Do not intentionally exaggerate or suppress your emotional expression. The recording captures your natural speech — there are no "right" or "wrong" responses.

5. **Complete the prompt naturally:** Respond to the prompt as fully as you wish within the recommended 30–60 second window. If you finish your thought before 30 seconds, that is acceptable. If you need slightly longer, that is also fine.

6. **Microphone positioning:** Hold the recording device (phone, laptop) at a comfortable distance — roughly 15–30 cm from your mouth. Avoid covering or tapping the microphone during recording.

---

## 5. Participant Identification — Anonymous ID Convention

All recordings and associated metadata must use **anonymous participant identifiers**. No personally identifying information is permitted in filenames, metadata fields, or audio content.

### ID Format

```
P001
P002
P003
...
P099
```

### File Naming Convention

```
P001.wav
P002.wav
P003.wav
```

### Prohibited Identifiers

The following must **never** be used as filenames or metadata values:

- Student names (e.g., `john.wav`, `nandini_recording.wav`)
- Email addresses (e.g., `student@university.edu.wav`)
- Student ID numbers or phone numbers
- Any other directly identifying information

---

## 6. Privacy & Ethical Requirements

### Informed Consent
- Every participant must provide informed consent before recording.
- Participants must be told that the recording will be used for academic research on voice-derived stress/risk indicators.
- Participation is entirely voluntary; participants may withdraw at any time.

### Data Protection
- The mapping between participant identity and anonymous ID (if maintained for longitudinal tracking) must be stored in a **secure, encrypted location separate from the repository**.
- Raw audio recordings must **never** be committed to the Git repository.
- Audio files are excluded from version control via the project `.gitignore`.

### Repository Policy

> [!CAUTION]
> **No private recordings in Git.**
> Real participant audio files (`.wav`, `.mp3`, `.flac`) under `data/raw/audio/` are excluded by `.gitignore`. Only synthetic or generated test fixtures may be committed under `tests/`.

---

## 7. Scope Disclaimer

> [!IMPORTANT]
> **Non-Clinical Use Only**
>
> The voice recording is used exclusively to derive voice-level stress/risk indicators for this academic assessment system. The acoustic analysis extracts speech features (pitch, energy, spectral properties) as indicators of expressive state — it does **not** diagnose any medical or psychiatric condition.
>
> The system is a supportive, non-clinical tool for student well-being awareness and early-intervention recommendations.

---

## 8. Summary of Audio Input Contract

```
Input:
    - File:      P001.wav (anonymous participant ID)
    - Format:    WAV (PCM, uncompressed)
    - Channels:  Mono
    - Rate:      16,000 Hz
    - Duration:  ~30–60 seconds
    - Content:   Response to the standardized academic prompt

Preprocessing pipeline will:
    - Validate file format and duration
    - Convert to mono if stereo
    - Resample to 16 kHz if different
    - Normalize amplitude
    - Handle leading/trailing silence

Output:
    - Clean, standardized waveform ready for acoustic feature extraction
```
