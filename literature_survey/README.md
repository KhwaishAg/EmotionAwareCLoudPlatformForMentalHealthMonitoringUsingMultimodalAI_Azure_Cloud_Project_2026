# Literature Survey

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This document organizes the literature survey across the three core modalities and multimodal fusion techniques supporting the student stress/risk assessment system.

---

## Structure of the Literature Review

1. [Questionnaire-Based Psychological & Stress Assessment](#1-questionnaire-based-psychological--stress-assessment)
2. [Voice-Based Stress & Risk Assessment](#2-voice-based-stress--risk-assessment)
3. [Behavioral & Contextual Assessment](#3-behavioral--contextual-assessment)
4. [Multimodal Fusion Strategies](#4-multimodal-fusion-strategies)

---

## 1. Questionnaire-Based Psychological & Stress Assessment *(Placeholder)*

*Primary Responsibility: Khwaish (`feature/Khwaish`)*

- **Focus Areas:**
  - Validated psychometric instruments for perceived stress (e.g., PSS-10, DASS-21, GHQ-12)
  - Subjective bias, recall inaccuracies, and social desirability in self-reports
  - Compact survey design for low user fatigue in student environments
- **Status:** *To be expanded by Khwaish on `feature/Khwaish`.*

---

## 2. Voice-Based Stress & Risk Assessment

*Primary Responsibility: Nandini (`feature/Nandini`)*

### A. Speech Emotion Recognition (SER) & Acoustic Stress Manifestations
Psychological and physiological stress produces measurable perturbations in human speech production:
- **Autonomic Nervous System Arousal:** Acute stress alters respiration rate, vocal fold muscle tension, and salivation, directly impacting fundamental frequency ($F_0$), speech intensity, and articulation rate.
- **Vocal Tract Resonance Changes:** Muscle tension alters the geometry of the vocal tract, shifting vowel formant locations and modifying the spectral energy distribution.

### B. Acoustic Feature Extraction Methodologies

Acoustic features relevant to stress detection are categorized into spectral, prosodic, and voice quality domains:

1. **Mel-Frequency Cepstral Coefficients (MFCCs):**
   - Compact representation of the short-term power spectrum mapped to the perceptually-motivated Mel scale.
   - Typically, the first 13–20 coefficients along with their first ($\Delta$) and second ($\Delta\Delta$) temporal derivatives are used to capture static and dynamic spectral envelope properties.
2. **Prosodic & Pitch Features ($F_0$):**
   - Fundamental frequency mean, standard deviation, median, range, and slope.
   - Energy / Root Mean Square (RMS) contours reflect vocal intensity dynamics.
3. **Spectral Descriptors:**
   - **Spectral Centroid:** Represents the "brightness" or center of gravity of the spectrum; rises during heightened vocal effort.
   - **Spectral Rolloff & Flux:** Measures frequency below which a given percentage of energy resides and the rate of spectral change over time.
   - **Spectral Bandwidth:** Spread of frequency components around the centroid.
4. **Voice Quality & Micro-Perturbation Features (Exploratory):**
   - **Jitter:** Cycle-to-cycle frequency variations of vocal fold vibrations.
   - **Shimmer:** Cycle-to-cycle amplitude variations of vocal fold vibrations.
   - **Harmonics-to-Noise Ratio (HNR):** Ratio of periodic to aperiodic energy, indicating hoarseness or vocal strain.
   *(Note: Reliable jitter/shimmer/HNR extraction requires clean, high-SNR recording conditions and will be evaluated if supported by the dataset.)*

### C. Machine Learning & Modeling Paradigms

> [!IMPORTANT]
> **Modeling Strategy Note:**
> While modern research explores deep neural networks and self-supervised representations, **deep learning is NOT mandatory for this project**.
>
> The voice modality implementation will first establish robust **classical machine learning baselines** (e.g., Logistic Regression, Random Forest, Support Vector Machines) trained on structured acoustic feature vectors. This ensures interpretability, low computational overhead, and reliable performance on small-to-moderate sample sizes before exploring complex architectures.

The literature encompasses two primary modeling paradigms:

1. **Classical Machine Learning Baselines (Primary Initial Path):**
   - **Feature Pipeline:** Audio segmentation -> silence trimming -> acoustic feature extraction -> statistical aggregation (mean, std, min, max) -> standard scaling.
   - **Classifiers:**
     - *Logistic Regression:* Provides an interpretable, calibrated probabilistic baseline.
     - *Random Forest:* Captures non-linear feature interactions and provides intrinsic feature importance metrics.
     - *Support Vector Machines (SVM) & Gradient Boosted Trees (XGBoost/LightGBM):* Well-established performers on tabular acoustic feature sets with limited sample sizes.
2. **Deep Learning & Self-Supervised Audio Representations (Future Exploratory Path):**
   - *Convolutional Neural Networks (CNNs):* Applied to 2D log-mel spectrograms for spatial pattern detection.
   - *Recurrent Architectures (LSTM / GRU):* Applied to temporal sequences of frame-level features.
   - *Self-Supervised Pretrained Encoders:* Models such as **wav2vec 2.0** and **HuBERT**, pretrained on large speech corpora, offer rich contextual representations but require significant computational resources and risk overfitting on small target datasets.

---

## 3. Behavioral & Contextual Assessment *(Placeholder)*

*Primary Responsibility: Netal (`feature/Netal`)*

- **Focus Areas:**
  - Passive and self-reported behavioral tracking in student populations
  - Academic calendar impacts (midterms, finals, project milestones) on sleep and routine regularity
  - Feature representations for temporal behavioral sequences
- **Status:** *To be expanded by Netal on `feature/Netal`.*

---

## 4. Multimodal Fusion Strategies

*Primary Responsibility: Netal (`feature/Netal`) & Collaborative Team Effort*

- **Focus Areas:**
  - **Early (Feature-Level) Fusion:** Concatenating raw or transformed features into a single vector prior to classification; challenges with modality dimensional imbalance.
  - **Late (Decision-Level) Fusion:** Combining posterior probabilities or decision scores from independently trained modality classifiers via weighted voting, stacking, or Bayesian updating.
  - **Cross-Modal Consistency & Conflict Detection:** Mathematical formulations for identifying discordance between subjective reporting and objective physiological/acoustic markers.
  - **Confidence Weighting:** Dynamically down-weighting modalities exhibiting high uncertainty or poor signal-to-noise ratios.
- **Status:** *Under ongoing joint architectural research.*
