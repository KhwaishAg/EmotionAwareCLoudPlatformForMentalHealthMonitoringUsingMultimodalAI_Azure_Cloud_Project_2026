# Academic References & Bibliography

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This directory manages the academic literature, citation standards, and verified reference collections supporting the **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

---

## Reference Management Policy

All academic claims, architectural choices, and feature engineering selections in this project must be grounded in peer-reviewed scientific literature.

### General Policies:
1. **No Fabricated Citations:** Team members must never generate synthetic, hallucinated, or unverified bibliographic entries. If a specific paper's details cannot be independently verified, list the **research topic or open query** rather than inventing authors, journal names, or DOIs.
2. **Standardized Citation Format:** Use **IEEE** or **APA 7th Edition** style consistently across all technical reports, slide decks, and documentation.
3. **BibTeX Management:** Machine-readable citation records should be maintained in a shared `references.bib` file stored in this directory as work progresses.

---

## Verified Research Foundations & Core Domains

The project builds upon foundational research across several key literature domains:

### 1. Speech Emotion Recognition (SER) & Voice-Based Stress Detection
- **Physiological Basis of Vocal Stress:** Studies demonstrating how sympathetic nervous system arousal during stress impacts vocal fold tension, glottal pulse characteristics, and fundamental frequency ($F_0$).
- **Acoustic Markers of Psychological Strain:** Empirical research analyzing the correlation between perceived stress/anxiety and shifts in pitch variability, speech rate, pause duration, and intensity dynamics.
- **Acted vs. Naturalistic Speech Corpora:** Comparative analyses of acoustic features in acted emotion datasets (e.g., RAVDESS, EmoDB, SAVEE) versus naturalistic, conversational, or task-induced stress recordings.

### 2. Acoustic Feature Extraction & Signal Processing
- **Cepstral Analysis:** Foundations of Mel-Frequency Cepstral Coefficients (MFCCs) and their discrete cosine transform formulations for spectral envelope capture in speech signals.
- **Prosodic Feature Sets:** Methods for robust pitch tracking (e.g., autocorrelation, YIN algorithm, probabilistic pitch estimation) and energy contour modeling.
- **Spectral Descriptors:** Definitions, extraction algorithms, and physical interpretations of spectral centroid, spectral flux, spectral rolloff, and spectral bandwidth.
- **Micro-Perturbation Metrics:** Theoretical foundations and computational extraction of vocal jitter, shimmer, and Harmonics-to-Noise Ratio (HNR) in acoustic voice quality evaluation.
- **Standardized Feature Toolkits:** Literature utilizing established open-source acoustic extraction frameworks (e.g., OpenSMILE, Librosa, Praat/Parselmouth) for affective computing benchmarks.

### 3. Psychometric & Questionnaire-Based Stress Assessment
- **Validated Psychological Scales:** Foundational literature on standardized instruments such as the Perceived Stress Scale (PSS-10, Cohen et al.) and Depression Anxiety Stress Scales (DASS-21, Lovibond & Lovibond).
- **Self-Report Biases:** Research examining retrospective recall errors, subjective interpretation differences, and social desirability bias in student stress surveys.

### 4. Multimodal Fusion in Affective Computing
- **Early vs. Late Fusion Paradigms:** Comparative studies evaluating decision-level fusion (weighted voting, probability averaging, stacking) versus feature-level fusion for heterogeneous modalities.
- **Cross-Modal Disagreement Analysis:** Literature exploring discordance between self-reported psychological states and physiological/acoustic manifestations, and how such divergence informs clinical and non-clinical risk estimation.
- **Uncertainty Quantification in Affective AI:** Methods for estimating predictive confidence in multimodal classifiers to prevent unwarranted certainty on ambiguous inputs.

---

## Reference Topic Index for Modality Investigation

The following verified research topics form the study foundation for Member 2 (Nandini) on the Voice Modality:

| Domain | Key Investigative Topics | Primary Acoustic Indicators |
|---|---|---|
| **Acoustic Stress Profiling** | Voice stress analysis, acoustic correlates of cognitive load | Pitch ($F_0$) mean/variance, jitter, speech rate |
| **Spectral Dynamics** | Spectral tilt, high-frequency energy distribution under stress | Spectral centroid, spectral rolloff, MFCC 1–13 |
| **Energy & Intensity** | Vocal effort variation during academic narrative elicitation | RMS energy contour, energy standard deviation |
| **Silence & Temporal Dynamics** | Hesitation, pause distribution, phonation-to-silence ratio | Unvoiced segment duration, pause count/length |
| **Machine Learning Benchmarks** | Tabular acoustic classification using classical estimators | Logistic Regression, Random Forest, SVM baselines |

---

## Citation Template (IEEE Format)

When adding verified references to this directory or project reports, format entries according to standard IEEE conventions:

```
[1] A. Author and B. Author, "Title of the research paper," Journal Name or Conference Proceedings, vol. X, no. Y, pp. 123-135, Month Year. DOI: 10.xxxx/xxxxx.
```
