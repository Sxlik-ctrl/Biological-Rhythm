# Automated Biological Rhythm Analysis & Health State Classification

A research-oriented Python project for automated analysis of biological activity rhythms and exploratory health-state classification using the **Depresjon actigraphy dataset**.

The project processes minute-level activity recordings, derives activity and activity-based sleep features, performs 24-hour Cosinor rhythm analysis, conducts statistical comparisons, trains machine-learning classifiers, evaluates feature importance, and provides an interactive Streamlit research dashboard.

> **Research disclaimer:** This project is an academic/research analytics system. It is not a clinical diagnostic tool, and its machine-learning results should not be interpreted as validated clinical performance.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Project Pipeline](#project-pipeline)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Analysis Phases](#analysis-phases)
- [Dashboard](#dashboard)
- [Visualization Resolution](#visualization-resolution)
- [Machine Learning](#machine-learning)
- [Explainability](#explainability)
- [Important Methodological Limitations](#important-methodological-limitations)
- [Key Results](#key-results)
- [Technologies Used](#technologies-used)
- [Reproducibility](#reproducibility)
- [Academic Use](#academic-use)

---

## Project Overview

Biological rhythms describe recurring patterns in physiological and behavioral activity. One important rhythm is the approximately 24-hour **circadian rhythm**.

This project develops an automated pipeline for analyzing actigraphy activity recordings and extracting:

- activity patterns,
- day/night activity differences,
- activity-derived sleep characteristics,
- 24-hour circadian rhythm parameters,
- statistical differences between study groups,
- machine-learning features for exploratory health-state classification,
- model explainability measures.

The project combines classical statistical rhythm analysis with machine learning and an interactive research dashboard.

---

## Objectives

The major objectives are:

1. Load and clean actigraphy activity data.
2. Preserve cleaned minute-level observations for detailed analysis.
3. Generate a standardized 1-hour analytical dataset.
4. Extract activity-related subject features.
5. Estimate sleep-related features from nighttime activity.
6. Fit a 24-hour Cosinor model for each subject.
7. Compare features between condition and control groups.
8. Perform exploratory feature selection.
9. Train and evaluate multiple classification models.
10. Interpret model behavior using permutation importance.
11. Integrate the results into an interactive Streamlit dashboard.

---

## Project Pipeline

```text
Raw Depresjon Dataset
        |
        v
Data Extraction
        |
        v
Data Cleaning
        |
        +------------------------------+
        |                              |
        v                              v
Cleaned 1-minute data            1-hour analytical data
        |                              |
        |                              +-------------------+
        |                                                  |
        v                                                  v
Dashboard visualization                         Activity/Sleep analysis
                                                       |
                                                       v
                                                Cosinor analysis
                                                       |
                                                       v
                                             Statistical comparison
                                                       |
                                                       v
                                             Feature selection
                                                       |
                                                       v
                                           Machine learning
                                                       |
                                                       v
                                                Explainability
                                                       |
                                                       v
                                             Final validation
                                                       |
                                                       v
                                                Streamlit UI
```

---

# Dataset

The main dataset used by this project is the **Depresjon actigraphy dataset**.

The extracted Depresjon structure used by the project is:

```text
data/
└── extracted/
    └── depresjon/
        └── data/
            ├── scores.csv
            ├── condition/
            │   ├── condition_1.csv
            │   ├── condition_2.csv
            │   └── ...
            └── control/
                ├── control_1.csv
                ├── control_2.csv
                └── ...
```

The processed data contains one cleaned activity file per subject:

```text
data/
└── processed/
    └── depresjon/
        ├── condition_1/
        │   ├── activity_cleaned.csv
        │   └── activity_hourly.csv
        ├── condition_2/
        │   ├── activity_cleaned.csv
        │   └── activity_hourly.csv
        └── ...
```

The cleaned minute-level activity files use the following schema:

```text
datetime
activity
subject_id
label
```

Example:

```csv
datetime,activity,subject_id,label
2003-05-07 12:00:00,0,condition_1,condition
2003-05-07 12:01:00,143,condition_1,condition
2003-05-07 12:02:00,0,condition_1,condition
2003-05-07 12:03:00,20,condition_1,condition
```

The processed dataset contains:

- **55 subjects**
- **23 condition subjects**
- **32 control subjects**

The project also contains MMASH data under:

```text
data/extracted/mmash/
```

The primary analysis described in the current pipeline is based on Depresjon.

---

# Project Structure

```text
biological_rhythm_project/
│
├── data/
│   ├── raw/
│   │   ├── depresjon.zip
│   │   └── mmash.zip
│   │
│   ├── extracted/
│   │   ├── depresjon/
│   │   └── mmash/
│   │
│   └── processed/
│       ├── depresjon/
│       │   ├── condition_1/
│       │   │   ├── activity_cleaned.csv
│       │   │   └── activity_hourly.csv
│       │   ├── ...
│       │   └── control_32/
│       │       ├── activity_cleaned.csv
│       │       └── activity_hourly.csv
│       │
│       └── depresjon_hourly_visualization.csv
│
├── src/
│   ├── phase3_activity_sleep_analysis.py
│   ├── phase4_cosinor_analysis.py
│   ├── phase5_statistical_feature_selection.py
│   ├── phase6_health_state_classification.py
│   ├── phase7_explainability_final_evaluation.py
│   └── phase8_final_integration.py
│
├── results/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6_final/
│   ├── phase7/
│   └── phase8/
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# Requirements

Recommended environment:

- Python 3.10+
- Windows / Linux / macOS
- pip
- Streamlit
- Pandas
- NumPy
- SciPy
- scikit-learn
- Plotly
- Matplotlib

Install dependencies using:

```powershell
python -m pip install -r requirements.txt
```

If the project uses a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

# Installation

Clone or copy the project:

```powershell
cd D:\Projects\biological_rhythm_project
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

Make sure the dataset is available under:

```text
data/raw/
```

---

# Running the Project

The analysis scripts are located in:

```text
src/
```

Run them in pipeline order.

## Phase 3 — Activity and Sleep Analysis

```powershell
python src\phase3_activity_sleep_analysis.py
```

Outputs are written to:

```text
results/phase3/
```

---

## Phase 4 — Cosinor Analysis

```powershell
python src\phase4_cosinor_analysis.py
```

Outputs:

```text
results/phase4/
```

---

## Phase 5 — Statistical Analysis and Feature Selection

```powershell
python src\phase5_statistical_feature_selection.py
```

Outputs:

```text
results/phase5/
```

---

## Phase 6 — Classification

```powershell
python src\phase6_health_state_classification.py
```

The final corrected classification results are stored under:

```text
results/phase6_final/
```

---

## Phase 7 — Explainability

```powershell
python src\phase7_explainability_final_evaluation.py
```

Outputs:

```text
results/phase7/
```

---

## Phase 8 — Final Integration and Validation

```powershell
python src\phase8_final_integration.py
```

Outputs:

```text
results/phase8/
```

The validation file is:

```text
results/phase8/phase8_validation_checks.csv
```

The completed pipeline contains **10/10 validation checks passed**.

---

# Dashboard

The project includes a Streamlit research dashboard.

Start it with:

```powershell
cd D:\Projects\biological_rhythm_project

python -m streamlit run dashboard\app.py
```

The dashboard contains:

- Overview
- Subject Explorer
- Activity Analysis
- Sleep Analysis
- Circadian Rhythm
- Statistical Analysis
- Classification
- Explainability

The dashboard is designed as a **scientific/clinical research analytics interface**, rather than a generic AI-style dashboard.

---

# Visualization Resolution

The dashboard preserves the original cleaned 1-minute activity data.

The Subject Explorer can dynamically aggregate this data for visualization at:

```text
15 minutes
30 minutes
45 minutes
60 minutes
90 minutes
120 minutes
```

For example:

| Resolution | Approx. observations per 24 hours |
|---:|---:|
| 15 min | 96 |
| 30 min | 48 |
| 45 min | 32 |
| 60 min | 24 |
| 90 min | 16 |
| 120 min | 12 |

This control changes the **actual observed data aggregation**, not merely the X-axis labels.

The dashboard uses:

```text
data/processed/depresjon/<subject>/activity_cleaned.csv
```

for this visualization.

The established Phase 3–8 analytical data remains based on the project's 1-hour analytical dataset.

This distinction prevents the visualization control from silently changing the project's statistical and machine-learning results.

---

# Phase 3 — Activity and Sleep Analysis

Phase 3 generates subject-level activity and sleep features.

## Activity features

Examples include:

- mean activity
- median activity
- activity standard deviation
- minimum activity
- maximum activity
- total activity
- daytime mean activity
- nighttime mean activity
- day/night activity ratio
- most active hour
- least active hour
- activity coefficient of variation

## Sleep features

Examples include:

- estimated sleep hours
- estimated wake hours
- estimated sleep periods
- mean nighttime activity
- activity threshold
- sleep regularity

### Sleep estimation methodology

The current implementation uses a heuristic activity-based method:

- subject-specific low-activity threshold
- nighttime window
- low-activity nighttime observations
- consecutive low-activity periods

These estimates are **not clinical sleep measurements**.

In particular, the current `estimated_sleep_hours` feature represents accumulated low-activity nighttime hours across the subject's recording period rather than average sleep duration per night.

---

# Phase 4 — 24-Hour Cosinor Analysis

The project models activity using a fixed 24-hour Cosinor rhythm:

```text
Y(t) = M + A cos(2πt/24 − φ)
```

where:

- `M` = Mesor
- `A` = Amplitude
- `φ` = Acrophase

Extracted features include:

- Mesor
- Amplitude
- Relative amplitude
- Acrophase
- Trough
- Fitted maximum
- Fitted minimum
- R²
- RMSE
- Cosinor coefficients

All 55 subjects were successfully fitted.

### Observed Cosinor summary

| Metric | Mean |
|---|---:|
| R² | 0.789 |
| Amplitude | 148.74 |
| Acrophase | ~15.01 h |

The group-level results are descriptive and should not be interpreted as evidence of causality.

---

# Phase 5 — Statistical Analysis

The statistical pipeline compares condition and control groups at the subject level.

Methods include:

- Mann–Whitney U test
- Cohen's d
- Rank-biserial effect size
- Benjamini–Hochberg FDR correction
- correlation-based redundancy screening

The analysis identified **7 FDR-significant features** at:

```text
FDR-adjusted p < 0.05
```

The feature-selection stage is exploratory.

---

# Phase 6 — Machine Learning

The classification stage evaluates:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine
4. Gradient Boosting

The final feature set contains 15 features.

Duration-related variables such as:

- `hourly_records`
- `total_activity`

were removed from the final classifier because they can reflect recording duration rather than biological characteristics.

## Final 5-fold CV results

| Model | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.782 | 0.775 | 0.744 | 0.843 |
| Random Forest | 0.709 | 0.692 | 0.614 | 0.847 |
| SVM | 0.709 | 0.700 | 0.637 | 0.786 |
| Gradient Boosting | 0.818 | 0.808 | 0.771 | 0.917 |

## Final Gradient Boosting OOF results

```text
Accuracy            0.8182
Balanced accuracy   0.8071
Precision           0.8095
Recall              0.7391
F1                  0.7727
ROC-AUC             0.8954
```

These results are exploratory model estimates on 55 subjects.

They should not be interpreted as clinical diagnostic accuracy.

---

# Phase 7 — Explainability

The final classifier is interpreted using **held-out permutation importance**.

The highest individual permutation importance was:

```text
estimated_sleep_hours
```

Feature groups were summarized into:

- Activity
- Sleep
- Circadian / Cosinor

The Sleep group had the highest mean absolute importance in the final analysis.

Permutation importance indicates predictive contribution within the evaluated model. It does not demonstrate causation.

---

# Phase 8 — Final Integration

Phase 8 combines the results from all previous stages and performs project-level validation.

The final validation included checks for:

- total subject count
- condition subject count
- control subject count
- activity feature count
- sleep feature count
- Cosinor subject count
- FDR-significant feature count
- model performance
- explainability output
- final project consistency

Final validation:

```text
10/10 checks passed
```

---

# Key Results

The final project summary contains:

```text
Dataset subjects                 55
Condition subjects               23
Control subjects                 32

FDR-significant features          7

Gradient Boosting accuracy       0.8182
Gradient Boosting F1             0.7727
Gradient Boosting ROC-AUC        0.8954

Top explainability feature       estimated_sleep_hours
Highest feature group            Sleep
```

---

# Important Methodological Limitations

## 1. Small subject-level sample

The dataset contains only 55 subjects:

```text
23 condition
32 control
```

Therefore, the machine-learning results should be considered exploratory.

---

## 2. Sleep is heuristic

The project estimates sleep from activity patterns.

It does not use validated polysomnography or clinical sleep staging.

Therefore:

```text
estimated_sleep_hours
```

should not be interpreted as clinically measured sleep duration.

---

## 3. Accumulated sleep metric

The current Phase 3 implementation calculates low-activity nighttime hours across the recording period.

Consequently, values such as 200–300 hours can be legitimate accumulated values.

They do **not** mean that a participant sleeps 200–300 hours per night.

---

## 4. Feature-selection leakage limitation

The final classifier removed recording-duration variables, but the 15-feature set was derived from Phase 5 exploratory feature selection before final cross-validation.

Therefore, the final pipeline should not be described as having completely nested, leakage-free feature selection.

A future methodological improvement would perform feature selection independently inside each training fold.

---

## 5. Exploratory classification

The classification models are designed to investigate whether extracted rhythm-related features contain information associated with the dataset's labels.

They do not establish:

- clinical diagnosis,
- clinical screening performance,
- causality,
- medical suitability,
- generalization to a new clinical population.

---

# Technologies Used

## Programming

- Python

## Data Analysis

- Pandas
- NumPy
- SciPy

## Machine Learning

- scikit-learn

## Visualization

- Matplotlib
- Plotly

## Dashboard

- Streamlit

## Statistical / Rhythm Analysis

- Mann–Whitney U
- Cohen's d
- Rank-biserial effect size
- Benjamini–Hochberg FDR
- 24-hour Cosinor analysis
- Permutation importance

---

# Reproducibility

For a clean reproduction:

```powershell
cd D:\Projects\biological_rhythm_project
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the analysis phases in order:

```powershell
python src\phase3_activity_sleep_analysis.py
python src\phase4_cosinor_analysis.py
python src\phase5_statistical_feature_selection.py
python src\phase6_health_state_classification.py
python src\phase7_explainability_final_evaluation.py
python src\phase8_final_integration.py
```

Then start the dashboard:

```powershell
python -m streamlit run dashboard\app.py
```

---

# Generated Results

Important output directories:

```text
results/phase3/
results/phase4/
results/phase5/
results/phase6_final/
results/phase7/
results/phase8/
```

The final integrated outputs are in:

```text
results/phase8/
```

including:

```text
final_project_summary.csv
final_activity_summary.csv
final_sleep_summary.csv
final_cosinor_summary.csv
final_statistical_summary.csv
final_ml_summary.csv
final_feature_importance.csv
final_feature_group_importance.csv
phase8_validation_checks.csv
phase8_report.json
```

---

# Academic Use

This project was developed as a **B.Tech Information Technology major project** focused on automated biological rhythm analysis and exploratory health-state classification.

The project demonstrates integration of:

```text
Data Engineering
      +
Time-Series Analysis
      +
Circadian Rhythm Modeling
      +
Statistical Analysis
      +
Machine Learning
      +
Explainable AI
      +
Interactive Visualization
```

The intended use is academic research, demonstration, experimentation, and educational analysis.

---

## Project Status

```text
Data preparation       ✓ Complete
Activity analysis      ✓ Complete
Sleep analysis         ✓ Complete
Cosinor analysis       ✓ Complete
Statistical analysis   ✓ Complete
Classification         ✓ Complete
Explainability         ✓ Complete
Final validation       ✓ 10/10 checks passed
Dashboard              ✓ Implemented
```

---

## License / Dataset Notice

This repository contains code and project outputs. Dataset files should be distributed only according to the terms and license of their original source.

If redistributing this project, verify the applicable dataset citation and usage requirements before including the raw dataset.
