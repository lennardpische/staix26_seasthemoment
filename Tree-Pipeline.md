# STAI-X 2026: Tree-based pipeline

## Overview

This repository contains a tree-based machine learning pipeline for the **STAI-X 2026 Challenge**, which predicts **suspected nonfatal overdose ED visits per 10,000 ED visits** across U.S. jurisdictions.

The solution combines:

* Socioeconomic indicators
* Weather data
* Google Trends signals
* State Department of Health press releases
* MAT density heatmap features
* Temporal rolling statistics

Separate XGBoost models are trained for each overdose category to directly optimize competition performance.

---

## Competition Objective

Predict:

```text
rate_per_10000_ed_visits
```

for every `(period_id, jurisdiction, overdose_category)` combination in the validation set.

### Target Categories

* `all_drugs`
* `all_opioids`
* `all_stimulants`

Models are trained independently for each category.

---

## Evaluation Metric

The leaderboard score is the average MAE across all three categories:

```text
Score = mean(
    MAE_all_drugs,
    MAE_all_opioids,
    MAE_all_stimulants
)
```

Each category contributes equally to the final score.

---

## Data Sources

### Tabular Features

* Unemployment rate
* Labor force
* Average temperature
* Total precipitation
* Google Trends:

  * overdose
  * fentanyl
  * naloxone
  * opioid
  * methamphetamine

### Text Features

Derived from `state_doh_release`:

* Presence of text
* Character count
* Numeric character count
* Statistical term mentions
* Opioid mentions
* Stimulant mentions
* Total drug mentions

### Image Features

Extracted from MAT density heatmaps:

* Mean intensity
* Median intensity
* Standard deviation
* Maximum intensity
* Number of hotspots
* Fraction above mean intensity

---

## Feature Engineering

### Weather

```python
weather_extremity = temp_avg_f * precip_in
```

### Google Trends Aggregations

```python
gtrends_max
gtrends_total
gtrends_std
```

### Geographic

```python
region
```

Jurisdictions are mapped to U.S. census-style regions.

### Temporal Features

Rolling mean and standard deviation using:

* 3-period lookback
* 12-period lookback

Applied to weather, unemployment, Google Trends, and text-derived variables.

---

## Model Architecture

Three independent XGBoost regressors:

```text
all_drugs      → XGBoost
all_opioids    → XGBoost
all_stimulants → XGBoost
```

This mirrors the competition scoring structure and allows category-specific learning.

---

## Validation Strategy

### Cross Validation

* 4-fold Grouped Cross Validation
* Grouped by `period_id`
* Evaluation metric: MAE

Grouping by period prevents temporal leakage and better simulates future forecasting.

---

## Hyperparameter Tuning

Parameters considered:

```python
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
reg_alpha
reg_lambda
n_estimators
```

Objective:

```text
Minimize Mean Absolute Error (MAE)
```

---

## Training Pipeline

```text
Load Data
    ↓
Feature Engineering
    ↓
Generate Text Features
    ↓
Generate Image Features
    ↓
Create Rolling Statistics
    ↓
Split by Overdose Category
    ↓
4-Fold Grouped CV
    ↓
Tune XGBoost Models
    ↓
Train Final Models
    ↓
Generate Validation Predictions
    ↓
Create Submission
```

---

## Submission

Output predictions for every row in `sample_submission.csv` and save:

```text
submission.csv
```

Required columns:

```text
row_id
rate_per_10000_ed_visits
```

---

## Key Design Principles

* Category-specific modeling
* Leakage-resistant validation
* Multimodal feature integration
* Temporal trend extraction
* Strong tabular-learning performance
* Simple, reproducible, and scalable architecture
