# Rain Prediction for Melbourne — sklearn ML Pipeline

> **Binary classification · Feature Engineering · GridSearchCV · Random Forest vs. Logistic Regression**

---

## Overview

This project builds an end-to-end machine learning pipeline to predict whether it will rain **today** in the Melbourne metropolitan area, using only meteorological observations available from the previous day. The dataset originates from the [Australian Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/) and contains ~145,000 daily weather observations across Australia.

The focus is on engineering a robust, leak-free pipeline using `scikit-learn`, evaluating two classifiers under stratified cross-validation, and interpreting results through feature importance analysis.

---

## Problem Statement

Rainfall prediction is a **binary classification** problem with real-world asymmetry: missing a rain event (false negative) carries higher practical cost than a false alarm. The naive majority-class baseline achieves ~76% accuracy by always predicting "no rain" — the goal is to build a model that meaningfully improves upon this, particularly for rain-day recall.

---

## Technical Highlights

- **Leakage-aware feature engineering**: Identified and resolved a subtle temporal leakage issue with the `RainToday` column; renamed columns to reflect what is realistically observable at prediction time.
- **Geographic scoping**: Restricted data to Melbourne-area stations (Melbourne, MelbourneAirport, Watsonia) to avoid conflating climatologically distinct patterns across Australia's diverse climate zones.
- **Season feature**: Mapped raw dates to Southern Hemisphere seasons, capturing intra-year climatological variation without introducing ordinal date artifacts.
- **sklearn Pipeline composition**: Entire preprocessing + classification workflow encapsulated as a single `Pipeline` object, ensuring the preprocessor is fit only on training folds during cross-validation.
- **`ColumnTransformer`**: Applies `StandardScaler` to numeric features and `OneHotEncoder` to categorical features (wind direction, location, season) automatically.
- **`GridSearchCV` + `StratifiedKFold`**: Systematic hyperparameter search with 5-fold stratified cross-validation to preserve class balance across folds.
- **Feature importance analysis**: Extracted and visualised Gini-based importances from the Random Forest, correctly reconstructing original feature names post-encoding.

---

## Results

| Model | Test Accuracy | Rain Recall | Rain F1 |
|---|---|---|---|
| **Random Forest** (best) | **84%** | 0.57 | 0.63 |
| Logistic Regression | 83% | 0.51 | 0.59 |
| Naive baseline (majority class) | 76% | 0.00 | 0.00 |

**Best Random Forest hyperparameters:**
- `n_estimators=100`, `max_depth=20`, `min_samples_split=5`

**Most predictive features:** `Humidity3pm`, `Pressure3pm`, `Humidity9am`, `Temp3pm` — physically consistent with precipitation dynamics in temperate coastal climates.

---

## Dataset

- **Source:** [Australian Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/) via IBM Skills Network mirror
- **Raw size:** ~145,000 rows × 23 columns
- **After cleaning & geographic filter:** 7,557 rows (Melbourne area)
- **Target:** `RainToday` (binary: Yes/No) — ~24% positive class
- **Class imbalance:** ~3.2:1 (No Rain : Rain)

---

## Project Structure

```
.
├── AUSWeather_RainPrediction_Portfolio.ipynb   # Main notebook
└── README.md
```

---

## How to Run

```bash
# Clone and install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Open the notebook
jupyter notebook AUSWeather_RainPrediction_Portfolio.ipynb
```

The notebook is self-contained: the dataset is loaded directly from a public URL in the first code cell.

---

## Skills Demonstrated

`scikit-learn` · `pandas` · `Pipeline` · `ColumnTransformer` · `GridSearchCV` · `StratifiedKFold` · `RandomForestClassifier` · `LogisticRegression` · Feature Engineering · Class Imbalance Handling · Confusion Matrix Analysis · Feature Importance Visualisation

---

## Author

**Yassine Chakir** — Data Scientist  
[LinkedIn](https://linkedin.com/in/yassine-chakir-b2aba3247) · yassine.chakir@gmx.de
