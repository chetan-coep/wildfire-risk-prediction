# 🔥 Wildfire Risk Prediction: Satellite-Driven Analysis with Explainable AI

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 📌 Project Overview
As global temperatures rise, the frequency and unpredictability of wildfires have escalated, causing severe ecological damage. Traditional fire prediction systems rely on static meteorological indices which fail to capture the complex, non-linear interactions between long-term climate drivers and short-term weather anomalies.

This project builds an automated, data-driven pipeline that merges **geospatial satellite data** with **temporal meteorological data** to predict wildfire risk. Crucially, it employs Explainable AI (XAI) via SHAP to attribute local risk to specific environmental factors (such as a sudden spike in Vapor Pressure Deficit), enabling climate scientists and emergency responders to make informed interventions.

## 📊 The Dataset
The dataset represents a full year of wildfire and climate observations across **India (Jan 1, 2022 – Dec 31, 2022)**, totaling **94,417 fire events**. 
* **Satellite Data:** NASA FIRMS (SUOMI VIIRS C2) active fire data.
* **Meteorological Data:** ERA5 Climate Reanalysis.

### ⚙️ Feature Engineering & Selection
Extensive data preprocessing was performed to prepare the environmental data for machine learning:
* **Thermodynamics:** Converted raw Kelvin data to Celsius (`temp_c`, `dewpoint_c`) and calculated **Vapor Pressure Deficit (VPD)**.
* **Wind Vectors:** Calculated scalar `wind_speed` from raw $u$ and $v$ vectors using $sqrt(u^2+v^2)$.
* **Cyclic Time Encoding:** Converted raw `month` (1-12) into `month_sin` and `month_cos` so algorithms understand the continuous chronological loop of seasons.
* **Target Transformation:** Applied `log1p` transformation to the highly right-skewed `fire_intensity` target to stabilize model gradients.
* **Collinearity Handling:** Removed lagging soil moisture variables (14-day, 30-day) after SHAP analysis proved they caused massive mathematical instability.

## 🚀 The Machine Learning Journey
Rather than jumping straight to complex black-box models, this repository documents a deliberate data science progression—starting from a linear baseline, diagnosing its failures, and progressively upgrading to tree-based ensembles. *(Note: All models were evaluated using a randomized 80/20 train/test split to prevent seasonal data leakage).*

### 1. Baseline: Multiple Linear Regression (`1LR.ipynb`)
* **Goal:** Establish a mathematical baseline.
* **Result:** Failed (Negative Test R²). 
* **Insight:** Used **SHAP (SHapley Additive exPlanations)** to audit the model. Discovered massive multicollinearity between overlapping weather features (like 7-day vs 30-day soil moisture) causing runaway weights. Proved that wildfire physics cannot be mapped with a rigid straight line.

### 2. Regularization: Ridge Regression (`3RLR.ipynb`)
* **Goal:** Cure the multicollinearity and stabilize the linear formula.
* **Result:** Achieved perfect stability (Overfit Gap < 1%), but still heavily underfitted (Test R² ~3.8%).
* **Insight:** Successfully forced the overlapping soil moisture weights to behave logically. However, the low accuracy served as definitive proof that the data requires an algorithm capable of handling complex "IF/AND" conditional thresholds.

### 3. Non-Linear Logic: Decision Tree (`2DT.ipynb`)
* **Goal:** Capture the conditional thresholds of nature (e.g., *IF* high wind *AND* low humidity).
* **Result:** Massive improvement, but fell into the "Single Tree Paradox."
* **Insight:** An unrestricted tree perfectly memorized the training data (Train R² 71%, Test R² 23%), creating a massive overfit gap. Applying hyperparameters (`max_depth=18`) controlled the gap, but sacrificed some complex environmental rules, dropping absolute accuracy. 

### 4. Ensemble Methods: Random Forest / XGBoost (`4RF.ipynb`)
* **Goal:** Combine hundreds of deep decision trees to capture high-accuracy thresholds while using voting mechanisms to cancel out the memorized noise.
* **Result:** Effectively balances the bias-variance tradeoff, providing the highest accuracy and generalizability for predicting unseen fire events.

## 👨‍💻 Author
**Chetan Kamble** *Computer Engineering Student, COEP Technological University* [LinkedIn] | [GitHub]
