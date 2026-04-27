# Predicting Calorie Expenditure During Exercise: Association Statistical Analysis Using Linear Regression

> A biostatistical modeling study applying single and multivariable linear regression to predict calorie burn from physiological and demographic features — Cairo University, Systems & Biomedical Engineering Department.

---

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Dataset Description](#dataset-description)
4. [Methodology](#methodology)
5. [Implementation Details](#implementation-details)
6. [Results & Insights](#results--insights)
7. [Technologies Used](#technologies-used)
8. [How to Run the Project](#how-to-run-the-project)
9. [Project Structure](#project-structure)
10. [Contributors](#contributors)
11. [References](#references)
12. [Future Work](#future-work)

---

## Overview

Accurate prediction of calorie expenditure during physical activity is a cornerstone of personalized healthcare, sports science, and biomedical device development. Quantifying energy expenditure enables clinicians, fitness professionals, and researchers to design evidence-based exercise prescriptions, monitor metabolic health, and build intelligent monitoring systems for wearable devices.

This project applies classical statistical regression techniques — specifically simple (univariate) and multivariable linear regression — to a large-scale exercise dataset to model calorie burn as a function of measurable physiological and demographic variables. The study is designed to be both statistically rigorous and interpretable, making it well-suited for biomedical applications where model transparency is essential.

The full interactive implementation is available on Kaggle:  
🔗 [View Notebook on Kaggle](https://www.kaggle.com/code/nardeenezzzarif/statistics-project-d11655)

---

## Objectives

- Investigate and quantify the linear associations between individual physiological variables (e.g., heart rate, body temperature, exercise duration) and calorie expenditure.
- Implement and validate simple linear regression using both manual closed-form derivation and the scikit-learn library.
- Construct and evaluate a multivariable OLS regression model incorporating all available predictors.
- Rigorously verify standard regression assumptions, including residual normality, distributional behavior of features, and absence of multicollinearity.
- Provide a reproducible, well-documented statistical pipeline suitable for academic submission and engineering portfolio use.

---

## Dataset Description

| Property | Detail |
|---|---|
| **Source** | [Kaggle — Calories Burnt Prediction Dataset](https://www.kaggle.com/) |
| **Original sample size** | 15,000 observations |
| **Post-cleaning sample size** | 14,615 observations |
| **Number of predictors** | 7 (after preprocessing) |
| **Response variable** | `Calories` — continuous numerical (kcal) |
| **Regression suitability** | Satisfies n ≥ p + 5 (14,615 ≥ 12) |

### Feature Summary

| Variable | Type | Unit | Description |
|---|---|---|---|
| `Age` | Quantitative | Years | Participant age |
| `Height` | Quantitative | cm | Body height |
| `Weight` | Quantitative | kg | Body weight |
| `Duration` | Quantitative | Minutes | Exercise session length |
| `Heart_Rate` | Quantitative | bpm | Average heart rate during exercise |
| `Body_Temp` | Quantitative | °C | Core body temperature |
| `Gender` | Categorical (Binary) | Female / Male | Biological sex |
| `Calories` | **Response** | kcal | Calorie expenditure (target) |

---

## Methodology

The study follows a comprehensive statistical modeling pipeline, structured as follows:

### 1. Data Preprocessing

**Outlier Detection and Removal (IQR Method)**  
Univariate outliers in all quantitative predictor variables were identified using the Interquartile Range method:

```
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

A total of 385 observations (2.57%) were flagged and removed, leaving 14,615 clean records. The target variable (`Calories`) was excluded from outlier filtering to avoid biasing the regression.

**Feature Standardization (Z-Score Normalization)**  
All continuous predictors were standardized to zero mean and unit variance:

```
z = (x − μ) / σ
```

This ensures numerical stability in OLS estimation and comparability across features with different physical units (e.g., Age in years vs. Heart Rate in bpm).

**Categorical Encoding (One-Hot Encoding, k−1 Strategy)**  
The binary `Gender` variable was encoded using k−1 one-hot encoding to produce a single binary column (`Gender_male`), preventing perfect multicollinearity (the "Dummy Variable Trap") and preserving full rank of the design matrix.

---

### 2. Statistical Analysis

**Descriptive Statistics**  
Comprehensive measures of central tendency and dispersion were computed for all quantitative features:
- Central tendency: Mean, Median, Mode
- Dispersion: Standard Deviation, Variance, Range, IQR
- Shape: Skewness, Kurtosis

**Distributional Assessment and Normality Testing**  
Feature distributions were examined using:
- Histograms (shape and modality)
- Boxplots (spread and residual outliers)
- Q–Q plots (tail deviations from normality)

Formal normality tests were applied at α = 0.05:
- **Anderson–Darling test** — sensitive to tail behavior
- **D'Agostino–Pearson test** — sensitive to skewness and kurtosis

All features formally rejected normality due to the large sample size (N > 5,000); however, visual diagnostics confirmed practical suitability for linear regression in all cases.

**Correlation Analysis (Pearson Coefficient)**  
Pearson correlation coefficients were computed between each predictor and the response variable to quantify linear associations and detect potential multicollinearity among predictors.

---

### 3. Regression Modeling

**Simple Linear Regression — Manual + Validation**  
A univariate regression model of the form `Y = β₀ + β₁X + ε` was fitted for each predictor individually. Coefficients were derived analytically using closed-form estimators:

```
b₁ = Sxy / Sxx
b₀ = ȳ − b₁x̄
```

Manual results were validated against scikit-learn's `LinearRegression`, confirming exact agreement to five decimal places.

**Multivariable Linear Regression (OLS via statsmodels)**  
A full multiple regression model was fitted using all 7 predictors:

```
Y = β₀ + β₁X₁ + β₂X₂ + ⋯ + β₇X₇ + ε
```

Model performance was assessed using:
- **F-statistic** — global model significance
- **R² and Adjusted R²** — goodness of fit
- **p-values** — individual predictor significance
- **Q–Q plot of residuals** — normality of error terms

---

## Implementation Details

The full analysis is implemented in a structured Jupyter Notebook with the following key steps:

1. **Data loading and initial inspection** — checking for missing values, duplicates, and structural anomalies
2. **Outlier detection** — IQR-based flagging and removal per feature
3. **Exploratory Data Analysis** — histograms, boxplots, and distribution plots for all variables
4. **Normality testing** — Anderson–Darling and D'Agostino–Pearson on all features
5. **Standardization and encoding** — Z-score normalization and k−1 one-hot encoding; verification of mean ≈ 0 and σ = 1
6. **Train/test stratified split** — stratified on `Gender` to preserve class proportions
7. **Simple regression** — manual coefficient derivation and sklearn cross-validation
8. **Correlation matrix** — Pearson heatmap with interpretation
9. **Multivariable OLS regression** — statsmodels summary with F-test, R², and coefficient table
10. **Residual diagnostics** — Q–Q plot of model residuals

---

## Results & Insights

### Model Performance

| Metric | Value |
|---|---|
| R² | 0.967 |
| Adjusted R² | 0.967 |
| F-statistic | 4.821 × 10⁴ |
| F-test p-value | ~0 (statistically significant) |

The model explains **96.7% of the variance** in calorie expenditure, demonstrating strong predictive power and negligible overfitting (as confirmed by Adjusted R²).

### Predictor Importance (Univariate Slopes)

| Feature | Slope (β₁) | Interpretation |
|---|---|---|
| Duration | 58.66 | Strongest predictor — time is the dominant driver |
| Heart Rate | 55.04 | Second strongest — reflects exercise intensity |
| Body Temp | 51.23 | Strong positive association — heat dissipation |
| Weight | 2.05 | Moderate effect |
| Age | 9.38 | Moderate positive, consistent with higher basal metabolic cost |
| Height | 0.98 | Minimal effect in isolation |
| Gender (Male) | 2.65 | Small positive effect for males |

### Pearson Correlation with Calories

| Feature | r | Strength |
|---|---|---|
| Duration | 0.955 | Very Strong Positive |
| Heart Rate | 0.897 | Very Strong Positive |
| Body Temp | 0.824 | Strong Positive |
| Weight | 0.583 | Moderate Positive |
| Height | 0.448 | Moderate Positive |
| Age | 0.154 | Weak Positive |

### Key Findings

- **Duration and Heart Rate** are the dominant predictors of calorie burn, collectively reflecting exercise intensity and time — the two most physiologically meaningful energy-expenditure drivers.
- **Multicollinearity** is present: `Body_Temp` shifts from a positive univariate coefficient (+51.23) to a negative multivariable coefficient (−12.01), indicating shared variance with Duration and Heart Rate. Coefficient magnitudes in the multivariable model should be interpreted with this in mind.
- **Residuals** are approximately normal in the central range but exhibit heavier-than-normal tails, which is acceptable given the high R² and large sample size.
- **Gender and Height** contribute minimally to prediction once exercise-intensity variables are included.

---

## Technologies Used

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 1.5.3 | Data manipulation, cleaning, descriptive statistics |
| `numpy` | 1.24.3 | Numerical computations and array operations |
| `scipy` | 1.10.1 | Statistical tests and IQR calculations |
| `matplotlib` | 3.7.1 | Core data visualization and plotting |
| `seaborn` | 0.12.2 | Statistical visualization and distribution plots |
| `scikit-learn` | 1.2.2 | Linear regression, preprocessing, train/test split |
| `statsmodels` | — | OLS regression, F-statistics, p-values, model summaries |

---

## How to Run the Project

### Option A — Run on Kaggle (Recommended)

Access the fully executable notebook directly on Kaggle (no installation required):  
🔗 [https://www.kaggle.com/code/nardeenezzzarif/statistics-project-d11655](https://www.kaggle.com/code/nardeenezzzarif/statistics-project-d11655)

### Option B — Run Locally

**Step 1: Clone the repository**
```bash
git clone https://github.com/your-username/calorie-prediction-regression.git
cd calorie-prediction-regression
```

**Step 2: Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install pandas==1.5.3 numpy==1.24.3 scipy==1.10.1 matplotlib==3.7.1 \
            seaborn==0.12.2 scikit-learn==1.2.2 statsmodels jupyter
```

**Step 4: Download the dataset**  
Download the *Calories Burnt Prediction* dataset from [Kaggle](https://www.kaggle.com/) and place it in the project root directory.

**Step 5: Launch the notebook**
```bash
jupyter notebook statistics-project.ipynb
```

**Step 6: Run all cells**  
In Jupyter, select `Kernel > Restart & Run All` to execute the complete pipeline from data loading through regression modeling.

---

## Project Structure

```
calorie-prediction-regression/
│
├── statistics-project.ipynb    # Full analysis notebook (EDA, modeling, diagnostics)
├── Team_12_Report.pdf          # Research report (IEEE-format)
└── README.md                   # Project documentation (this file)
```

---

## Contributors

This project was developed as a team coursework submission for the Biostatistics course, Systems & Biomedical Engineering Department, Cairo University.

| Contributor | Role |
|---|---|
| **Jana Gamal** | Data selection and preprocessing — train/test split, data cleaning, outlier removal, one-hot encoding, and Z-score standardization |
| **Nardeen Ezz** | Feature analysis — comprehensive descriptive statistics, histograms, normality testing (Anderson–Darling, D'Agostino–Pearson), Q–Q plots |
| **Youssef Samy** | Simple linear regression — manual coefficient derivation, visualization, and cross-validation with statsmodels and scikit-learn |
| **Farah Yehya** | Feature correlation analysis, multivariable regression model, result organization for paper and presentation |

---

## References

1. Biostatistics course slides. Faculty of Engineering, Cairo University.
2. Montgomery, D. C., & Runger, G. C. (2018). *Applied Statistics and Probability for Engineers* (7th ed.). John Wiley & Sons.
3. Gamal, J., Yehya, F., Ezz, N., & Samy, Y. (2024). *Association Analysis Using Linear Regression: Predicting Calorie Expenditure During Exercise*. Team 12 Research Report, Cairo University.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
5. Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python. *Proceedings of the 9th Python in Science Conference*.

---

## Future Work

Several avenues exist for extending this work within a biomedical engineering context:

- **Non-linear modeling** — Explore polynomial regression, decision trees, or gradient boosting (e.g., XGBoost) to capture non-linear relationships between physiological signals and calorie burn that linear models may underfit.
- **Feature engineering** — Derive interaction terms (e.g., Duration × Heart_Rate) or metabolic indices (e.g., BMI from Height and Weight) to improve model expressiveness without sacrificing interpretability.
- **Multicollinearity mitigation** — Apply Ridge or Lasso regularization to produce more stable coefficient estimates in the presence of correlated predictors, particularly among Duration, Heart Rate, and Body Temperature.
- **Population stratification** — Develop gender- or age-stratified models to capture subgroup-specific physiological dynamics.
- **Real-time wearable integration** — Deploy the regression model as an embedded inference module in wearable biomedical devices (e.g., smartwatches, chest-strap monitors) for continuous, personalized calorie tracking using live heart rate and motion sensor data.
- **Longitudinal study design** — Extend to repeated-measures or mixed-effects models that account for within-subject variability across multiple exercise sessions.

---

*This project was developed as part of the Biostatistics curriculum at the Systems & Biomedical Engineering Department, Cairo University.*