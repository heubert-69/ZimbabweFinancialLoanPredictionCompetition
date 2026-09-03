# Financial Time Series Analysis and Prediction

## Overview

This project implements a comprehensive machine learning pipeline for analyzing and predicting financial time series data using multiple modeling approaches. The analysis includes extensive feature engineering, model comparison, and performance evaluation across various algorithms.

## Dataset

The dataset consists of time series financial data with the following characteristics:
- **Rows:** 2,154,021 observations
- **Features:** 13 initial variables including:
  - Location data (lat, lon)
  - Time series indicators (TWS_t, SPEI_01_t, SPEI_03_t, SPEI_06_t, SPEI_12_t)
  - SOIL_MOISTURE_t
  - Seasonal components (month_sin, month_cos)
  - Target variable (target)

## Methodology

### 1. Exploratory Data Analysis (EDA)

- **Statistical Summaries:** Comprehensive data profiling using `skimpy` library
- **Missing Value Analysis:** No missing values detected in the dataset
- **Distribution Analysis:** 
  - Target variable distribution visualization
  - Box plots for outlier detection
  - Probability plots for normality assessment
- **Feature Distributions:** Distribution plots for all numerical features

### 2. Preprocessing and Feature Engineering

#### Target Transformation
- Applied log transformation to the target variable using `np.log1p()` for improved model performance

#### Temporal Feature Engineering
- **Calendar Features:** Extracted year and month from time indices
- **Cyclical Encoding:** 
  - Created `month_sin` and `month_cos` features for seasonal pattern capture
  - Utilizing `sin` and `cos` transformations for circular nature of months

#### Location-Based Feature Engineering
- **Grouping:** Features grouped by location coordinates (lat, lon)
- **Lagged Features:**
  - `_lag1`, `_lag3`, `_lag6` for each variable
  - Enables capture of temporal dependencies
- **Difference Features:**
  - `_diff1`, `_diff3` for trend analysis
- **Rolling Statistics:**
  - Rolling means and standard deviations with windows of 3, 6, and 12
  - Captures moving averages and volatility patterns

#### Cross-Scale Feature Creation
- **SPEI Contrasts:**
  - `SPEI_short_long`: SPEI_01_t - SPEI_12_t
  - `SPEI_medium_long`: SPEI_03_t - SPEI_12_t
  - `SPEI_03_06`: SPEI_03_t - SPEI_06_t
- These features capture drought/wetness changes across different timescales

### 3. Modeling Approach

#### Models Tested
1. **Ridge Regression**
   - Linear model with L2 regularization
   - Baseline for performance comparison

2. **Lasso Regression**
   - Linear model with L1 regularization
   - Feature selection capability

3. **XGBoost**
   - Gradient boosting with tree-based learners
   - High-performance ensemble method

4. **LightGBM**
   - Efficient gradient boosting framework
   - Fast training with leaf-wise growth

5. **CatBoost**
   - Gradient boosting with categorical feature support
   - Robust to overfitting

#### Custom GARCH Integration
- Implemented `ExplainableGARCH` transformer
- Extracts conditional volatility and variance features
- Provides standardized residuals for enhanced model inputs

#### Pipeline Architecture
Each model is wrapped in a scikit-learn pipeline:
- **Preprocessing:** `StandardScaler` or `QuantileTransformer` for normalization
- **Dimensionality Reduction:** `TruncatedSVD` for linear models
- **Modeling:** Various regression algorithms

### 4. Cross-Validation Strategy

- **Method:** Time Series Split (5 folds)
- **Evaluation Metrics:**
  - Weighted R² Score
  - Weighted MAPE (Mean Absolute Percentage Error)
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)

### 5. Experiment Tracking

- **MLflow Integration:** All experiments logged and tracked
- **Artifacts Saved:**
  - Model performance metrics
  - Residual analysis plots
  - Residual distribution visualizations
  - Time series residual plots

## Key Results

### Model Performance Comparison

| Model     | Weighted R² | Weighted MAPE | RMSE    | MAE     |
|-----------|-------------|---------------|---------|---------|
| LightGBM  | 0.318        | 559.64%       | 0.686   | 0.436   |
| CatBoost  | 0.317        | 578.31%       | 0.686   | 0.437   |
| XGBoost   | 0.313        | 588.96%       | 0.689   | 0.440   |
| Ridge     | 0.048        | 310.97%       | 0.812   | 0.544   |
| Lasso     | 0.048        | 310.36%       | 0.812   | 0.544   |

### Observations

1. **Best Performance:** LightGBM consistently outperforms other models across most metrics
2. **Ensemble Methods:** Gradient boosting approaches (XGBoost, LightGBM, CatBoost) significantly outperform linear models
3. **Residual Patterns:** All tree-based models show relatively low residual mean and standard deviation
4. **MAPE Variation:** High MAPE values indicate some challenges with extreme value prediction

## Key Python Libraries

- **Data Processing:** pandas, numpy, polars
- **Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn, xgboost, lightgbm, catboost
- **Time Series:** arch (GARCH models), hmmlearn, statsmodels
- **Preprocessing:** sklearn.preprocessing, sklearn.impute
- **Model Explainability:** shap
- **Experiment Tracking:** mlflow

## Usage

1. **Data Loading:** Load Train.csv and Test.csv files
2. **EDA:** Run exploratory analysis to understand data patterns
3. **Feature Engineering:** Apply `feature_engineer()` function to create derived features
4. **Model Training:** Select and train models from the pipeline
5. **Evaluation:** Use cross-validation for robust performance assessment
6. **Prediction:** Generate predictions on test data
7. **Submission:** Create submission.csv with predictions

## Technical Notes

- **GPU Acceleration:** Can be enabled for XGBoost and LightGBM for faster training
- **Memory Optimization:** Large dataset handling with pandas optimizations
- **Parallel Processing:** Multi-threading support for faster model training
- **Reproducibility:** Random seeds set for consistent results

## Future Improvements

- **Hyperparameter Optimization:** Implement systematic hyperparameter tuning (GridSearchCV or Optuna)
- **Deep Learning:** Explore LSTM networks for temporal sequence modeling
- **Feature Selection:** Further reduce feature space for better generalization
- **Ensemble:** Create weighted ensembles of top-performing models
- **Real-time Monitoring:** Implement model performance monitoring in production

## Author

Developed as part of a financial time series prediction project using advanced machine learning techniques and comprehensive feature engineering.
