# Store Sales Time Series Forecasting

A comprehensive time series forecasting project utilizing both classical statistical methods and state-of-the-art deep learning models to predict retail store sales.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Key Features](#key-features)

## Overview

This project addresses the challenge of forecasting daily sales across multiple store locations and product families. The analysis encompasses comprehensive data preprocessing, feature engineering, and the implementation of both traditional statistical methods and advanced deep learning architectures to achieve accurate sales predictions.

## Dataset

**Source:** [Kaggle Store Sales Time Series Forecasting Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

The dataset includes:
- **train.csv**: Historical sales data with dates, store numbers, product families, and promotion information
- **test.csv**: Test set for generating predictions
- **stores.csv**: Store metadata including location, type, and cluster information
- **oil.csv**: Daily oil prices (relevant for Ecuador's economy)
- **holidays_events.csv**: Holiday and event information
- **transactions.csv**: Daily transaction counts per store

**Data Characteristics:**
- 1,782 individual time series (54 stores × 33 product families)
- Training period: 2013-01-01 to 2017-08-15
- Prediction horizon: 16 days

## Project Structure

```
Store-Sales-Time-Series-Forecasting/
│
├── timeseries-basic-method.ipynb    # Classical statistical methods (ARIMA, Exponential Smoothing)
├── time-series-deep-learning.ipynb  # Deep learning models (LSTM, N-HiTS, TFT)
└── README.md
```

## Methodology

### 1. Data Preprocessing
- Merged multiple data sources (sales, stores, oil prices, holidays, transactions)
- Handled missing values and duplicates
- Feature engineering for temporal patterns
- Data transformation pipeline with log transformation and scaling

### 2. Feature Engineering

**Temporal Features:**
- Year, month, day, day of year
- Day of week, week of year
- Linear time trend

**External Covariates:**
- Oil prices with 7-day and 28-day moving averages
- Store-specific transactions
- Holiday indicators (national, local, events)
- Promotion data with moving averages

**Past Covariates:**
- Sales moving averages (7-day and 28-day windows)
- Transaction patterns
- Historical promotion effects

### 3. Model Development

**Classical Methods (timeseries-basic-method.ipynb):**
- Stationarity testing (ADF, KPSS)
- Autocorrelation analysis (ACF, PACF)
- ARIMA modeling with auto-parameter selection
- Exponential smoothing variants (Simple, Holt, Holt-Winters)

**Deep Learning Approaches (time-series-deep-learning.ipynb):**
- Data preparation for global forecasting models
- Hyperparameter optimization using Optuna
- Multi-variate time series modeling with covariates
- Model validation and backtesting

## Models Implemented

### Classical Statistical Models

1. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Auto-tuned parameter selection
   - Order: (0, 1, 2) identified via auto_arima

2. **Exponential Smoothing Methods**
   - Simple Exponential Smoothing
   - Holt's Linear Trend Method
   - Holt-Winters Multiplicative

### Deep Learning Models

1. **LSTM (Long Short-Term Memory)**
   - Input chunk length: 131
   - Hidden dimensions: 39
   - Number of layers: 3
   - Learning rate: 0.00199
   - **Mean RMSLE: 0.40007**

2. **N-HiTS (Neural Hierarchical Interpolation for Time Series)**
   - Input chunk length: 164
   - Number of stacks: 2
   - Number of blocks: 3
   - Layer width: 128
   - **Mean RMSLE: 0.39741** ⭐ Best Performance

3. **TFT (Temporal Fusion Transformer)**
   - Input chunk length: 230
   - Hidden size: 16
   - LSTM layers: 3
   - Attention heads: 4
   - **Mean RMSLE: 0.41027**

## Results

| Model | Mean RMSLE | Performance Rank |
|-------|------------|------------------|
| N-HiTS | 0.39741 | 🥇 1st |
| LSTM | 0.40007 | 🥈 2nd |
| TFT | 0.41027 | 🥉 3rd |

All models were evaluated using Root Mean Squared Logarithmic Error (RMSLE) across 1,782 time series, with a validation horizon of 16 days.

## Requirements

### Core Libraries
```
pandas
numpy
scipy
scikit-learn
statsmodels
pmdarima
```

### Deep Learning
```
torch
pytorch-lightning
darts
optuna
```

### Visualization
```
matplotlib
seaborn
```

## Usage

### Classical Methods

```python
# Load and prepare data
import pandas as pd
from pmdarima.arima import auto_arima

data = pd.read_csv('train.csv')
# ... preprocessing steps ...

# Fit ARIMA model
model = auto_arima(data, freq='D')
predictions = model.predict(n_periods=16)
```

### Deep Learning Models

```python
from darts.models import NHiTSModel
from darts import TimeSeries

# Prepare time series data
series = TimeSeries.from_dataframe(df, time_col='date', value_cols='sales')

# Train N-HiTS model
model = NHiTSModel(
    input_chunk_length=164,
    output_chunk_length=16,
    num_stacks=2,
    num_blocks=3
)
model.fit(series, past_covariates=covariates)

# Generate forecasts
forecast = model.predict(n=16)
```

## Key Features

- **Multi-variate Forecasting**: Incorporates external factors (oil prices, holidays, promotions)
- **Hierarchical Modeling**: Handles 1,782 individual time series across stores and product families
- **Automated Hyperparameter Tuning**: Optuna-based optimization for deep learning models
- **Comprehensive Data Pipeline**: Includes missing value handling, feature transformation, and scaling
- **Zero-Forecasting Logic**: Handles special cases where historical sales are zero
- **Model Comparison**: Systematic evaluation of classical and deep learning approaches

## Acknowledgments

This project is based on the [Kaggle Store Sales Time Series Forecasting Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).