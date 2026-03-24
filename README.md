# Wind_Power_Forecaster

# 🌬️ Probabilistic Wind Power Forecaster for Day-Ahead Trading

## 📊 Data Source
- Dataset: GEFCom2014 (Global Energy Forecasting Competition 2014) - Wind Forecasting Track.
- Provider: IEEE Working Group on Energy Forecasting (WGEF).
- Description: Contains hourly Numerical Weather Predictions (NWP) from ECMWF (at 10m and 100m heights) and normalized wind power generation data.

## 📊 Performance (Out-of-Time Test)
Evaluated on a strict hold-out 30-day future set (GEFCom2014 Wind Track data). 
**Metric: Mean Pinball Loss** (Lower is better).

| Model | Nature | Mean Pinball Loss |
| :--- | :--- | :--- |
| Baseline | Linear Regression | 0.0391 |
| PyTorch LSTM | Deep Learning | 0.0375 |
| **LightGBM** | **Tree Ensemble** | **0.0344** 🥇 |

## 📁 Repository Structure
- `data/`: Raw NWP and power datasets (ignored via `.gitignore`).
- `notebooks/`: Comprehensive EDA, Feature Importance, and Power Curve non-linear analysis.
- `src/`:
  - `data_pipeline.py`: Unified feature engineering and temporal Train/Val/Test splitting.
  - `models/`: Implementations of LightGBM, LSTM, and Baseline.
  - `evaluate.py`: Standardized Pinball Loss evaluation.
- `results/`: Output predictions and feature importance plots.

## 💻 Quick Start
```bash
pip install -r requirements.txt
python src/data_pipeline.py
python src/models/lgbm_model.py
python src/evaluate.py
