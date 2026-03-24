# Probabilistic Wind Power Forecaster for Day-Ahead Electricity Trading

## Project Overview

This project develops a highly robust **Probabilistic Wind Power Forecasting** system tailored specifically for the **Day-Ahead Electricity Market**.

In real-world electricity spot markets, traditional point forecasting (minimizing RMSE/MAE) is fundamentally flawed because it fails to quantify the uncertainty of weather-driven generation. This project implements **Quantile Regression** to output a full spectrum of probabilistic scenarios (P10, P50, P90). By accurately predicting the asymmetric risk of under-generation vs. over-generation, this model empowers automated trading algorithms to dynamically formulate bidding strategies and minimize grid **Imbalance Penalties**.

The modeling pipeline leverages the authoritative **GEFCom2014 Wind Track** dataset, utilizing pure exogenous Numerical Weather Predictions (NWP) from ECMWF.

## Feature Engineering: A Purely Statistical Approach

Instead of relying on rigid aerodynamic mechanics, I adopted a data-driven, statistical approach to uncover critical non-linear relationships:

- **Polynomial Kinematics (`WS_100_cubed`)**: Engineered the cube of wind speed ($v^3$). Pearson correlation analysis and mutual information confirmed its statistical dominance in capturing the theoretical kinetic energy formula, without imposing strict physical bounds.
- **Cyclical Trigonometric Encoding**: Transformed `WindDirection` (0°-360°) and `Timestamp` (Hour, Month) into continuous spatial features using `sine` and `cosine` projections. This elegant technique completely eliminates the numerical discontinuity at midnight (23h to 0h) and naturally captures **thermodynamic diurnal patterns** (e.g., thermal convection vs. nocturnal low-level jets).
- **Vertical Wind Profile**: Incorporated NWP data from both 10m and 100m heights to implicitly allow the models to derive atmospheric stability and the wind shear index.

## Methodology & Architecture

The project employs a modular, enterprise-grade pipeline (`src/data_pipeline.py` -> `src/models/` -> `src/evaluate.py`). I designed a progressive modeling strategy to validate the necessity of non-linear architectures:

1. **Statistical Baseline (`statsmodels`)**: **Linear Quantile Regression**. Established the baseline Pinball Loss. It struggled to fit the truncated power curves (e.g., constant power at rated speed).
2. **Core ML (`LightGBM`)**: **Gradient Boosting with Quantile Objective**. Directly optimized for the asymmetric loss. Captured the non-linear "hard thresholds" perfectly.
3. **Deep Learning (`PyTorch`)**: **LSTM with Custom Tensor Loss**. Designed a sliding-window sequence model and implemented a custom tensor-based Pinball Loss function for backpropagation.

## Performance Evaluation (Out-of-Time Testing)

The models were rigorously evaluated on a **future 30-day continuous period**

**Evaluation Metric: Pinball Loss (Quantile Loss)**
*(Lower score is better. Targeting the 10th, 50th, and 90th percentiles)*


| Model                                     | Architecture Nature                  | Mean Pinball Loss | Business Implication                                                                              |
| :---------------------------------------- | :----------------------------------- | :---------------- | :------------------------------------------------------------------------------------------------ |
| **Climatology Benchmark**                 | Unconditional Empirical Distribution | 0.0614            | historical probability distribution independent of any features.                                  |
| **Linear Baseline (Linear Quantile Reg)** | Linear / Parametric                  | 0.0391            | ~36% improvement over Climatology Benchmark.                                                      |
| **PyTorch LSTM**                          | Deep Seq2Seq                         | 0.0375            | Outperforms linear, but limited by the lack of autoregressive lags in day-ahead NWP tabular data. |
| **LightGBM**                              | Non-Linear Tree                      | **0.0344** 🥇     | **Optimal choice. ~12% improvement over baseline.** Excels at hard-threshold interactions.        |

*( **Note on State-of-the-Art Context**: The absolute winning score of the global GEFCom2014 Wind Track was **0.038**.)*

### Analytical Insight: Why did LightGBM outperform LSTM?

Despite the immense power of Deep Learning, LightGBM emerged as the victor on this specific tabular dataset.
In Day-Ahead forecasting, the lack of real-time autoregressive inputs ($T-1$ power) renders the LSTM's sequential memory less effective. The process is overwhelmingly driven by exogenous weather states at the exact target hour. Furthermore, wind power curves exhibit severe non-linear truncations (e.g., mandatory cut-out at 25m/s). Neural networks, designed for smooth continuous function approximation, struggle with these abrupt step-functions without overfitting. Conversely, the `IF-ELSE` splitting mechanism of Decision Trees natively excels at isolating these hard thresholds and tabular cross-features.

### Trading Strategy Implications (Why P10 and P90?)

- **High Penalty Scenario**: If the grid imposes a severe Imbalance Penalty for *under-generation*, trading algorithms must bid conservatively using the **P10 forecast** (90% confidence to deliver the promised MW).
- The predicted quantile distribution directly feeds into linear programming / stochastic solvers to maximize expected financial returns.

## Repository Structure

```text
wind_power_Forecaster/
├── data/                                   # Raw GEFCom2014-W Task 15 dataset
├── notebooks/
│   └── Exploratory_Data_Analysis.ipynb
├── src/
│   ├── models/
│   │   ├── baselines.py
│   │   ├── lgbm_model.py
│   │   └── lstm_model.py
│   ├── utils/
├── results/   
├── README_en.md
├── README_zh.md
└── requirements.txt
```

```

```
