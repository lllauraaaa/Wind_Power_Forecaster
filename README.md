# 面向日前电力现货交易的概率型风电预测系统

## 📊数据来源
数据集：GEFCom2014 (2014年全球能源预测大赛) - 风电预测赛道。
发布方：IEEE 电能与能源协会 (PES) 能源预测工作组 (WGEF)。
数据内容：包含欧洲中期天气预报中心 (ECMWF) 提供的 10m 和 100m 高度每小时气象预报 (NWP) 以及归一化后的风电实际出力数据。
使用范围：本项目采用 Task 15 数据（含 2 年历史训练集及 1 个月的严格时序测试集），以精准模拟现实中的日前交易盲测场景。

## 📌 项目概述
本项目采用**分位数回归（Quantile Regression）**输出 P10（保底）、P50（中性）、P90（乐观）发电量概率分布，为量化交易算法提供核心的风险约束，从而最大化期望报价利润。

## 🚀 核心技术亮点
- **物理机理与统计学融合**：通过三角函数（sin/cos）对时间与风向进行连续性编码以捕获热力学日内模式；引入风速多项式特征（$v^3$）以贴合动力学能量转换。
- **多范式对比架构**：独立实现了线性基准（`statsmodels`）、深度序列模型（`PyTorch LSTM`），以及非线性树模型（`LightGBM`）。

## 📊 性能表现 (Out-of-Time 严格盲测)
在未来 30 天连续时间段上进行预测（基于 GEFCom2014 数据集）。
**评估指标: 平均 Pinball Loss / 分位数损失** (越低越好)。

| 模型架构 | 本质特性 | 平均 Pinball Loss |
| :--- | :--- | :--- |
| Baseline | 线性 / 参数化 | 0.0391 |
| PyTorch LSTM | 深度序列学习 | 0.0375 |
| **LightGBM** | **非线性树模型集成** | **0.0344** 🥇 |

## 📁 代码库结构
- `data/`: 原始气象预报与实际功率数据。
- `notebooks/`: 探索性数据分析（EDA）、特征相关性矩阵与风功率曲线截断分析。
- `src/`:
  - `data_pipeline.py`: 统一特征工程与严格时序切分（Train/Val/Test）。
  - `models/`: 包含 LightGBM, LSTM 及 Baseline 的核心训练脚本。
  - `evaluate.py`: 统一读取各模型预测结果并计算战绩榜。
- `results/`: 存放各模型输出的预测 CSV 及特征重要性图表。

## 💻 快速运行
```bash
pip install -r requirements.txt
python src/data_pipeline.py
python src/models/lgbm_model.py
python src/evaluate.py
```bash


# Wind_Power_Forecaster

# Probabilistic Wind Power Forecaster for Day-Ahead Trading

## 📌 Overview
This system utilizes **Quantile Regression** to output P10, P50, and P90 power scenarios. 

## 📊 Data Source
- Dataset: GEFCom2014 (Global Energy Forecasting Competition 2014) - Wind Forecasting Track.
- Provider: IEEE Working Group on Energy Forecasting (WGEF).
- Description: Contains hourly Numerical Weather Predictions (NWP) from ECMWF (at 10m and 100m heights) and normalized wind power generation data.
- Scope: This project utilizes Task 15 (2 years of historical data for training and 1 month of out-of-time data for blind testing) to simulate real-world day-ahead market scenarios.

## 🚀 Core Highlights
- **Physics-Informed Statistical Features**: Implemented trigonometric cyclical encoding for time/wind direction to capture thermodynamic diurnal patterns, and polynomial kinematics ($v^3$) to capture non-linear energy conversion.
- **Multi-Paradigm Architecture**: Evaluated a Linear Baseline (`statsmodels`), a Deep Seq2Seq model (`PyTorch LSTM` with custom tensor Pinball loss), and Non-Linear Tree Ensembles (`LightGBM`).

## 📊 Performance
Evaluated on a 30-day future set (GEFCom2014 Wind Track data). 
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
