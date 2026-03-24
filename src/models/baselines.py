import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from src.utils.evaluate import pinball_loss
from src.utils.data_processor import load_and_process_data
import warnings
warnings.filterwarnings('ignore') # 忽略统计模型拟合中的一些常规收敛警告

# =========================================================
# 1. 数据读取与合并
# =========================================================
# 训练集是1号风场（zone 1）2年历史数据（2012.01.01-2013.12.01），包含特征和标签
train_file = 'data/Interim/train.csv'
# 测试集是1号风场（zone 1）未来30天（2013.12.01-2014.01.01）的特征和标签
test_X_file = 'data/Interim/test_data.csv'
test_y_file = 'data/Interim/test_label.csv'

# 数据预处理
X_train, y_train, X_val, y_val,X_test, y_test,feature_cols,test_set = load_and_process_data(train_file,test_X_file,test_y_file)

# =========================================================
# 2. 特征归一化
# =========================================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)    #计算 train set 均值和方差，然后做归一化。
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)   #使用 train set 的均值和方差，做归一化。
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)    #使用 train set 的均值和方差，做归一化。

# 手动添加截距项
X_train_sm = X_train_scaled.copy()
X_test_sm = X_test_scaled.copy()

X_train_sm.insert(0, 'const', 1.0)
X_test_sm.insert(0, 'const', 1.0)

# =========================================================
# 3. Baseline 训练: 线性分位数回归
# =========================================================
quantiles = [0.1, 0.5, 0.9]
baseline_preds = pd.DataFrame(index=test_set.index)

print("🚀 开始训练 Baseline (Linear Quantile Regression)...")
for q in quantiles:
    print(f"拟合分位数 q={q} ...")
    model = sm.QuantReg(y_train, X_train_sm)
    res = model.fit(q=q, max_iter=2000)  # 设置上限，防止死循环

    print(f"q={q} 模型特征系数:")
    coefficients = res.params.drop('const').abs().sort_values(ascending=False)
    print(coefficients)

    baseline_preds[f'q_{q}'] = res.predict(X_test_sm)

# =========================================================
# 4. Baseline 评估: Pinball Loss
# =========================================================
print("\n--- Baseline 预测完成，Test Set (未来30天) 评估结果 ---")

losses = {}

for q in quantiles:
    loss_val = pinball_loss(y_test.values, baseline_preds[f'q_{q}'].values, q)
    losses[f'q_{q}'] = loss_val
    print(f"Pinball Loss (q={q}): {loss_val:.4f}")

baselines_mean_loss = np.mean(list(losses.values()))
print(f"==> Baseline 平均 Pinball Loss: {baselines_mean_loss:.4f}")

# =========================================================
# 5. 与官方 Benchmark 对比
# =========================================================
# benchmark 数据导入
benchmark_file = 'data/Interim/benchmark.csv'
df_bench_raw = pd.read_csv(benchmark_file)

# 筛选 Zone 1 并转换时间戳
df_bench_zone1 = df_bench_raw[df_bench_raw['ZONEID'] == 1].copy()
df_bench_zone1['TIMESTAMP'] = pd.to_datetime(df_bench_zone1['TIMESTAMP'])

#  将 Benchmark 与测试集真实数据 (test_set) 对齐
df_eval = pd.merge(test_set[['TIMESTAMP', 'TARGETVAR']], df_bench_zone1, on='TIMESTAMP', how='inner')


# 计算 Official Benchmark Pinball Loss
benchmark_losses = {}

print("\n--- Official Benchmark 评估结果 ---")
for q in quantiles:
    col_name = str(q) if str(q) in df_eval.columns else f"{q:.2f}"
    
    # 提取真实值与 Benchmark 预测值
    y_pred_bench = df_eval[col_name].values
    
    # 计算误差
    loss_val = pinball_loss(y_test.values, y_pred_bench, q)
    benchmark_losses[f'q_{q}'] = loss_val
    print(f"Official Benchmark Pinball Loss (q={q}): {loss_val:.4f}")

benchmark_mean_loss = np.mean(list(benchmark_losses.values()))
print(f"==> Official Benchmark 平均 Pinball Loss: {benchmark_mean_loss:.4f}\n")

# 对比下降百分比
improvement = (benchmark_mean_loss - baselines_mean_loss) / benchmark_mean_loss * 100

print("=========================================================")
print(f"官方 Benchmark 平均误差 : {benchmark_mean_loss:.4f}")
print(f"Baseline 平均误差  : {baselines_mean_loss:.4f}")
print(f"通过引入气象特征并应用线性分位数回归，预测误差下降了 {improvement:.2f}%。")
print("=========================================================")
