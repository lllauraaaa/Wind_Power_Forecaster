import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from src.utils.evaluate import pinball_loss
from src.utils.data_processor import load_and_process_data

# =========================================================
# 1. 数据读取与合并
# =========================================================
# 训练集是1号风场（zone 1）2年历史数据（2012.01.01-2013.12.01），包含特征和标签
train_file = 'data/Interim/train.csv'
# 测试集是1号风场（zone 1）未来30天（2013.12.01-2014.01.01）的特征和标签
test_X_file = 'data/Interim/test_data.csv'
test_y_file = 'data/Interim/test_label.csv'

# 数据预处理
X_train, y_train, X_val, y_val,X_test, y_test, _, _ = load_and_process_data(train_file,test_X_file,test_y_file)

# =========================================================
# 2. 训练三个分位数模型（P10,P50,P90），并加入 Validation Set 进行 early stopping
# =========================================================
print("🚀 开始训练 LightGBM ...")

quantiles = [0.1, 0.5, 0.9]
lgbm_preds = pd.DataFrame(index=X_test.index)
models = {} 

for q in quantiles:
    print(f"拟合 LightGBM 分位数 q={q} ...")
    
    # 构建 LightGBM 数据集格式
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'quantile',
        'alpha': q,
        'learning_rate': 0.05,
        'num_leaves': 31,      # 树的复杂度，太大容易过拟合
        'max_depth': -1,
        'feature_fraction': 0.8, # 每次分裂随机选择80%的特征，防过拟合
        'seed': 42,
        'verbose': -1          # 关闭训练日志
    }
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False), # 如果验证集误差 50 轮不下降就停止
        lgb.log_evaluation(period=0) # 关闭每轮的训练日志
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000, # 最大迭代次数，靠 early stopping 截断
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    models[f'q_{q}'] = model
    
    # test 预测
    lgbm_preds[f'q_{q}'] = model.predict(X_test, num_iteration=model.best_iteration)
    print(f"  --> q={q} 最佳迭代次数: {model.best_iteration}")

# =========================================================
# 3. light GBM 评估: Pinball Loss
# =========================================================
print("\n--- LightGBM 预测完成，Test Set (未来30天) 评估结果 ---")
lgbm_losses = {}
for q in quantiles:
    loss_val = pinball_loss(y_test.values, lgbm_preds[f'q_{q}'].values, q)
    lgbm_losses[f'q_{q}'] = loss_val
    print(f"LGBM Pinball Loss (q={q}): {loss_val:.4f}")

lgbm_mean_loss = np.mean(list(lgbm_losses.values()))
print(f"==> LGBM 平均 Pinball Loss: {lgbm_mean_loss:.4f}")

# =========================================================
# 4. 特征重要性对比图
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 16))

for ax, q in zip(axes, [0.1, 0.5, 0.9]):
    lgb.plot_importance(
        models[f"q_{q}"],
        importance_type="gain",
        ax=ax,
        title=f"P{int(q*100)} Feature Importance"
    )

plt.tight_layout()
plt.savefig("results/lgbm_feature_importance", dpi=300, bbox_inches="tight")
plt.show()