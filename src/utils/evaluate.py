import numpy as np

def pinball_loss(y_true, y_pred, quantile):
    """
    计算 Pinball Loss (分位数损失函数)。这是GEFCom2014 官方使用的评估标准。
    
    参数:
    y_true: 真实功率值 (实际发电量)
    y_pred: 预测的功率分位数值
    quantile: 目标分位数 (如 0.1, 0.5, 0.9)

    """
    error = y_true - y_pred
    # 非对称惩罚：根据误差的正负和设定的分位数进行加权
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    
    # 平均 Pinball Loss
    return np.mean(loss)
