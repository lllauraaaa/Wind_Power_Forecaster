import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils.evaluate import pinball_loss
from src.utils.data_processor import load_and_process_data
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. 数据准备与滑动窗口构造 (Sliding Window)
# =========================================================
# 训练集是1号风场（zone 1）2年历史数据（2012.01.01-2013.12.01），包含特征和标签
train_file = 'data/Interim/train.csv'
# 测试集是1号风场（zone 1）未来30天（2013.12.01-2014.01.01）的特征和标签
test_X_file = 'data/Interim/test_data.csv'
test_y_file = 'data/Interim/test_label.csv'

# 数据预处理
X_train, y_train, X_val, y_val,X_test, y_test, _, _ = load_and_process_data(train_file,test_X_file,test_y_file)

# 归一化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)


seq_length = 12  # 以过去12小时的数据来预测未来1小时的风电功率

def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y.iloc[i + seq_length]) 
    return np.array(xs), np.array(ys).reshape(-1, 1)

print(f"正在构建序列数据 (Sequence Length: {seq_length}) ...")
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length)

# =========================================================
# 2. PyTorch Dataset 与 DataLoader
# =========================================================
class WindPowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 256 # 适当增大 Batch Size 提速
train_loader = DataLoader(WindPowerDataset(X_train_seq, y_train_seq), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(WindPowerDataset(X_val_seq, y_val_seq), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(WindPowerDataset(X_test_seq, y_test_seq), batch_size=batch_size, shuffle=False)

# =========================================================
# 3. 定义支持多输出的分位数 LSTM 网络
# =========================================================
class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(QuantileLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True 保证输入 shape 为 (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 隐状态初始化
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        # 提取序列最后一个时间步的隐状态送入全连接层
        out = self.fc(out[:, -1, :]) 
        return out

# =========================================================
# 4. PyTorch 张量化的 Pinball Loss 
# =========================================================
def quantile_loss(preds, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        err = target - preds[:, i:i+1]
        loss = torch.max(q * err, (q - 1) * err)
        losses.append(loss)
    # 将3个分位数的loss按特征维度拼接，然后求总均值作为梯度回传信号
    return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

# =========================================================
# 5. 实例化模型并训练
# =========================================================
device = torch.device('cpu') 
print(f"Using device: {device}")

quantiles = [0.1, 0.5, 0.9]
input_size = X_train_seq.shape[2]
hidden_size = 32 
num_layers = 1   
output_size = len(quantiles)

model = QuantileLSTM(input_size, hidden_size, num_layers, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) 

num_epochs = 15 

print("🚀 开始训练 LSTM ...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = quantile_loss(preds, y_batch, quantiles)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        
    train_loss /= len(train_loader.dataset)
    
    # Validation 
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            loss = quantile_loss(preds, y_batch, quantiles)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)
    
    if (epoch+1) % 3 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# =========================================================
# 6. 测试与最终评估 (计算 Test Set 的 Pinball Loss)
# =========================================================
model.eval()
test_preds = []
test_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        test_preds.append(np.array(preds.cpu().tolist()))
        test_targets.append(np.array(y_batch.cpu().tolist()))

test_preds = np.concatenate(test_preds, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

print("\n--- PyTorch LSTM 预测完成，Test Set (未来30天) 评估结果 ---")
lstm_losses = {}

for i, q in enumerate(quantiles):
    err = test_targets - test_preds[:, i:i+1]
    loss_val = np.mean(np.maximum(q * err, (q - 1) * err))
    lstm_losses[f'q_{q}'] = loss_val
    print(f"LSTM Pinball Loss (q={q}): {loss_val:.4f}")

lstm_mean_loss = np.mean(list(lstm_losses.values()))
print(f"==> LSTM 平均 Pinball Loss: {lstm_mean_loss:.4f}")