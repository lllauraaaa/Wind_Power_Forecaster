import pandas as pd
import numpy as np

def load_and_process_data(train_file,test_X_file,test_y_file):
    df_train_raw = pd.read_csv(train_file)
    df_test_X_raw = pd.read_csv(test_X_file)
    df_test_y_raw = pd.read_csv(test_y_file)

    # 1. 数据读取与合并
    # 对于测试集的真实标签，确保只筛选 Zone 1 的数据并与特征按时间对齐
    df_test_y_zone1 = df_test_y_raw[df_test_y_raw['ZONEID'] == 1].copy()
    df_test_merged = pd.merge(df_test_X_raw, df_test_y_zone1[['TIMESTAMP', 'TARGETVAR']], on='TIMESTAMP', how='inner')


    # 2. 特征工程
    def build_features(df):
        df = df.copy()
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df = df.sort_values('TIMESTAMP').reset_index(drop=True)
        
        # 100m 高度特征
        df['WS_100'] = np.sqrt(df['U100']**2 + df['V100']**2)
        df['WD_100'] = np.arctan2(df['V100'], df['U100']) * (180 / np.pi)
        df['WS_100_cubed'] = df['WS_100'] ** 3
        df['WD_100_sin'] = np.sin(df['WD_100'] * np.pi / 180)
        df['WD_100_cos'] = np.cos(df['WD_100'] * np.pi / 180)
        
        # 10m 高度特征
        df['WS_10'] = np.sqrt(df['U10']**2 + df['V10']**2)
        df['WD_10'] = np.arctan2(df['V10'], df['U10']) * (180 / np.pi)
        df['WS_10_cubed'] = df['WS_10'] ** 3
        df['WD_10_sin'] = np.sin(df['WD_10'] * np.pi / 180)
        df['WD_10_cos'] = np.cos(df['WD_10'] * np.pi / 180)
        
        # 时间周期特征
        df['Hour'] = df['TIMESTAMP'].dt.hour
        df['Month'] = df['TIMESTAMP'].dt.month
        df['Hour_sin'] = np.sin(df['Hour'] * (2 * np.pi / 24))
        df['Hour_cos'] = np.cos(df['Hour'] * (2 * np.pi / 24))
        df['Month_sin'] = np.sin(df['Month'] * (2 * np.pi / 12))
        df['Month_cos'] = np.cos(df['Month'] * (2 * np.pi / 12))
        
        return df

    df_train_fe = build_features(df_train_raw)
    df_test_fe = build_features(df_test_merged)

    # 剔除可能存在的缺失值
    df_train_fe = df_train_fe.dropna()
    df_test_fe = df_test_fe.dropna()

    
    # 3. 数据集切分 (Train / Val / Test)
    feature_cols = [
        'WS_100', 'WS_100_cubed', 'WD_100_sin', 'WD_100_cos',
        'WS_10', 'WS_10_cubed', 'WD_10_sin', 'WD_10_cos',
        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'
    ]
    target_col = 'TARGETVAR'

    # 将2年历史数据切分为 90% Train 和 10% Validation
    split_idx = int(len(df_train_fe) * 0.9)
    train_set = df_train_fe.iloc[:split_idx]
    val_set = df_train_fe.iloc[split_idx:]
    # 30天未来数据作为 test
    test_set = df_test_fe 

    X_train, y_train = train_set[feature_cols], train_set[target_col]
    X_val, y_val = val_set[feature_cols], val_set[target_col]
    X_test, y_test = test_set[feature_cols], test_set[target_col]
    
    return X_train, y_train, X_val, y_val,X_test, y_test,feature_cols,test_set