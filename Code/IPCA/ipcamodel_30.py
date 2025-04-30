import pandas as pd  
import numpy as np
from ipca import ipca

# 读取parquet文件  
df = pd.read_parquet('../data/input/Processed_data_1600.parquet')  
df=df[df['date_id'] >= 1668]
df['datetime_id'] = df['date_id'].astype(str) + '-' + df['time_id'].astype(str) 
# 调整 datetime_id 为第一列  
cols = ['datetime_id'] + [col for col in df.columns if col != 'datetime_id']  
df = df[cols]  
responder_cols = [col for col in df.columns if col.startswith('responder_') and col != 'responder_6']  
cols_to_drop = responder_cols + ['date_id', 'time_id']  
df = df.drop(columns=cols_to_drop) 

# 1. 选取特征和目标
features = [f'feature_{i:02d}' for i in range(79)]
target = 'responder_6'

# 2. 按日期排序并分割训练/测试集
df = df.sort_values('datetime_id')
unique_dates = df['datetime_id'].unique()
n_train = int(0.8 * len(unique_dates))
train_dates = unique_dates[:n_train]
test_dates = unique_dates[n_train:]

train_df = df[df['datetime_id'].isin(train_dates)]
test_df = df[df['datetime_id'].isin(test_dates)]

# 3. 构建多级索引（datetime_id, symbol_id）
train_df = train_df.set_index(['datetime_id', 'symbol_id'])
test_df = test_df.set_index(['datetime_id', 'symbol_id'])

# 3.1 对特征做横截面zscore
from scipy.stats import zscore  
def cross_section_zscore(group):  
    return group.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))  
# 只对特征做横截面zscore  
train_df[features] = train_df.groupby('datetime_id')[features].transform(  
    lambda x: (x - x.mean()) / (x.std() + 1e-8)  
)  
test_df[features] = test_df.groupby('datetime_id')[features].transform(  
    lambda x: (x - x.mean()) / (x.std() + 1e-8)  
)  

# 4. 构建 RZ
RZ_train = train_df[[target] + features]
RZ_test = test_df[[target] + features]

# 5. 建模与拟合
model = ipca(RZ=RZ_train, return_column=target)
results = model.fit(K=3, OOS=False)

# 6. 在测试集上预测
Gamma = results['Gamma']
Factor = results['Factor']
# 预测值：Beta = Z * Gamma，y_hat = Beta * Factor
Z_test = RZ_test[features]
if model.add_constant:
    Z_test = np.concatenate([Z_test.values, np.ones((Z_test.shape[0], 1))], axis=1)
    Gamma_mat = Gamma.values
else:
    Gamma_mat = Gamma.values
Beta_test = Z_test @ Gamma_mat

# Ensure the type of idx[0] matches Factor.columns' dtype
date_keys = [str(idx[0]) for idx in RZ_test.index]
factor_idx = Factor.columns.get_indexer(date_keys)
Factor_test = Factor.values[:, factor_idx].T

y_pred = np.sum(Beta_test * Factor_test, axis=1)

# 7. 计算测试集加权R^2，使用df['weight']作为权重
y_true = RZ_test[target].values
w = test_df.loc[RZ_test.index, 'weight'].values  # 保证索引对齐
R2 = 1 - np.sum(w * (y_true - y_pred) ** 2) / np.sum(w * y_true ** 2)
pd.DataFrame({'Test_Weighted_R2': [R2]}).to_csv('output_R2.csv', index=False)  