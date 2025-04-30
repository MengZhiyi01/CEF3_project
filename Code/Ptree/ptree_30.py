import pandas as pd  
import numpy as np
from P_tree import PTree
from P_tree_with_split_info_and_weights import PTreeWithSplitInfoAndWeights
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

# 3. 训练ptree模型
# 假设前几列为['symbol_id', target, 'datetime_id', 'weight']，特征从第index列开始
index = train_df.columns.get_loc(features[0])
train_data = train_df[['symbol_id', target, 'datetime_id', 'weight'] + features].copy()
train_data = train_data.rename(columns={
    'symbol_id': 'gvkey',
    target: 'xret',
    'datetime_id': 'date',
    'weight': 'lag_me'
})
train_data['xret']=train_data['xret'].astype(float)
train_data['xret']=train_data['xret']/100

# 标准化每个feature列到0到1范围
def rank_standardize_cross_section(df, features, date_col='date'):  
    df = df.copy()  
    for feature in features:  
        count = df.groupby(date_col)[feature].transform('count')  
        rank = df.groupby(date_col)[feature].rank(method='dense')  
        df[feature] = ((rank - 1) / (count - 1)) * 2 - 1  
        df[feature] = df[feature].fillna(0)  
    return df  

train_data = rank_standardize_cross_section(train_data, features, date_col='date')  

# 训练树
J = 3  # 分裂次数，可根据需要调整
n_min = 2  # 叶节点最小资产数
gamma = 0.0001
m = 3  # 分割点数量
ptree = PTree(train_data, J=J, n_min=n_min, gamma=gamma, m=m, index=index)
leaf_nodes, final_weights, split_info, f_train, max_sharpe = ptree.grow_tree()
