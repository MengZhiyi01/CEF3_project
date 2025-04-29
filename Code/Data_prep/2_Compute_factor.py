import numpy as np
import pandas as pd
from tqdm import tqdm
import polars as pl

# Step 1: Read data
day = 1600
data = pl.read_parquet(f'../data/output/Processed_data_{day}.parquet').to_pandas()
data = data.set_index(['date_id', 'time_id'])

# Step 2: Define the function to compute the factor
def _get_neutral_signal(group, feature):
    original_index = group.index
    sorted_indices = group[feature].argsort()
    n = len(sorted_indices)
    split_point = n // 2

    signals = np.zeros(n)
    if n % 2 == 0:
        signals[sorted_indices[:split_point]] = -1  # 低半部分设为卖出
        signals[sorted_indices[split_point:]] = 1  # 高半部分设为买入
    else:
        signals[sorted_indices[:split_point]] = -1
        signals[split_point] = 0  # 中位数不交易
        signals[sorted_indices[split_point + 1:]] = 1

    return pd.Series(signals, index=original_index)

def strategy(data, feature_columns):
    # Create a copy to avoid modifying the original data
    data_test = data.copy()
    portfolio_dict = {}

    # 按双重索引分组
    grouped = data_test.groupby(data_test.index, group_keys=False)

    # Process each feature column
    for feature in tqdm(feature_columns):
        # 为每个特征创建独立的apply函数
        def feature_processor(f):
            return lambda x: _get_neutral_signal(x, f)

        # 生成信号列
        suffix = feature.split('_')[-1]
        print(f'suffix is {suffix}')
        data_test[f'_temp_signal_{suffix}'] = grouped.apply(feature_processor(feature))

        portfolio_dict[f'factor_{suffix}'] = grouped.apply(
            lambda x: (x['responder_6'] * x[f'_temp_signal_{suffix}']).sum() / len(x),
            include_groups=False
        )

        del data_test[f'_temp_signal_{suffix}']

    portfolio_returns = pd.concat(portfolio_dict, axis=1)

    # 拆分索引列为独立的date_id和time_id
    portfolio_returns = portfolio_returns.reset_index()

    # 拆分元组列（如果存在）
    if 'index' in portfolio_returns.columns:
        # 将元组拆分为两列
        portfolio_returns[['date_id', 'time_id']] = pd.DataFrame(
            portfolio_returns['index'].tolist(),
            index=portfolio_returns.index
        )
        # 删除原始元组列
        portfolio_returns.drop(columns=['index'], inplace=True)
        # 调整列顺序
        cols = ['date_id', 'time_id'] + [c for c in portfolio_returns.columns if c not in ['date_id', 'time_id']]
        portfolio_returns = portfolio_returns[cols]

    return portfolio_returns

features = [f'feature_{i:02d}' for i in range(79)]
factor_compute = strategy(data, features)  # Skip the first two columns (date_id and time_id)
# Save the result to a parquet file
print('Saving the result to parquet file')
factor_compute.to_parquet(f'../data/output/Factor_compute_{day}.parquet')