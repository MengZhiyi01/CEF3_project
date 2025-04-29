# 本仓库用于合并feature和factor

import numpy as np
import pandas as pd
from tqdm import tqdm
import polars as pl

# Step 1: Read data
day = 1600
feature_data = pl.read_parquet(f'../data/output/Processed_data_{day}.parquet').to_pandas()
feature_data = feature_data.set_index(['date_id', 'time_id'])

factor_data = pl.read_parquet(f'../data/output/PCA_Factor_{day}_n40.parquet').to_pandas()

# Step 2: 对factor_data进行滚动平均处理
# 设置回看的天数k
k = 5

# 获取factor_data中所有的列名，除了date_id和time_id
factor_cols = [col for col in factor_data.columns if col not in ['date_id', 'time_id']]

# 按time_id分组，对每个time_id内的数据计算滚动平均
factor_data_rolled = []
for time_id, group in factor_data.groupby('time_id'):
    # 按date_id排序
    group = group.sort_values('date_id')
    
    # 算滚动平均
    for col in factor_cols:
        group[col] = group[col].rolling(window=k+1, min_periods=1).mean()
    
    factor_data_rolled.append(group)

# 合并所有time_id的数据
factor_data = pd.concat(factor_data_rolled, ignore_index=True)
print(f"完成对factor数据的{k}天滚动平均计算")

# Step 3: 对factor_data进行date_id移位操作
# 将factor_data的date_id向后移动一天（+1）
factor_data_shifted = factor_data.copy()
factor_data_shifted['date_id'] = factor_data_shifted['date_id'] + 1  # date_id向后移动一天

# 设置索引以便于合并
factor_data_shifted = factor_data_shifted.set_index(['date_id', 'time_id'])

# Step 4: 合并 feature_data 和 调整后的 factor_data
# 通过索引 date_id 和 time_id 进行合并
merged_data = feature_data.join(factor_data_shifted, how='left')

# 清洗有nan数据行
# 这里可以选择删除含有nan的行
merged_data = merged_data.dropna()

# 确保factor_cols里面的数据全部都是float32类型
merged_data[factor_cols] = merged_data[factor_cols].astype(np.float32)

# Step 5: 保存合并后的数据
merged_data.reset_index().to_parquet(f'../data/output/Fature_factor_data_{day}_n40.parquet')

print('合并完成并成功保存!')
