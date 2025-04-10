import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import t
import polars as pl


day = 1695
data = pd.read_parquet(f'../processed_data/Factor_compute_{day}.parquet')

# 对每一列因子进行t-test
def t_test(data, column):
    # Perform t-test
    mean = data[column].mean()
    std = data[column].std(ddof=1)
    n = len(data[column])
    t_statistic = mean / (std / np.sqrt(n))
    # Calculate the p-value
    p_value = 2 * (1 - t.cdf(abs(t_statistic), df=n-1))
    return t_statistic, p_value, mean, std
    # Create a DataFrame from the list of tuples
def compute_t_test(data, feature_columns):
    results = []
    for column in tqdm(feature_columns):
        t_statistic, p_value, mean, std= t_test(data, column)
        results.append((column, mean, std, t_statistic, p_value))
    return pd.DataFrame(results, columns=['factor', 'mean', 'std','t_statistic','p_value'])

# 计算t-test
factor_columns = [f'factor_{i:02d}' for i in range(79)]
t_test_results = compute_t_test(data, factor_columns)

# 排序p值
t_test_results = t_test_results.sort_values(by='p_value')
# 保存结果
t_test_results.to_csv(f'../processed_data/t_test_results_{day}.csv')