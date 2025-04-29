# Author: Xinlei Hao
import pandas as pd
import numpy as np
import gc
import os
from time import time
import pickle
import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
import lightgbm as lgb

from typing import List
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import logging
import warnings
import glob

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class CONFIG:
    seed = 666
    path = "data/"
    save_path = 'processed_data/'
    start_date = 1695 # 设定开始时间


class SpecialCols:
    date_id = "date_id"
    time_id = "time_id"
    symbol_id = "symbol_id"
    weight_col = "weight"
    id_cols = [date_id, time_id, symbol_id]
    target_col = "responder_6"
    target_cols = ["responder_%d" % i for i in range(9)]


class DataProcessor:
    def __init__(self, ts_window=100, start_date=678) -> None:
        logger.info("data preparing")
        self._data_path = os.path.join(CONFIG.path, "train.parquet/**/*.parquet")
        parquet_files = glob.glob(self._data_path, recursive=True)
        self._data = pl.concat([pl.read_parquet(f) for f in parquet_files])
        self._data = (
            #pl.read_parquet(self._data_path)  # 读取数据
            self._data
            .filter(pl.col(SpecialCols.date_id) >= start_date) # 默认选取677天之后的数据
            .with_columns([pl.col(SpecialCols.id_cols)])  # 设置索引
        )

        self._ts_window = ts_window

        print(f'the number of null is:{sum(self._data.null_count().row(0))}')

        gc.collect()
        logger.info("start computing")

    def xs_symbol_id(self, feats: List[str] = None) -> pl.DataFrame:
        '''
        目标：
        对于某个symbol_id的某个feature在某个time_id的nan,按照这个time_id其他symbol_id的这个feature的取值的均值进行填充
        '''
        # 提取所需列，如果 feats 为 None 则复制整个数据
        data = (
            self._data.select(SpecialCols.id_cols+[SpecialCols.weight_col] + feats + SpecialCols.target_cols) # polars 没有索引，需要选取
            if feats is not None
            else self._data.clone()
        )
        # 获取特征列
        feat_cols = [col for col in data.columns if col not in (*SpecialCols.id_cols,SpecialCols.weight_col, *SpecialCols.target_cols)] # *元素解包为独立参数

        # 按照 date_id 和 time_id 分组后，对每个特征的每个time_id计算均值
        # 如果均值计算结果为 NaN，则替换为 null: 因为polars会将 NaN看作浮点数，null_count会检测不出来
        grouped_means = data.group_by([SpecialCols.date_id, SpecialCols.time_id]).agg([
            pl.col(col).mean().fill_nan(None).alias(f"{col}_mean")
            for col in feat_cols
        ]).sort([SpecialCols.date_id, SpecialCols.time_id])

        # 将均值合并回原始 DataFrame
        means_df = grouped_means.join(data, on=[SpecialCols.date_id, SpecialCols.time_id], how="left")

        # 用对应的mean填充，然后删去mean
        means_df = means_df.with_columns([
            pl.when(pl.col(col).is_null()).then(pl.col(f"{col}_mean")).otherwise(pl.col(col)).alias(
                col)
            for col in feat_cols
        ]).drop([col for col in means_df.columns if col.endswith('_mean')])

        self._data = means_df
        gc.collect()
        logger.info("finish xs_symbol_id")
        print(f'the number of null after xs_symbol_id is:{sum(self._data.null_count().row(0))}')
        return self

    def ffill(self, feats: List[str] = None) -> pl.DataFrame:
        '''
        目标：
        对于某个symbol_id的某个feature在某个time_id的nan,按照其前一个time_id的值进行填充
        现在是根据symbol_id分组，顺序排列date_id, time_id，然后前向填充
        '''
        # 提取所需列，如果 feats 为 None 则复制整个数据
        data = (
            self._data.select(SpecialCols.id_cols+[SpecialCols.weight_col] + feats + SpecialCols.target_cols) # polars 没有索引，需要选取
            if feats is not None
            else self._data.clone()
        )
        # 获取特征列
        feat_cols = [col for col in data.columns if col not in (*SpecialCols.id_cols,SpecialCols.weight_col, *SpecialCols.target_cols)] # *元素解包为独立参数

        # 对于每个特征列，进行分组并向前填充: 根据SpecialCols.date_id分组的话就根本填充不了, 有空再看
        # self._data = data.with_columns([
        #     pl.col(col)
        #     .fill_null(strategy="forward")
        #     .over(pl.col(SpecialCols.symbol_id),pl.col(SpecialCols.date_id)).alias(col)
        #     for col in feat_cols
        # ])

        # 排序 DataFrame 确保顺序不乱
        data = data.sort(by=[SpecialCols.symbol_id, SpecialCols.date_id, SpecialCols.time_id])

        # 这里根据symbol_id进行分组
        data = data.with_columns([
            pl.col(col)
            .fill_null(strategy="forward")
            .over(pl.col(SpecialCols.symbol_id)).alias(col)
            for col in feat_cols
        ])

        # 重新按照原来顺序排列
        self._data = data.sort(by=[*SpecialCols.id_cols])

        gc.collect()
        logger.info("finish ffill")
        print(f'the number of null after ffill is:{sum(self._data.null_count().row(0))}')
        return self

    def whole_ave(self, feats: List[str] = None) -> pl.DataFrame:
        '''
        目标：
        对于特征列中的空值，填其
        '''
        # 提取所需列，如果 feats 为 None 则复制整个数据
        data = (
            self._data.select(SpecialCols.id_cols+[SpecialCols.weight_col] + feats + SpecialCols.target_cols) # polars 没有索引，需要选取
            if feats is not None
            else self._data.clone()
        )
        # 获取特征列
        feat_cols = [col for col in data.columns if col not in (*SpecialCols.id_cols,SpecialCols.weight_col, *SpecialCols.target_cols)] # *元素解包为独立参数

        mean_values = data.select(feat_cols).mean()

        # 使用全局均值填充空值

        self._data = data.with_columns([
            pl.when(pl.col(col).is_null()).then(pl.lit(mean_values[col].item())).otherwise(pl.col(col)).alias(col)
            for col in feat_cols
        ])

        gc.collect()
        logger.info("finish whole_ave")
        print(f'the number of null after whole_ave is:{sum(self._data.null_count().row(0))}')
        return self

t_start = time()
# 创建 DataProcessor 实例
dp = DataProcessor(start_date=CONFIG.start_date)

# 连续调用方法
dp = dp.xs_symbol_id().ffill().whole_ave()

data_processed = dp._data

# 将 DataFrame 存储为 Parquet 文件
data_processed.write_parquet(
    os.path.join(CONFIG.save_path, f"Processed_data_{CONFIG.start_date}.parquet")
)
t_end = time()
print(f'Finish filling null, the number of null is:{sum(data_processed.null_count().row(0))} - Time count{t_end - t_start},Processed_data_{CONFIG.start_date}.parquet has been saved')
