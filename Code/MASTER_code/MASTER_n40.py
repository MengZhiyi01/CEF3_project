# Author of this notebook: Xinlei Hao
import argparse
from typing import Optional
import copy
import torch
import torch.optim as optim


import numpy as np
import pandas as pd
import polars as pl

from typing import Tuple, Union, List
from copy import deepcopy
from tqdm import tqdm
import bisect
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math

import pickle
import os
import json
from itertools import product

import warnings
warnings.filterwarnings("ignore")

def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    idx = df.index if axis == 0 else df.columns
    if (
        not idx.is_monotonic_increasing
        and isinstance(idx, pd.MultiIndex)
        and not idx.is_lexsorted()
    ):
        return df.sort_index(axis=axis)
    else:
        return df

def np_ffill(arr: np.array):
    mask = np.isnan(arr.astype(float))  # np.isnan only works on np.float
    # get fill index
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

# v0.8.6
class TSDataSampler:
    """
    (T)ime-(S)eries DataSampler
    This is the processed_data of TSDatasetH

    It works like `torch.data.utils.Dataset`, it provides a very convenient interface for constructing time-series
    dataset based on tabular data.
    - On time step dimension, the smaller index indicates the historical data and the larger index indicates the future
      data.

    If user have further requirements for processing data, user could process them based on `TSDataSampler` or create
    more powerful subclasses.

    Known Issues:
    - For performance issues, this Sampler will convert dataframe into arrays for better performance. This could processed_data
      in a different data type

    """

    def __init__(
            self, data: pd.DataFrame, start, end, step_len: int, fillna_type: str = "none", dtype=None, flt_data=None
    ):
        """
        Build a dataset which looks like torch.data.utils.Dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The raw tabular data
        start :
            The indexable start time
        end :
            The indexable end time
        step_len : int
            The length of the time-series step
        fillna_type : int
            How will qlib handle the sample if there is on sample in a specific date.
            none:
                fill with np.nan
            ffill:
                ffill with previous sample
            ffill+bfill:
                ffill with previous samples first and fill with later samples second
        flt_data : pd.Series
            a column of data(True or False) to filter data.
            None:
                kepp all data

        """
        self.start = start
        self.end = end
        self.step_len = step_len
        self.fillna_type = fillna_type
        # assert get_level_index(data, "datetime") == 0
        self.data = lazy_sort_index(data)

        kwargs = {"object": self.data}
        if dtype is not None:
            kwargs["dtype"] = dtype

        self.data_arr = np.array(**kwargs)  # Get index from numpy.array will much faster than DataFrame.values!
        # NOTE:
        # - append last line with full NaN for better performance in `__getitem__`
        # - Keep the same dtype will processed_data in a better performance
        self.data_arr = np.append(
            self.data_arr, np.full((1, self.data_arr.shape[1]), np.nan, dtype=self.data_arr.dtype), axis=0
        )
        self.nan_idx = -1  # The last line is all NaN

        # the data type will be changed
        # The index of usable data is between start_idx and end_idx
        self.idx_df, self.idx_map = self.build_index(self.data)
        self.data_index = deepcopy(self.data.index)

        if flt_data is not None:
            if isinstance(flt_data, pd.DataFrame):
                assert len(flt_data.columns) == 1
                flt_data = flt_data.iloc[:, 0]
            # NOTE: bool(np.nan) is True !!!!!!!!
            # make sure reindex comes first. Otherwise extra NaN may appear.
            flt_data = flt_data.reindex(self.data_index).fillna(False).astype(np.bool)
            self.flt_data = flt_data.values
            self.idx_map = self.flt_idx_map(self.flt_data, self.idx_map)
            self.data_index = self.data_index[np.where(self.flt_data)[0]]
        self.idx_map = self.idx_map2arr(self.idx_map)

        # self.start_idx, self.end_idx = self.data_index.slice_locs(
        #     start=time_to_slc_point(start), end=time_to_slc_point(end)
        # )
        # self.start_idx, self.end_idx = 0, len(self.idx_map) # 这里进行了非常粗暴的改动
        # Find the index positions of start and end in the level 0 index
        # 这里下面两行进行了改动
        self.start_idx = self.data_index.get_level_values(0).searchsorted(start, side='left')
        self.end_idx = self.data_index.get_level_values(0).searchsorted(end, side='right')

        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)  # for better performance

        del self.data  # save memory

    @staticmethod
    def idx_map2arr(idx_map):
        # pytorch data sampler will have better memory control without large dict or list
        # - https://github.com/pytorch/pytorch/issues/13243
        # - https://github.com/airctic/icevision/issues/613
        # So we convert the dict into int array.
        # The arr_map is expected to behave the same as idx_map

        dtype = np.int32
        # set a index out of bound to indicate the none existing
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)

        max_idx = max(idx_map.keys())
        arr_map = []
        for i in range(max_idx + 1):
            arr_map.append(idx_map.get(i, no_existing_idx))
        arr_map = np.array(arr_map, dtype=dtype)
        return arr_map

    @staticmethod
    def flt_idx_map(flt_data, idx_map):
        idx = 0
        new_idx_map = {}
        for i, exist in enumerate(flt_data):
            if exist:
                new_idx_map[idx] = idx_map[i]
                idx += 1
        return new_idx_map

    def get_index(self):
        """
        Get the pandas index of the data, it will be useful in following scenarios
        - Special sampler will be used (e.g. user want to sample day by day)
        """
        return self.data_index[self.start_idx: self.end_idx]

    def config(self, **kwargs):
        # Config the attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def build_index(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        The relation of the data

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe with <datetime, DataFrame>

        Returns
        -------
        Tuple[pd.DataFrame, dict]:
            1) the first element:  reshape the original index into a <datetime(row), instrument(column)> 2D dataframe
                instrument SH600000 SH600004 SH600006 SH600007 SH600008 SH600009  ...
                datetime
                2021-01-11        0        1        2        3        4        5  ...
                2021-01-12     4146     4147     4148     4149     4150     4151  ...
                2021-01-13     8293     8294     8295     8296     8297     8298  ...
                2021-01-14    12441    12442    12443    12444    12445    12446  ...
            2) the second element:  {<original index>: <row, col>}
        """
        # object incase of pandas converting int to float
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        idx_df = lazy_sort_index(idx_df.unstack())
        # NOTE: the correctness of `__getitem__` depends on columns sorted here
        idx_df = lazy_sort_index(idx_df, axis=1)

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    @property
    def empty(self):
        return len(self) == 0

    def _get_indices(self, row: int, col: int) -> np.array:
        """
        get series indices of self.data_arr from the row, col indices of self.idx_df

        Parameters
        ----------
        row : int
            the row in self.idx_df
        col : int
            the col in self.idx_df

        Returns
        -------
        np.array:
            The indices of data of the data
        """
        indices = self.idx_arr[max(row - self.step_len + 1, 0): row + 1, col]

        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])

        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices

    def _get_row_col(self, idx) -> Tuple[int]:
        """
        get the col index and row index of a given sample index in self.idx_df

        Parameters
        ----------
        idx :
            the input of  `__getitem__`

        Returns
        -------
        Tuple[int]:
            the row and col index
        """
        # The the right row number `i` and col number `j` in idx_df
        if isinstance(idx, (int, np.integer)):
            real_idx = self.start_idx + idx
            if self.start_idx <= real_idx < self.end_idx:
                i, j = self.idx_map[real_idx]  # TODO: The performance of this line is not good
            else:
                raise KeyError(f"{real_idx} is out of [{self.start_idx}, {self.end_idx})")
        elif isinstance(idx, tuple):
            # <TSDataSampler object>["datetime", "instruments"]
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            # NOTE: This relies on the idx_df columns sorted in `__init__`
            j = bisect.bisect_left(self.idx_df.columns, inst)
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return i, j

    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int]]):
        """
        # We have two method to get the time-series of a sample
        tsds is a instance of TSDataSampler

        # 1) sample by int index directly
        tsds[len(tsds) - 1]

        # 2) sample by <datetime,instrument> index
        tsds['2016-12-31', "SZ300315"]

        # The return value will be similar to the data retrieved by following code
        df.loc(axis=0)['2015-01-01':'2016-12-31', "SZ300315"].iloc[-30:]

        Parameters
        ----------
        idx : Union[int, Tuple[object, str]]
        """
        # Multi-index type
        mtit = (list, np.ndarray)
        if isinstance(idx, mtit):
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(*self._get_row_col(idx))

        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)

        data = self.data_arr[indices]
        if isinstance(idx, mtit):
            # if we get multiple indexes, addition dimension should be added.
            # <sample_idx, step_idx, feature_idx>
            data = data.reshape(-1, self.step_len, *data.shape[1:])
        return data

    def __len__(self):
        return len(self.idx_map)

# class DailyBatchSamplerRandom(Sampler):
#     def __init__(self, data_source, shuffle=False):
#         self.data_source = data_source
#         self.shuffle = shuffle
#         # calculate number of samples in each batch
#         self.daily_count = pd.Series(index=self.data_source.get_index(), dtype=pd.Float32Dtype()).groupby("time_id").size().values
#         self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
#         self.daily_index[0] = 0
#
#     def __iter__(self):
#         if self.shuffle:
#             index = np.arange(len(self.daily_count))
#             np.random.shuffle(index)
#             for i in index:
#                 yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
#         else:
#             for idx, count in zip(self.daily_index, self.daily_count):
#                 yield np.arange(idx, idx + count)
#
#     def __len__(self):
#         return len(self.data_source)

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle

        # 获取索引并按time_id分组
        index_df = self.data_source.get_index()
        self.daily_count = pd.Series(index=index_df).groupby(level="time_id").size().values

        # 计算每个批次的起始索引
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

        print(f"数据集总样本数: {len(data_source)}")
        print(f"时间分组数量: {len(self.daily_count)}")
        print(f"每个时间组的样本数: 最小 {min(self.daily_count)}, 最大 {max(self.daily_count)}, 平均 {np.mean(self.daily_count):.2f}")

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        # 返回批次数量而不是样本总数
        return len(self.daily_count)


def calc_weighted_r2(pred, label, weight):
    """
    Calculate the weighted R-squared (R2) score.

    Parameters:
    pred (array-like): Predicted values.
    label (array-like): True values.
    weight (array-like): Weights for each sample.

    Returns:
    float: Weighted R-squared score.
    """
    df = pd.DataFrame({'pred': pred, 'label': label, 'weight': weight})
    mask = ~df['label'].isna()
    df = df[mask]

    numerator = np.sum(df['weight'] * (df['label'] - df['pred']) ** 2)
    denominator = np.sum(df['weight'] * df['label'] ** 2)
    weighted_r2 = 1 - numerator / denominator
    return weighted_r2


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path='../models/',
                 save_prefix=''):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() else f"cuda:{GPU}" if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.fitted = False
        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix

    def init_model(self):
        if self.model is None:
            raise ValueError("models has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label, weights):
        '''
        '''
        mask = ~torch.isnan(label)
        pred = pred[mask]
        label = label[mask]
        weights = weights[mask]

        numerator = torch.sum(weights * (label - pred) ** 2)
        denominator = torch.sum(weights * label ** 2)
        r2 = 1 - numerator / denominator

        return -r2  # Since we want to minimize the loss, return the negative R-squared value

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        total_batches = len(data_loader)
        print(f"每个epoch的批次总数: {total_batches}")

        # 实际处理的样本计数
        total_samples = 0

        for batch_data in tqdm(data_loader, desc="训练中", total=total_batches):
            batch_size = batch_data.shape[0]
            total_samples += batch_size
            batch_loss = 0

            for i in range(batch_size):
                data = batch_data[i]
                feature = data[:, :, 1:-1].to(self.device)
                label = data[:, -1, -1].to(self.device)
                weights = data[:, -1, 0].to(self.device)

                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label, weights)
                batch_loss += loss

            batch_loss = batch_loss / batch_size
            losses.append(batch_loss.item())

            self.train_optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        print(f"本epoch实际处理样本数: {total_samples}")
        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for batch_data in tqdm(data_loader, desc="Validation", leave=False):
            batch_size = batch_data.shape[0]
            batch_loss = 0

            for i in range(batch_size):
                data = batch_data[i]
                feature = data[:, :, 1:-1].to(self.device)
                label = data[:, -1, -1].to(self.device)
                weights = data[:, -1, 0].to(self.device)  # Assuming weights are at index 0

                with torch.no_grad():
                    pred = self.model(feature.float())
                    loss = self.loss_fn(pred, label, weights)
                    batch_loss += loss

            # 计算批次的平均损失
            batch_loss = batch_loss / batch_size
            losses.append(batch_loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=False, drop_last=True,batch_size=32):
        '''
        :param data: This is an instance of TSDataSampler, which contains the data and sampling logic.
        :param shuffle: This parameter is used to create the DailyBatchSamplerRandom sampler, indicating whether to shuffle the data at the beginning of each epoch.
        :param drop_last: This parameter indicates whether to drop the last incomplete batch.
        :return: DataLoader instance: DataLoader uses the sampler to generate data indices and loads data from TSDataSampler according to these indices. drop_last=True means that if the data cannot be divided evenly, the last incomplete batch will be dropped.
        '''
        sampler = DailyBatchSamplerRandom(data, shuffle)
        # data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last, batch_size= batch_size)
        return data_loader

    def load_param(self, param_path):
        # self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        # 确保加载路径是基于 base_dir 的绝对路径
        absolute_param_path = os.path.join(base_dir, param_path)
        self.model.load_state_dict(torch.load(absolute_param_path, map_location=self.device))
        self.fitted = True

    def fit(self, dl_train, dl_valid, patience=5, min_delta=0.001, batch_size=32):
        # Initialize training and validation data loaders
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True, batch_size=batch_size)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=True, batch_size=batch_size)

        self.fitted = True  # Mark the models as trained
        best_param = None  # Used to store the best models parameters
        best_val_loss = float('inf')  # Initialize the best validation loss to infinity
        epochs_no_improve = 0  # Record the number of epochs without improvement

        for step in range(self.n_epochs):
            # Train one epoch and calculate the training loss
            train_loss = self.train_epoch(
                train_loader)  # Since we want to minimize the loss, return the negative R-squared value
            # Validate one epoch and calculate the validation loss
            val_loss = self.test_epoch(valid_loader)

            # Print the training and validation loss for the current epoch
            # print("Epoch %d, train_loss_ %.6f, valid_loss %.6f " % (step, train_loss, val_loss))
            print("Epoch %d, train_weighted_r_2 %.6f, valid_weighted_r_2 %.6f " % (step, -train_loss, -val_loss))

            # If the validation loss improves
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss  # Update the best validation loss
                best_param = copy.deepcopy(self.model.state_dict())  # Save the current best models parameters
                epochs_no_improve = 0  # Reset the number of epochs without improvement
            else:
                epochs_no_improve += 1  # Increase the number of epochs without improvement

            # If the number of epochs without improvement reaches the patience value, stop training early
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {step}")
                break

            # torch.save(best_param, f'{self.save_path}{self.save_prefix}master_{step}.pkl')

        # 确保保存目录存在
        os.makedirs(self.save_path, exist_ok=True)

        # 动态生成包含参数的模型文件名
        model_name = (
            f"{self.save_prefix}_"
            f"dmodel{self.d_model}_"
            f"tn{self.t_nhead}_"
            f"sn{self.s_nhead}_"
            f"do{self.T_dropout_rate}_"
            f"bs{batch_size}_"
            f"lr{self.lr}_"
            f"beta{self.beta}"
        )
        # model_path = os.path.join(self.save_path, f"{model_name}.pkl")
        # torch.save(best_param, model_path)
        # 确保保存路径是基于 base_dir 的绝对路径
        absolute_save_path = os.path.join(base_dir, self.save_path)
        os.makedirs(absolute_save_path, exist_ok=True)

        # 修改这里：保存整个模型而不只是参数
        model_path = os.path.join(absolute_save_path, f"{model_name}.pt")

        # 如果有最佳参数，则先加载到模型中
        if best_param is not None:
            self.model.load_state_dict(best_param)

        # 保存整个模型
        torch.save(self.model, model_path)
        print(f"完整模型已保存至: {model_path}")

        # 同时也保存一份参数，以保持向后兼容性
        params_path = os.path.join(absolute_save_path, f"{model_name}_params.pkl")
        torch.save(best_param, params_path)
        print(f"模型参数也单独保存至: {params_path}")

    def predict(self, dl_test, batch_size = 32):
        if not self.fitted:
            raise ValueError("models is not fitted yet!")

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False, batch_size=batch_size)

        preds = []
        weighted_r2s = []

        self.model.eval()
        for batch_data in test_loader:
            batch_size = batch_data.shape[0]
            batch_preds = []
            batch_labels = []
            batch_weights = []

            for i in range(batch_size):
                data = batch_data[i]
                feature = data[:, :, 1:-1].to(self.device)
                label = data[:, -1, -1]
                weight = data[:, -1, 0]

                with torch.no_grad():
                    pred = self.model(feature.float()).detach().cpu().numpy()

                batch_preds.append(pred.ravel())
                batch_labels.append(label.detach().numpy())
                batch_weights.append(weight.detach().numpy())

            # 合并这个批次的所有预测
            preds.extend(np.concatenate(batch_preds))

            # 计算这个批次的weighted_r2
            all_preds = np.concatenate(batch_preds)
            all_labels = np.concatenate(batch_labels)
            all_weights = np.concatenate(batch_weights)
            weighted_r2 = calc_weighted_r2(all_preds, all_labels, all_weights)
            weighted_r2s.append(weighted_r2)

        predictions = pd.Series(np.array(preds), index=dl_test.get_index())

        return predictions, None, np.mean(weighted_r2s)

## Utils for MASTER models


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model) # Create a zero tensor pe of shape (max_len, d_model) to store positional encodings.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Create a float tensor position from 0 to max_len-1 and expand it to shape (max_len, 1).
                                                                          # This is done for broadcasting operations later.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # Assign computed sine values to even columns of pe
        pe[:, 1::2] = torch.cos(position * div_term) # Assign computed cosine values to odd columns of pe
        self.register_buffer("pe", pe) # Register pe as a persistent buffer, making it part of the module but not models parameters.

    def forward(self, x):
        return x + self.pe[:x.shape[1], :] # In the forward pass, add the input data x with the corresponding positional encoding self.pe[:x.shape[1], :].
                                          # Here, self.pe[:x.shape[1], :] extracts the positional encodings matching the time steps of the input data.

class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead) # Used to scale the attention score matrix

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(nn.Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # Input LayerNorm
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1) # Transpose dimensions 0 and 1, resulting in shape (8,472,256)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead) # 256/2
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1) # Shape (8, 472, 472).
                                                                                                           # Represents attention weights for each stock j from all stocks i at every time step t.
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1)) # Different from TA here
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output # Shape (472,8,256)

class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False) # Input and output dimensions are both d_model
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0: # If dropout parameter is greater than 0, create a Dropout layer for each attention head and store them in self.attn_dropout list.
            for i in range(nhead):
                self.attn_dropout.append(nn.Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # Input LayerNorm
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5) # Layer normalization, normalizing all feature dimensions for each sample
        # FFN layerNorm
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x) # Input is normalized, resulting in shape (482,8,256)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead) # Dimension per attention head
        att_output = []
        for i in range(self.nhead): # Ensures that each attention head processes different feature subspaces.
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim] # Extract partial data from dimension d_model along the third dimension, resulting in shape (482,8,64)
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1) # Calculate attention weight matrix, shape (482,8,8).
            # Each element atten_weight[n, t_i, t_j] represents the attention weight from time step t_i to time step t_j in the nth sample.
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh)) # Each element represents the weighted sum result at time step t_i in the nth sample, shape (482,8,64)
        att_output = torch.concat(att_output, dim=-1) # Concatenate outputs from all attention heads along the last dimension, forming the final attention output tensor, shape (482,8,256)

        # FFN
        xt = x + att_output # Residual connection adding original input x and attention output att_output, resulting in shape (482,8,256)
        xt = self.norm2(xt) # Layer normalization on xt for each feature dimension, making mean close to 0 and variance close to 1, resulting in shape (482,8,256)
        att_output = xt + self.ffn(xt) # Non-linear transformation through feed-forward network ffn and another residual connection, resulting in shape (482,8,256)

        return att_output # Final output, shape (482,8,256)

class TemporalAttention(nn.Module):
    '''
    Its main purpose is to extract the most important information from features across multiple time steps and generate a comprehensive temporal embedding.
    '''
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z): # z: [N, T, D], e.g.(475,8,256)
        h = self.trans(z) # [N, T, D], e.g.(475,8,256), Transform the input temporal embeddings z into new representations h
        query = h[:, -1, :].unsqueeze(-1) # [N, D]-> [N, D, 1] e.g.(475,256,1), Extract the feature vector of the last time step of each sample,
                                         # representing the latest state or trend used to measure the importance of other time step embeddings.
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, 1] --> [N, T] e.g.(475,8), Attention score matrix where each row corresponds to the attention scores of all time steps in one sample,
                                                 # representing the importance scores of each time step.
        lam = torch.softmax(lam, dim=1).unsqueeze(1) # Apply softmax on scores over time steps to get weights [N, T] --> [N, 1, T], e.g.(475,1,8)
        output = torch.matmul(lam, z).squeeze(1)  # Generate a comprehensive temporal embedding by weighted sum of all time step features according to the probability distribution.
                                                # Get final output [N, 1, T], [N, T, D] --> [N, D], e.g.(475,256)
        return output # Comprehensive temporal embedding, e.g.(475,256)



class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__() # This line calls the constructor of the superclass (nn.Module) to ensure that the module is properly initialized.
        self.trans = nn.Linear(d_input, d_output) # This line creates a linear transformation layer that maps the input of dimension d_input to an output of dimension d_output

        # These lines store the output dimension and the scaling factor beta as instance variables.
        self.d_output =d_output
        self.t = beta

    def forward(self, gate_input):
        # This line applies the linear transformation (defined in the __init__ method) to the input tensor.
        output = self.trans(gate_input)
        # This line applies the softmax function to the output, scaling it by beta (stored as self.t). The dim=-1 parameter indicates that softmax is applied along the last dimension.
        output = torch.softmax(output/self.t, dim=-1)
        # This line multiplies the softmax output by the output dimension d_output and returns the result.
        return self.d_output*output


class MASTER(nn.Module):
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=None):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        output = self.layers(src).squeeze(-1)

        return output
class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat: int = 20, d_model: int = 64, t_nhead: int = 4, s_nhead: int = 2, gate_input_start_index=None, gate_input_end_index=None,
            T_dropout_rate=0.5, S_dropout_rate=0.5, beta=5.0, **kwargs,
    ):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        super(MASTERModel, self).init_model()




def pad_data(data, symbol_range=range(40)):
    # Get all unique date_ids and time_ids
    date_ids = data['date_id'].unique()
    time_ids = data['time_id'].unique()

    # Create a MultiIndex with all combinations of date_id, time_id, and symbol_id
    full_index = pd.MultiIndex.from_product([date_ids, time_ids, symbol_range],
                                            names=['date_id', 'time_id', 'symbol_id'])

    # Set the index to ['date_id', 'time_id', 'symbol_id']
    data = data.set_index(['date_id', 'time_id', 'symbol_id'])

    # Reindex the data to include all combinations, filling missing values with 0
    padded_data = data.reindex(full_index, fill_value=0)

    return padded_data


def main(args):
    # Load and preprocess data
    day = 1600
    print(f'Use data from date:{day}')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, f'../data/output/Fature_factor_data_{day}_master_n40.parquet')
    data = pl.read_parquet(data_path).to_pandas()
    #data = pl.read_parquet(f'../data/output/Fature_factor_data_{day}_master.parquet').to_pandas()
    data = data.drop([f'responder_{i}' for i in [0, 1, 2, 3, 4, 5, 7, 8]], axis=1)

    date_select_begin = 1668
    data = data[data['date_id'] >= date_select_begin]  # 只保留date_id大于等于date_select_begin的数据
    # 做check,打印出date_id的最大值和最小值
    print(f"目前date_id的最小值: {data['date_id'].min()}")

    # def pad_data(data, symbol_range=range(40)):
    #     date_ids = data['date_id'].unique()
    #     time_ids = data['time_id'].unique()
    #     full_index = pd.MultiIndex.from_product([date_ids, time_ids, symbol_range],
    #                                             names=['date_id', 'time_id', 'symbol_id'])
    #     data = data.set_index(['date_id', 'time_id', 'symbol_id'])
    #     padded_data = data.reindex(full_index, fill_value=0)
    #     return padded_data
    #
    # data = pad_data(data).reset_index()
    data['time_id'] = ((data['date_id'].astype(np.float64)) * 1000 + (data['time_id'].astype(np.float64))).astype(
        np.float32)
    data = data.drop(['date_id'], axis=1).astype(np.float32)
    data = data.set_index(['time_id', 'symbol_id'])

    max_time_id = data.index.get_level_values('time_id').max()
    min_time_id = data.index.get_level_values('time_id').min()
    split_time_id = type(max_time_id)(min_time_id + int((max_time_id - min_time_id) * 0.8))
    split_time_id2 = type(max_time_id)(min_time_id + int((max_time_id - min_time_id) * 0.9))

    train_data = data.query('time_id <= @split_time_id').copy()
    valid_data = data.query('time_id > @split_time_id & time_id <= @split_time_id2').copy()
    test_data = data.query('time_id > @split_time_id2').copy()

    train_dataset = TSDataSampler(data=train_data, start=min_time_id, end=split_time_id, step_len=10,
                                  fillna_type='ffill+bfill')
    valid_dataset = TSDataSampler(data=valid_data, start=split_time_id + 1, end=split_time_id2, step_len=10,
                                  fillna_type='ffill+bfill')
    test_dataset = TSDataSampler(data=test_data, start=split_time_id2 + 1, end=max_time_id, step_len=10,
                                 fillna_type='ffill+bfill')

    # Initialize model with args
    model = MASTERModel(
        d_feat=args.d_feat, d_model=args.d_model, t_nhead=args.t_nhead, s_nhead=args.s_nhead,
        T_dropout_rate=args.dropout, S_dropout_rate=args.dropout, beta=args.beta,
        gate_input_end_index=args.gate_input_end_index, gate_input_start_index=args.gate_input_start_index,
        n_epochs=args.n_epoch, lr=args.lr, GPU=args.gpu, seed=args.seed,
        save_path=args.save_path, save_prefix=args.save_prefix
    )

    # Train and validate
    model.fit(train_dataset, valid_dataset, batch_size=args.batch_size)
    print(f"Model Trained with beta={args.beta}, d_model={args.d_model}, dropout={args.dropout}")

    # Test
    predictions, labels, weighted_R2 = model.predict(test_dataset, batch_size=args.batch_size)
    print(f"Test weighted R2: {weighted_R2}")
    return weighted_R2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # 核心参数
    parser.add_argument('--d_feat', type=int, default=79,
                        help='输入特征维度，需与数据实际特征数一致')
    # 原参数定义替换为：
    parser.add_argument('--d_model', type=int, nargs='+', default=[256],
                        help='模型隐层维度，候选值列表')
    parser.add_argument('--t_nhead', type=int, nargs='+', default=[4],
                        help='时序注意力头数，候选值列表')
    parser.add_argument('--s_nhead', type=int, nargs='+', default=[2],
                        help='空间注意力头数，候选值列表')
    parser.add_argument('--dropout', type=float, nargs='+', default=[0.2],
                        help='Dropout概率，候选值列表')
    parser.add_argument('--batch_size', type=int, nargs='+', default=[16],
                        help='批量大小，候选值列表')

    # 门控参数
    parser.add_argument('--beta', type=float, nargs='+', default=[5],
                        help='门控温度参数，建议测试离散值如1,2,5,10')
    parser.add_argument('--gate_input_start_index', type=int, default=79,
                        help='门控输入起始索引，需严格匹配特征列')
    parser.add_argument('--gate_input_end_index', type=int, default=89, #40个factor
                        help='门控输入结束索引，需严格匹配特征列')

    # 训练参数
    parser.add_argument('--n_epoch', type=int,
                        default=1,
                        help='训练轮次，建议10-20')
    parser.add_argument('--lr', type=float, default=[8e-6],
                        #choices=[5e-6, 8e-6, 2e-5],
                        help='学习率，建议1e-6到1e-4间搜索')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_path = os.path.join(base_dir, 'models/MASTER_save_model')
    # 路径参数
    parser.add_argument('--save_path', type=str, default=default_save_path,
                        help='模型保存路径')
    # parser.add_argument('--save_prefix', type=str, default='/modelsfull0424best',
    #                     help='模型文件名前缀')
    # 修改参数默认值，去掉前导斜杠
    parser.add_argument('--save_prefix', type=str, default='master0428',
                        help='模型文件名前缀')

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # 确保保存路径有效
    if not os.path.isabs(args.save_path):
        args.save_path = os.path.abspath(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    print(f"模型将保存到: {args.save_path}")

    # 定义参数空间
    param_grid = {
        'beta': args.beta,
        'd_model': args.d_model,
        't_nhead': args.t_nhead,
        's_nhead': args.s_nhead,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'lr': args.lr
    }

    # 生成所有组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    best_r2 = -float('inf')
    best_params = {}

    for params in param_combinations:
        current_args = argparse.Namespace(**vars(args))
        for key, value in params.items():
            setattr(current_args, key, value)
        print(f"\nTesting params: {params}")

        try:
            r2 = main(current_args)
        except Exception as e:
            print(f"参数组合 {params} 运行失败，错误: {e}")
            continue

        if r2 > best_r2:
            best_r2 = r2
            best_params = params.copy()
            best_params['r2'] = r2
            print(f"当前最佳参数: {best_params}")

    print("\nBest parameters found:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    # 保存最佳参数到pickle文件
    with open(os.path.join(args.save_path, f"{args.save_prefix}_best_params.pkl"), 'wb') as f:
        pickle.dump(best_params, f)
    print(f"最佳参数已保存到: {os.path.join(args.save_path, f'{args.save_prefix}_best_params.pkl')}")

    # 保存为可读的JSON格式
    with open(os.path.join(args.save_path, f"{args.save_prefix}_best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"最佳参数已保存到: {os.path.join(args.save_path, f'{args.save_prefix}_best_params.json')}")