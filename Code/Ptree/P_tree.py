import numpy as np
import pandas as pd
import logging
import time

# 配置日志记录系统的基础设置，设置日志级别为 INFO，格式为时间戳、日志级别和消息内容
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PTree:
    def __init__(self, data, J, n_min, gamma, m, index, if_ratio=False, n_ratio=0.01, n_ratio_min=10):
        """
        初始化 PTree 类。

        :param data (DataFrame): T x n x (K + index) 的非平衡面板数据。前index列包含股票代码 ('gvkey')、月度超额对数收益率 ('xret')、
        年份-月份('date')和t-1期的原始市值 ('lag_me')。从第index+1列开始是标准化的特征数据。
        :param J (int): 总的分裂次数，形成J+1个叶节点与特征资产。
        :param n_min (int): 叶节点中资产的最小数量（当if_ratio==False时）。
        :param gamma (float): 收缩参数，用于计算最优均值-方差权重。
        :param m (int): 划分间隔。
        :param index (int): 第一个特征的序列数
        :param if_ratio (bool): 是否按照横截面个股数量设置叶节点内资产的最小数量，默认为False不设置。
        :param n_ratio (float): 叶节点内资产的最小数量与横截面个股数量之比，默认为1%（当if_ratio==True时）。
        :param n_ratio_min (int): 当以比例设置资产的最小数量时，叶节点内资产的最小数量的下限，默认为10（当if_ratio==True时）
        """
        # 输入参数
        self.data = data
        self.J = J
        self.n_min = n_min
        self.gamma = gamma
        self.m = m
        self.index = index
        self.if_ratio = if_ratio
        self.n_ratio = n_ratio
        # 储存型参数
        self.tree = []  # 存储节点和分裂信息
        self.max_sharpe_ratio = -np.inf  # 用于跟踪迄今为止的最大夏普比率，以确保节点只在新的夏普比率大于当前最大值时才进行分裂
        self.node_counter = 1  # 节点计数器，用于给每个节点命名
        self.final_weights = None  # 用于储存叶节点的权重
        self.T = len(data['date'].unique())  # 记录总的时间长度，用于节点内最小资产数的判断
        if self.if_ratio:
            self.n_t_min = data['date'].value_counts().sort_index() * self.n_ratio  # 记录横截面叶节点内的最小资产数
            self.n_t_min[self.n_t_min < n_ratio_min] = n_ratio_min  # 调整最小资产数

    def grow_tree(self, is_long=False):
        """
        核心函数
        从根节点开始生长 P-Tree-model树。
        ***logging报告频率：每次分割仅报告一次最优分割点，共J次。

        Parameters
            1. is_long (bool): 是否是纯多头投资

        Returns: (leaf_nodes, self.final_weights, self.split_info, self.f, self.max_sharpe_ratio)
            1. leaf_nodes (list): 叶节点的列表，列表里每个元素都是单独叶节点(dict: asset, depth, name)
            2. self.final_weights (ndarray): 叶节点权重，一元数组，数组元素个数为叶节点的个数（J+1）
            3. self.split_info (DataFrame): 分割点信息，列名为：['分割序数', '节点', '特征', '特征名', '分割点', 'SR']，大小为J×6
            4. self.f (ndarray): 切点组合（隐含因子）超额收益率的时间序列，一元数组，长度为时间跨度（T）
            5. self.max_sharpe_ratio (float): 树最终形成的最大夏普比率
        """
        start_time = time.time()  # 记录开始时间

        # 1.1 初始化树信息（序列、面板信息、根节点）
        root = {'assets': self.data, 'depth': 0, 'name': f'N{self.node_counter}'}
        self.node_counter += 1
        self.tree.append(root)

        # 1.2 储存变量（因子，最大夏普比率）
        self.split_info = pd.DataFrame(columns=['分割序数', '节点', '特征', '特征名', '分割点', 'SR'])
        self.f = None

        # 1.3 遍历分裂次数
        for j in range(self.J):
            logging.info(f'开始第 {j + 1}/{self.J} 次分裂')
            best_split = None
            best_split_info = None
            best_criterion = self.max_sharpe_ratio
            best_node = None

            # 2.1 遍历当前所有的叶节点
            for node in (leaf_node := [n for n in self.tree if 'split' not in n]):
                if node['depth'] >= self.J:
                    continue

                # 3.1 预先计算除当前节点之外的所有叶节点的特征资产（特征资产集的部分）
                other_leaf_nodes = [n for n in self.tree if 'split' not in n and id(n) != id(node)]
                leaf_weighted_returns = []
                for leaf in other_leaf_nodes:
                    # 对于叶节点的特征资产加权收益率计算，并按照时间排序
                    grouped = leaf['assets'].groupby('date')
                    weighted_return = grouped.agg(
                        weighted_return=('xret', lambda x: (x * leaf['assets'].loc[x.index, 'lag_me']).sum() / leaf['assets'].loc[x.index, 'lag_me'].sum())
                    ).reset_index(drop=True)['weighted_return'].sort_index().values
                    leaf_weighted_returns.append(weighted_return)
                # 将列表list转为narray结构，便于后续纵向叠加np.vstack
                leaf_weighted_returns = np.array(leaf_weighted_returns)

                # 3.2 遍历当前节点下所有的分割候选点
                for k in range(self.index, self.data.shape[1]):
                    for cut in np.linspace(0, 1, self.m + 1)[1: -1]:
                        left_data = node['assets'][node['assets'].iloc[:, k] <= cut]
                        right_data = node['assets'][node['assets'].iloc[:, k] > cut]

                        # 4.1.1 判断左右两个叶节点组合的时间长度是否为self.T
                        T_left = len(left_data['date'].unique())
                        T_right = len(right_data['date'].unique())
                        if T_left != self.T or T_right != self.T:
                            continue

                        # 4.1.2 判断是否小于资产数目最小限制
                        if self.if_ratio:
                            n_left_data = left_data['date'].value_counts().sort_index()
                            n_right_data = right_data['date'].value_counts().sort_index()
                            if (n_left_data <= self.n_t_min).any() or (n_right_data <= self.n_t_min).any():
                                continue
                        else:
                            min_assets_left = left_data.groupby('date').size().min() if not left_data.empty else 0
                            min_assets_right = right_data.groupby('date').size().min() if not right_data.empty else 0
                            if min_assets_left < self.n_min or min_assets_right < self.n_min:
                                continue

                        # 4.2.1 计算新的叶节点组合并估算隐含因子
                        R_left = left_data.groupby('date').agg(
                            weighted_return=('xret', lambda x: (x * left_data.loc[x.index, 'lag_me']).sum() / left_data.loc[x.index, 'lag_me'].sum())
                        ).reset_index(drop=True)['weighted_return'].values

                        R_right = right_data.groupby('date').agg(
                            weighted_return=('xret', lambda x: (x * right_data.loc[x.index, 'lag_me']).sum() / right_data.loc[x.index, 'lag_me'].sum())
                        ).reset_index(drop=True)['weighted_return'].values

                        # 4.2.2 形成特征资产集
                        if len(R_left) != len(R_right) or len(R_left) < self.T or len(R_right) < self.T:
                            continue
                        elif leaf_weighted_returns.size > 0:
                            R_new = np.vstack([leaf_weighted_returns, R_left, R_right])
                        else:
                            R_new = np.vstack([R_left, R_right])

                        # 4.3 计算切点组合（f）和夏普比率
                        f, weights = self.calculate_factor(R_new, is_long)
                        sharpe_ratio = self.calculate_sharpe_ratio(f)

                        # 4.4 判断是否更新最优分割点
                        if sharpe_ratio > best_criterion:
                            # 记录分割序数，节点，特征序数，特征名，分割点，SR
                            best_split_info = [j+1, node["name"], k - (self.index - 1), node['assets'].columns[k], cut,
                                               sharpe_ratio * np.sqrt(12)]
                            best_split = {'feature': k, 'cut': cut}
                            best_criterion = sharpe_ratio
                            best_node = node  # 更新最佳节点
                            self.f = f
                            self.final_weights = weights  # 更新最后一次分割点的权重

            # 2.2 判断所有节点提取中的最优候选点，并实现分裂
            if best_split is not None and best_node is not None:
                self.tree, self.node_counter = self.split_node(self.tree, best_node, best_split, self.node_counter)
                self.max_sharpe_ratio = best_criterion
                self.split_info.loc[len(self.split_info)] = best_split_info
                logging.info(f'找到更好的分裂点: 节点名称={best_split_info[1]}，特征={best_split_info[2]}，特征名={best_split_info[3]}，'
                             f'切点={best_split_info[4]:.1f}，夏普比率={best_split_info[5]:.4f}')
            else:
                logging.info(f'第 {j + 1}/{self.J} 次分裂未找到合适的分裂点')

        # 1.4 树生长完成，获取最后的叶节点
        logging.info(f'树生长完成，耗时 {time.time() - start_time:.2f} 秒')
        leaf_nodes = [n for n in self.tree if 'split' not in n]

        return leaf_nodes, self.final_weights, self.split_info, self.f, self.max_sharpe_ratio

    def grow_tree_detail(self, is_long=False, is_equal=False):
        """
        逻辑和代码与grow_tree一致，只是报告信息更为细致。
        ***logging报告频率：每次出现更优分割点都会报告。

        Parameters
            1. is_long (bool): 是否是纯多头投资
            2. is_equal (bool): 是否是等权重投资

        Returns: (leaf_nodes, self.final_weights, self.split_info, self.f, self.max_sharpe_ratio)
            1. leaf_nodes (list): 叶节点的列表，列表里每个元素都是单独叶节点(dict: asset, depth, name)
            2. self.final_weights (ndarray): 叶节点权重，一元数组，数组元素个数为叶节点的个数（J+1）
            3. self.split_info (DataFrame): 分割点信息，列名为：['分割序数', '节点', '特征', '特征名', '分割点', 'SR']，大小为J×6
            4. self.f (ndarray): 切点组合（隐含因子）超额收益率的时间序列，一元数组，长度为时间跨度（T）
            5. self.max_sharpe_ratio (float): 树最终形成的最大夏普比率
        """
        start_time = time.time()  # 记录开始时间

        # 1.1 初始化树信息（序列、面板信息、根节点）
        root = {'assets': self.data, 'depth': 0, 'name': f'N{self.node_counter}'}
        self.node_counter += 1
        self.tree.append(root)

        # 1.2 储存变量（因子，最大夏普比率）
        self.split_info = pd.DataFrame(columns=['分割序数', '节点', '特征', '特征名', '分割点', 'SR'])
        self.f = None

        # 1.3 遍历分裂次数
        for j in range(self.J):
            logging.info(f'开始第 {j + 1}/{self.J} 次分裂')
            best_split = None
            best_split_info = None
            best_criterion = self.max_sharpe_ratio
            best_node = None
            leaf_node_num = 0

            # 2.1 遍历当前所有的叶节点
            for node in (leaf_node := [n for n in self.tree if 'split' not in n]):
                leaf_node_num += 1
                bool_split = False
                if node['depth'] >= self.J:
                    continue

                # 3.1 预先计算除当前节点之外的所有叶节点的特征资产（特征资产集的部分）
                other_leaf_nodes = [n for n in self.tree if 'split' not in n and id(n) != id(node)]
                leaf_weighted_returns = []
                for leaf in other_leaf_nodes:
                    # 对于叶节点的特征资产加权收益率计算，并按照时间排序
                    grouped = leaf['assets'].groupby('date')
                    if not is_equal:
                        weighted_return = grouped.agg(
                            weighted_return=('xret', lambda x: (x * leaf['assets'].loc[x.index, 'lag_me']).sum() /
                                                              leaf['assets'].loc[x.index, 'lag_me'].sum())
                        ).reset_index(drop=True)['weighted_return'].sort_index().values
                        leaf_weighted_returns.append(weighted_return)
                    else:
                        weighted_return = grouped.agg(
                            equal_weighted_return=('xret', lambda x: x.mean())
                        ).reset_index(drop=True)['equal_weighted_return'].sort_index().values
                        leaf_weighted_returns.append(weighted_return)
                # 将列表list转为narray结构，便于后续纵向叠加np.vstack
                leaf_weighted_returns = np.array(leaf_weighted_returns)

                # 3.2 遍历当前节点下所有的分割候选点
                for k in range(self.index, self.data.shape[1]):
                    for cut in np.linspace(0, 1, self.m + 1)[1: -1]:
                        left_data = node['assets'][node['assets'].iloc[:, k] <= cut]
                        right_data = node['assets'][node['assets'].iloc[:, k] > cut]

                        # 4.1.1 判断左右两个叶节点组合的时间长度是否为self.T
                        T_left = len(left_data['date'].unique())
                        T_right = len(right_data['date'].unique())
                        if T_left != self.T or T_right != self.T:
                            continue

                        # 4.1.2 判断是否小于资产数目最小限制
                        if self.if_ratio:
                            n_left_data = left_data['date'].value_counts().sort_index()
                            n_right_data = right_data['date'].value_counts().sort_index()
                            if (n_left_data <= self.n_t_min).any() or (n_right_data <= self.n_t_min).any():
                                continue
                        else:
                            min_assets_left = left_data.groupby('date').size().min() if not left_data.empty else 0
                            min_assets_right = right_data.groupby('date').size().min() if not right_data.empty else 0
                            if min_assets_left < self.n_min or min_assets_right < self.n_min:
                                continue

                        # 4.2.1 计算新的叶节点组合并估算隐含因子
                        if not is_equal:
                            R_left = left_data.groupby('date').agg(
                                weighted_return=('xret',
                                                 lambda x: (x * left_data.loc[x.index, 'lag_me']).sum() / left_data.loc[
                                                     x.index, 'lag_me'].sum())
                            ).reset_index(drop=True)['weighted_return'].values

                            R_right = right_data.groupby('date').agg(
                                weighted_return=('xret',
                                                 lambda x: (x * right_data.loc[x.index, 'lag_me']).sum() / right_data.loc[
                                                     x.index, 'lag_me'].sum())
                            ).reset_index(drop=True)['weighted_return'].values
                        else:
                            R_left = left_data.groupby('date').agg(
                                weighted_return=('xret', lambda x: x.mean())
                            ).reset_index(drop=True)['weighted_return'].sort_index().values

                            R_right = right_data.groupby('date').agg(
                                weighted_return=('xret', lambda x: x.mean())
                            ).reset_index(drop=True)['weighted_return'].sort_index().values

                        # 4.2.2 形成特征资产集
                        if len(R_left) != len(R_right) or len(R_left) < self.T or len(R_right) < self.T:
                            continue
                        elif leaf_weighted_returns.size > 0:
                            R_new = np.vstack([leaf_weighted_returns, R_left, R_right])
                        else:
                            R_new = np.vstack([R_left, R_right])

                        # 4.3 计算切点组合（f）和夏普比率
                        f, weights = self.calculate_factor(R_new, is_long)
                        sharpe_ratio = self.calculate_sharpe_ratio(f)

                        # 4.4 判断是否更新最优分割点
                        if sharpe_ratio > best_criterion:
                            logging.info(
                                f'找到更好的分裂点: 夏普比率={sharpe_ratio * np.sqrt(12):.4f}，'
                                f"特征={k - (self.index - 1)} {node['assets'].columns[k]}，切点={cut:.1f}，节点名称={node['name']}，进程为第{leaf_node_num}/{len(leaf_node)}个节点")
                            # 记录分割序数，节点，特征序数，特征名，分割点，SR
                            best_split_info = [j + 1, node["name"], k - (self.index - 1), node['assets'].columns[k], cut,
                                               sharpe_ratio * np.sqrt(12)]
                            best_split = {'feature': k, 'cut': cut}
                            best_criterion = sharpe_ratio
                            best_node = node  # 更新最佳节点
                            self.f = f
                            self.final_weights = weights  # 更新最后一次分割点的权重
                            bool_split = True

                # 3.3 报告未分裂的节点
                if not bool_split:
                    logging.info(f'节点{node["name"]}(第{leaf_node_num}/{len(leaf_node)}个节点)未找到合适的分裂点')

            # 2.2 判断所有节点提取中的最优候选点，并实现分裂
            if best_split is not None and best_node is not None:
                self.tree, self.node_counter = self.split_node(self.tree, best_node, best_split, self.node_counter)
                self.max_sharpe_ratio = best_criterion
                self.split_info.loc[len(self.split_info)] = best_split_info
            else:
                logging.info(f'第 {j + 1}/{self.J} 次分裂未找到合适的分裂点')

        # 1.4 树生长完成，获取最后的叶节点
        logging.info(f'树生长完成，耗时 {time.time() - start_time:.2f} 秒')
        leaf_nodes = [n for n in self.tree if 'split' not in n]

        return leaf_nodes, self.final_weights, self.split_info, self.f, self.max_sharpe_ratio

    def split_node(self, tree, node, split, node_counter):
        """
        将一个节点分裂为两个子节点。

        Parameters:
            1. tree (ndarray): 要分裂的树。
            2. node (dict): 要分裂的节点。
            3. split (dict): 分裂信息。
            4. node_counter (int): 叶节点的序数。

        Return: (tree, node_counter)
            1. tree (ndarray): 分裂完的树。
            2. node_counter (int): 叶节点的序数
        """
        # 1. 提取分裂点信息和被分裂的叶节点的面板数据
        feature = split['feature']
        cut = split['cut']
        assets = node['assets']

        # 2. 分裂叶节点
        left_node = {'assets': assets[assets.iloc[:, feature] <= cut], 'depth': node['depth'] + 1,
                     'name': f'N{node_counter}'}
        node_counter += 1
        right_node = {'assets': assets[assets.iloc[:, feature] > cut], 'depth': node['depth'] + 1,
                      'name': f'N{node_counter}'}
        node_counter += 1

        # 3. 标注分裂点，增加新的叶节点
        node['split'] = split
        tree.append(left_node)
        tree.append(right_node)

        return tree, node_counter

    def calculate_factor(self, R, is_long=False):
        """
        计算叶节点组合（特征资产）中的隐含因子与权重
        公式参考：
        Cong L W, Feng G, He J, et al. Growing the efficient frontier on panel trees[J]. Journal of Financial Economics, 2025, 167: 104024.
        原文P5，公式(3)与公式(5).

        Parameter:
            1. R (ndarray): 当前所有叶节点的特征资产的超额收益率矩阵
            2. is_long (bool): 是否考虑纯多头投资，默认为False

        Returns: (f, weights)
            1. f (ndarray): 收益率隐含因子的时间序列
            2. weights (ndarray): 形成切点组合的叶节点权重
        """
        # 1. 计算期望与方差
        mean_R = R.mean(axis=1)
        E_RR = np.dot(R, R.T) / R.shape[1]
        inv_cov_R = np.linalg.inv(E_RR + self.gamma * np.eye(E_RR.shape[0]))

        # 2. 计算叶节点投资组合的权重weights
        weights = inv_cov_R @ mean_R
        if not is_long:
            weights /= np.sum(np.abs(weights))  # 等比例调整权重，使其和为1
        else:
            weights[weights < 0] = 0
            weights /= np.sum(np.abs(weights))

        # 3. 计算隐含因子
        f = weights @ R

        return f, weights

    def calculate_sharpe_ratio(self, f):
        """
        计算隐含因子的夏普比率

        Parameter:
            1. f (ndarray): 收益率隐含因子(切点组合)的时间序列

        Return:
            1. float: 夏普比率.
        """
        return f.mean() / f.std()