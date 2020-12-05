import os
import sys
import abc
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torchlars
from tqdm import tqdm

from .SimModel import SimModel


# class Top1GroupDataset(Dataset):
#     def __init__(self, data, labels, data_idx):
#         self.raw_data_labels = np.concatenate([data, labels], axis=1)
#
#         self.filtered_data = {}
#         for i in data_idx.shape[0]:
#             prev_sim_score = self.filtered_data[data_idx[i]][0]
#             cur_sim_score = self.raw_data_labels[data_idx[i]][0]
#             if not (data_idx[i] in self.filtered_data and cur_sim_score < prev_sim_score):
#                 self.filtered_data[data_idx[i]] = self.raw_data_labels[data_idx[i]]
#
#         self.data_idx = np.array(list(self.filtered_data.keys()))
#         self.data_labels = np.array(list(self.filtered_data.values()))
#         self.data = self.data_labels[:, :-1]
#         self.labels = self.data_labels[:, -1]
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         return self.data[idx], self.labels[idx], self.data_idx[idx]


class Top1SimModel(SimModel):
    def __init__(self, num_common_features, **kwargs):
        super().__init__(num_common_features, **kwargs)

    @staticmethod
    def _group(data, labels, data_idx):
        assert data.shape[0] == labels.shape[0] == data_idx.shape[0]
        print("Start grouping, got {} samples".format(data_idx.shape[0]))
        raw_data_labels = np.concatenate([data, labels], axis=1)
        filtered_data = {}
        filtered_data_idx = {}
        for i in range(data_idx.shape[0]):
            cur_sim_score = raw_data_labels[i][0]
            idx1 = data_idx[i][0]
            if not (idx1 in filtered_data and cur_sim_score < filtered_data[idx1][0]):
                filtered_data[idx1] = raw_data_labels[i]
                filtered_data_idx[idx1] = data_idx[i][1]

        print("Finished grouping, got {} samples".format(len(filtered_data)))
        print("Exact matched rate {}".format(np.average([abs(k-v)<1e-7 for k, v in filtered_data_idx.items()])))
        print("Non-NaN rate {}".format(np.average(np.invert(np.isnan(list(filtered_data_idx.values()))))))
        grouped_data_idx = np.array(list(filtered_data.keys()))
        data_labels = np.array(list(filtered_data.values()))
        grouped_data = data_labels[:, :-1]
        grouped_labels = data_labels[:, -1]
        return grouped_data, grouped_labels, grouped_data_idx

    def train_combine(self, data1, data2, labels, data_cache_path=None):
        train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx = \
            self.prepare_train_combine(data1, data2, labels, data_cache_path)

        train_X, train_y, train_idx1 = self._group(train_X, train_y.reshape(-1, 1), train_idx)
        val_X, val_y, val_idx1 = self._group(val_X, val_y.reshape(-1, 1), val_idx)
        test_X, test_y, test_idx1 = self._group(test_X, test_y.reshape(-1, 1), test_idx)

        return self._train(train_X, val_X, test_X, train_y, val_y, test_y, train_idx1, val_idx1, test_idx1)
