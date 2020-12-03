import os
import abc
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import deprecation

from .OnePartyModel import BaseModel


class TwoPartyBaseModel(abc.ABC, BaseModel):
    def __init__(self, num_common_features, drop_key=True, grid_min=-3., grid_max=3.01, grid_width=0.2, **kwargs):

        super().__init__(**kwargs)

        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_width = grid_width
        self.drop_key = drop_key
        self.num_common_features = num_common_features

        self.sim_scaler = None

    @abc.abstractmethod
    def match(self, data1, data2, labels, idx=None, preserve_key=False, sim_threshold=0.0,
              grid_min=-3., grid_max=3.01, grid_width=0.2) -> tuple:
        """
        Match the data of two parties, return the matched data
        :param idx: Index of data1, only for evaluation. It should not be involved in linkage.
        :param sim_threshold: threshold of similarity score, everything below the threshold will be removed
        :param data1: data in party 1
        :param data2: data in party 2
        :param labels: labels (in party 1)
        :param preserve_key: whether to preserve common features in the output
        :return: [matched_data1, matched_data2], matched_labels
                 Each line refers to one sample
        """
        raise NotImplementedError

    def prepare_train_combine(self, data1, data2, labels, data_cache_path=None):
        if data_cache_path and os.path.isfile(data_cache_path):
            print("Loading data from cache")
            with open(data_cache_path, 'rb') as f:
                train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx = pickle.load(f)
        else:
            print("Splitting data")
            train_data1, val_data1, test_data1, train_labels, val_labels, test_labels, train_idx1, val_idx1, test_idx1 = \
                self.split_data(data1, labels, val_rate=self.val_rate, test_rate=self.test_rate)
            print("Matching training set")
            self.sim_scaler = None  # scaler will fit train_Xs and transform val_Xs, test_Xs
            train_Xs, train_y, train_idx = self.match(train_data1, data2, train_labels, idx=train_idx1,
                                                      preserve_key=False, grid_min=self.grid_min, grid_max=self.grid_max,
                                                      grid_width=self.grid_width)
            print("Matching validation set")
            val_Xs, val_y, val_idx = self.match(val_data1, data2, val_labels, idx=val_idx1, preserve_key=False,
                                                grid_min=self.grid_min, grid_max=self.grid_max, grid_width=self.grid_width)
            print("Matching test set")
            test_Xs, test_y, test_idx = self.match(test_data1, data2, test_labels, idx=test_idx1, preserve_key=False,
                                                   grid_min=self.grid_min, grid_max=self.grid_max, grid_width=self.grid_width)

            train_X = np.concatenate(train_Xs, axis=1)
            val_X = np.concatenate(val_Xs, axis=1)
            test_X = np.concatenate(test_Xs, axis=1)

            print("Replace NaN with mean value")
            col_mean = np.nanmean(train_X, axis=0)
            train_indices = np.where(np.isnan(train_X))
            train_X[train_indices] = np.take(col_mean, train_indices[1])
            print("Train done.")
            val_indices = np.where(np.isnan(val_X))
            val_X[val_indices] = np.take(col_mean, val_indices[1])
            print("Validation done.")
            test_indices = np.where(np.isnan(test_X))
            test_X[test_indices] = np.take(col_mean, test_indices[1])
            print("Test done.")

            if data_cache_path:
                print("Saving data to cache")
                with open(data_cache_path, 'wb') as f:
                    pickle.dump([train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx], f)

        return train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx

    def train_combine(self, data1, data2, labels, data_cache_path=None):
        train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx = \
            self.prepare_train_combine(data1, data2, labels, data_cache_path)

        return self._train(train_X, val_X, test_X, train_y, val_y, test_y,
                           train_idx[:, 0], val_idx[:, 0], test_idx[:, 0])



