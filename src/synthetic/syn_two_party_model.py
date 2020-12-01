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
from scipy.interpolate import griddata

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torchlars
from tqdm import tqdm
import deprecation

from .model import MLP
from .syn_one_party_model import BaseModel


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


class SimModel(TwoPartyBaseModel):
    def __init__(self, num_common_features, n_clusters=100, center_threshold=0.5, **kwargs):
        super().__init__(num_common_features, **kwargs)
        self.center_threshold = center_threshold
        self.n_clusters = n_clusters

    def merge_pred(self, pred_all: list):
        sort_pred_all = list(sorted(pred_all, key=lambda t: t[0], reverse=True))
        return sort_pred_all[0][0]

    @staticmethod
    def _collect(array):
        """
        Obtained from
        https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
        :param array:
        :return: <Values of unique elements>, <
        """
        idx_sort = np.argsort(array)
        sorted_array = array[idx_sort]

        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_array, return_counts=True, return_index=True)

        # splits the indices into separate arrays
        res = np.split(idx_sort, idx_start[1:])

        return vals, res

    @deprecation.deprecated()
    def __cal_sim_score_kmeans(self, key1, key2, seed=0):
        """
        Deprecated
        :return: numpy array with size (n, 3), table of similarity scores
                 ['index_data1': int, 'index_data2': int, 'sim_score': float]
        """

        # clustering
        print("Clustering")
        kmeans1 = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=seed).fit(key1)
        kmeans2 = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=seed).fit(key2)

        # filter close cluster centers
        print("Filter close cluster centers")
        close_clusters = []
        for i, center1 in enumerate(kmeans1.cluster_centers_):
            for j, center2 in enumerate(kmeans2.cluster_centers_):
                if np.linalg.norm(center1 - center2) < self.center_threshold:
                    close_clusters.append((i, j))
        print("Got {} close clusters after filtering".format(len(close_clusters)))

        labels1, indices1 = self._collect(kmeans1.labels_)
        labels2, indices2 = self._collect(kmeans2.labels_)

        # compare within the block
        print("Compare within each cluster")
        sim_scores = []
        for (label1, label2) in close_clusters:
            idx1 = indices1[np.argwhere(labels1 == label1).item()]
            idx2 = indices2[np.argwhere(labels2 == label2).item()]
            for i in idx1:
                for j in idx2:
                    score = -np.linalg.norm(key1[i] - key2[j])  # reverse distance
                    sim_scores.append([i, j, score])
        print("Done calculating similarity scores")

        # scale similarity scores to [0, 1]
        sim_scores = np.array(sim_scores)
        if self.sim_scaler is not None:
            sim_scores[:, -1] = self.sim_scaler.transform(sim_scores[:, -1].reshape(-1, 1)).flatten()
        else:
            self.sim_scaler = MinMaxScaler(feature_range=(0, 1))
            sim_scores[:, -1] = self.sim_scaler.fit_transform(sim_scores[:, -1].reshape(-1, 1)).flatten()
        print("Done scaling")

        return np.array(sim_scores)

    def _array2base(self, array, base):
        res = 0
        for i, num in enumerate(array[::-1]):
            res += num * base ** i
        return res

    def cal_sim_score_grid(self, key1, key2, grid_min=-3., grid_max=3.01, grid_width=0.2):
        print("Quantization")
        bins = np.arange(grid_min, grid_max, grid_width)
        quantized_key1 = np.digitize(key1, bins)
        quantized_key2 = np.digitize(key2, bins)
        quantized_key1 = np.array([self._array2base(k, bins.shape[0] + 1) for k in quantized_key1])
        quantized_key2 = np.array([self._array2base(k, bins.shape[0] + 1) for k in quantized_key2])

        print("Collect unique values")
        grid_ids1, indices1 = self._collect(quantized_key1)
        grid_ids2, indices2 = self._collect(quantized_key2)

        blocks = np.intersect1d(quantized_key1, quantized_key2)
        print("Finished quantization, got {} blocks".format(blocks.shape[0]))

        # compare within the block
        print("Compare within each block")
        sim_scores = []
        for quantized_id in tqdm(blocks):
            idx1 = indices1[np.argwhere(grid_ids1 == quantized_id).item()]
            idx2 = indices2[np.argwhere(grid_ids2 == quantized_id).item()]
            for i in idx1:
                for j in idx2:
                    score = -np.linalg.norm(key1[i] - key2[j])  # reverse distance
                    sim_scores.append([i, j, score])
        print("Done calculating similarity scores")

        # scale similarity scores to [0, 1]
        sim_scores = np.array(sim_scores)
        if self.sim_scaler is not None:
            sim_scores[:, -1] = self.sim_scaler.transform(sim_scores[:, -1].reshape(-1, 1)).flatten()
        else:
            self.sim_scaler = MinMaxScaler(feature_range=(0, 1))
            sim_scores[:, -1] = self.sim_scaler.fit_transform(sim_scores[:, -1].reshape(-1, 1)).flatten()
        print("Done scaling")

        return sim_scores

    def match(self, data1, data2, labels, idx=None, preserve_key=False, sim_threshold=0.0,
              grid_min=-3., grid_max=3.01, grid_width=0.2):
        """
        Match data1 and data2 according to common features
        :param idx: Index of training data1, not involved in linkage
        :param sim_threshold: sim_threshold: threshold of similarity score.
               Everything below the threshold will be removed
        :param data1: [<other features in party 1>, common_features]
        :param data2: [common_features, <other features in party 2>]
        :param labels: corresponding labels
        :param preserve_key:
        :return: [data1, data2], labels
                 data1 = [sim_score, <other features in party 1>]
                 data2 = [sim_score, <other features in party 2>]
        """
        # extract common features from data
        key1 = data1[:, -self.num_common_features:]
        key2 = data2[:, :self.num_common_features]

        # calculate similarity scores
        sim_scores = self.cal_sim_score_grid(key1, key2, grid_min=grid_min, grid_max=grid_max,
                                             grid_width=grid_width)

        if preserve_key:
            remain_data1 = data1
            remain_data2 = data2
        else:
            remain_data1 = data1[:, :-self.num_common_features]
            remain_data2 = data2[:, self.num_common_features:]

        real_sim_scores = []
        for idx1, idx2, score in sim_scores:
            real_sim_scores.append([idx[int(idx1)], int(idx2), score])
        real_sim_scores = np.array(real_sim_scores)

        # filter similarity scores (last column) by a threshold
        real_sim_scores = real_sim_scores[real_sim_scores[:, -1] >= sim_threshold]

        # save sim scores
        with open("cache/sim_scores.pkl", "wb") as f:
            pickle.dump(real_sim_scores, f)

        # convert to pandas
        data1_df = pd.DataFrame(remain_data1)
        data1_df['data1_idx'] = idx
        labels_df = pd.DataFrame(labels, columns=['y'])
        data2_df = pd.DataFrame(remain_data2)
        sim_scores_df = pd.DataFrame(real_sim_scores, columns=['data1_idx', 'data2_idx', 'score'])
        sim_scores_df[['data1_idx', 'data2_idx']].astype('int32')
        data1_labels_df = pd.concat([data1_df, labels_df], axis=1)

        matched_pairs = np.unique(sim_scores_df['data1_idx'].to_numpy())
        print("Got {} samples in A".format(matched_pairs.shape[0]))

        print("Linking records")
        data1_labels_scores_df = data1_labels_df.merge(sim_scores_df,
                                                       how='left', on='data1_idx')
        merged_data_labels_df = data1_labels_scores_df.merge(data2_df,
                                                             how='left', left_on='data2_idx', right_index=True)
        merged_data_labels_df['score'] = merged_data_labels_df['score'].fillna(value=0.0)
        print("Finished Linking, got {} samples".format(len(merged_data_labels_df.index)))

        # extracting data to numpy arrays
        ordered_labels = merged_data_labels_df['y'].to_numpy()
        data1_indices = merged_data_labels_df['data1_idx'].to_numpy()
        data2_indices = merged_data_labels_df['data2_idx'].to_numpy()
        data_indices = np.vstack([data1_indices, data2_indices]).T
        merged_data_labels_df.drop(['y', 'data1_idx', 'data2_idx'], axis=1, inplace=True)
        merged_data_labels = merged_data_labels_df.to_numpy()
        matched_data1 = np.concatenate([merged_data_labels[:, remain_data1.shape[1]].reshape(-1, 1),
                                        merged_data_labels[:, :remain_data1.shape[1]]], axis=1) # move score to column 0
        matched_data2 = merged_data_labels[:, remain_data1.shape[1]:]

        return [matched_data1, matched_data2], ordered_labels, data_indices


class ThresholdSimModel(SimModel):
    def __init__(self, num_common_features, sim_threshold=0.0, **kwargs):
        super().__init__(num_common_features, **kwargs)
        self.sim_threshold = sim_threshold

    def match(self, data1, data2, labels, idx=None, preserve_key=False, sim_threshold=0.0,
              grid_min=-3., grid_max=3.01, grid_width=0.2) -> tuple:
        [matched_data1, matched_data2], ordered_labels, data_indices = \
            super().match(data1, data2, labels, idx=idx, preserve_key=preserve_key, sim_threshold=self.sim_threshold)
        # remove similarity score from data
        return (matched_data1[:, 1:], matched_data2[:, 1:]), ordered_labels, data_indices


