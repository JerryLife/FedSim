import os
import abc
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree

from tqdm import tqdm
import deprecation

from .TwoPartyModel import TwoPartyBaseModel


class SimModel(TwoPartyBaseModel):
    def __init__(self, num_common_features, n_clusters=100, center_threshold=0.5,
                 blocking_method='grid', feature_wise_sim=False, **kwargs):
        super().__init__(num_common_features, **kwargs)

        self.feature_wise_sim = feature_wise_sim
        self.blocking_method = blocking_method
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
                    if self.feature_wise_sim:
                        score = -(key1[i] - key2[j]) ** 2
                    else:
                        score = -np.linalg.norm(key1[i] - key2[j])  # reverse distance
                    sim_scores.append(np.concatenate([np.array([i, j]), np.array(score)], axis=0))
        print("Done calculating similarity scores")

        # scale similarity scores to [0, 1]
        sim_scores = np.stack(sim_scores)
        if self.sim_scaler is not None:
            sim_scores[:, 2:] = self.sim_scaler.transform(sim_scores[:, 2:])
        else:
            self.sim_scaler = MinMaxScaler(feature_range=(0, 1))
            sim_scores[:, 2:] = self.sim_scaler.fit_transform(sim_scores[:, 2:])
        print("Done scaling")

        return np.array(sim_scores)

    def _array2base(self, array, base):
        res = 0
        for i, num in enumerate(array[::-1]):
            res += num * base ** i
        return res

    def cal_sim_score_grid(self, key1, key2, grid_min=-3., grid_max=3.01, grid_width=0.2):
        """
        :param key1: Common features in party 1
        :param key2: Common features in party 2
        :param grid_min: min value of grid
        :param grid_max: min value of grid
        :param grid_width: width of grid
        :return: sim_scores: Nx3 np.ndarray (idx_key1, idx_key2, sim_score)
        """
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
                    if self.feature_wise_sim:
                        score = -(key1[i] - key2[j]) ** 2
                    else:
                        score = -np.linalg.norm(key1[i] - key2[j]).reshape(-1)  # reverse distance
                    sim_scores.append(np.concatenate([np.array([i, j]), np.array(score)], axis=0))
        print("Done calculating similarity scores")

        # scale similarity scores to [0, 1]
        sim_scores = np.stack(sim_scores)
        if self.sim_scaler is not None:
            sim_scores[:, 2:] = self.sim_scaler.transform(sim_scores[:, 2:])
        else:
            self.sim_scaler = MinMaxScaler(feature_range=(0, 1))
            sim_scores[:, 2:] = self.sim_scaler.fit_transform(sim_scores[:, 2:])
        print("Done scaling")

        return sim_scores

    def cal_sim_score_knn(self, key1, key2, knn_k=3, kd_tree_leaf_size=40):
        """
        :param kd_tree_leaf_size: leaf size to build kd-tree
        :param knn_k: number of nearest neighbors to be calculated
        :param radius: only the distance with radius will be returned
        :param key1: Common features in party 1
        :param key2: Common features in party 2
        :return: sim_scores: Nx3 np.ndarray (idx_key1, idx_key2, sim_score)
        """
        print("Build KD-tree")
        tree = KDTree(key2, leaf_size=kd_tree_leaf_size)

        print("Query KD-tree")
        dists, idx2 = tree.query(key1, k=knn_k, return_distance=True)

        print("Calculate sim_scores")
        idx1 = np.repeat(np.arange(key1.shape[0]), knn_k)
        sim_scores = np.vstack([idx1, idx2.flatten(), -dists.flatten()]).T
        if self.sim_scaler is not None:
            # use train scaler
            sim_scores[:, -1] = self.sim_scaler.transform(sim_scores[:, -1].reshape(-1, 1)).flatten()
        else:
            # generate train scaler
            self.sim_scaler = MinMaxScaler(feature_range=(0, 1))
            sim_scores[:, -1] = self.sim_scaler.fit_transform(sim_scores[:, -1].reshape(-1, 1)).flatten()
        print("Done scaling")

        return sim_scores

    def cal_sim_score_radius(self, key1, key2, radius=3, kd_tree_leaf_size=40):
        """
        :param kd_tree_leaf_size: leaf size to build kd-tree
        :param radius: only the distance with radius will be returned
        :param key1: Common features in party 1
        :param key2: Common features in party 2
        :return: sim_scores: Nx3 np.ndarray (idx_key1, idx_key2, sim_score)
        """
        print("Build KD-tree")
        tree = KDTree(key2, leaf_size=kd_tree_leaf_size)

        print("Query KD-tree")
        idx2, dists = tree.query_radius(key1, r=radius, return_distance=True)

        print("Calculate sim_scores")
        repeat_times = [x.shape[0] for x in idx2]
        non_empty_idx = [x > 0 for x in repeat_times]
        idx1 = np.repeat(np.arange(key1.shape[0]), repeat_times)
        idx2 = np.concatenate(idx2[np.array(repeat_times) > 0])
        sims = -np.concatenate(dists[np.array(repeat_times) > 0])
        sim_scores = np.vstack([idx1, idx2, sims]).T
        if self.sim_scaler is not None:
            # use train scaler
            sim_scores[:, -1] = self.sim_scaler.transform(sim_scores[:, -1].reshape(-1, 1)).flatten()
        else:
            # generate train scaler
            self.sim_scaler = MinMaxScaler(feature_range=(0, 1))
            sim_scores[:, -1] = self.sim_scaler.fit_transform(sim_scores[:, -1].reshape(-1, 1)).flatten()
        print("Done scaling")

        return sim_scores

    def match(self, data1, data2, labels, idx=None, preserve_key=False, sim_threshold=0.0, grid_min=-3., grid_max=3.01,
              grid_width=0.2, knn_k=3, kd_tree_leaf_size=40, radius=0.1):
        """
        Match data1 and data2 according to common features
        :param radius:
        :param radius:
        :param knn_k:
        :param kd_tree_leaf_size:
        :param blocking_method: method of blocking before matching
        :param knn_k: number of nearest neighbors to be calculated
        :param kd_tree_leaf_size: leaf size to build kd-tree
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
        if self.blocking_method == 'grid':
            sim_scores = self.cal_sim_score_grid(key1, key2, grid_min=grid_min, grid_max=grid_max,
                                                 grid_width=grid_width)
        elif self.blocking_method == 'knn':
            sim_scores = self.cal_sim_score_knn(key1, key2, knn_k=knn_k, kd_tree_leaf_size=kd_tree_leaf_size)
        elif self.blocking_method == 'radius':
            sim_scores = self.cal_sim_score_radius(key1, key2, radius=radius, kd_tree_leaf_size=kd_tree_leaf_size)
        else:
            assert False, "Unsupported blocking method"

        if preserve_key:
            remain_data1 = data1
            remain_data2 = data2
        else:
            remain_data1 = data1[:, :-self.num_common_features]
            remain_data2 = data2[:, self.num_common_features:]

        # real_sim_scores = []
        # for idx1, idx2, score in sim_scores:
        #     real_sim_scores.append([idx[int(idx1)], int(idx2), score])
        # real_sim_scores = np.array(real_sim_scores)
        real_sim_scores = np.concatenate([idx[sim_scores[:, 0].astype(np.int)].reshape(-1, 1),
                                          sim_scores[:, 1:]], axis=1)

        # filter similarity scores (last column) by a threshold
        if not self.feature_wise_sim:
            real_sim_scores = real_sim_scores[real_sim_scores[:, -1] >= sim_threshold]
        elif not np.isclose(sim_threshold, 0.0):
            warnings.warn("Threshold is not used for feature-wise similarity")

        # save sim scores
        with open("cache/sim_scores.pkl", "wb") as f:
            pickle.dump(real_sim_scores, f)

        # convert to pandas
        data1_df = pd.DataFrame(remain_data1)
        data1_df['data1_idx'] = idx
        labels_df = pd.DataFrame(labels, columns=['y'])
        data2_df = pd.DataFrame(remain_data2)
        if self.feature_wise_sim:
            score_columns = ['score' + str(i) for i in range(self.num_common_features)]
        else:
            score_columns = ['score']
        sim_scores_df = pd.DataFrame(real_sim_scores, columns=['data1_idx', 'data2_idx'] + score_columns)
        sim_scores_df[['data1_idx', 'data2_idx']].astype('int32')
        data1_labels_df = pd.concat([data1_df, labels_df], axis=1)

        matched_pairs = np.unique(sim_scores_df['data1_idx'].to_numpy())
        print("Got {} samples in A".format(matched_pairs.shape[0]))

        print("Linking records")
        data1_labels_scores_df = sim_scores_df.merge(data1_labels_df,
                                                     how='right', on='data1_idx')
        merged_data_labels_df = data1_labels_scores_df.merge(data2_df,
                                                             how='left', left_on='data2_idx', right_index=True)
        merged_data_labels_df[score_columns] = merged_data_labels_df[score_columns].fillna(value=0.0)
        print("Finished Linking, got {} samples".format(len(merged_data_labels_df.index)))

        # extracting data to numpy arrays
        ordered_labels = merged_data_labels_df['y'].to_numpy()
        data1_indices = merged_data_labels_df['data1_idx'].to_numpy()
        data2_indices = merged_data_labels_df['data2_idx'].to_numpy()
        data_indices = np.vstack([data1_indices, data2_indices]).T
        merged_data_labels_df.drop(['y', 'data1_idx', 'data2_idx'], axis=1, inplace=True)
        merged_data_labels = merged_data_labels_df.to_numpy()
        # merged_data_labels: |sim_scores|data1|data2|
        sim_dim = self.num_common_features if self.feature_wise_sim else 1
        matched_data1 = merged_data_labels[:, :sim_dim + remain_data1.shape[1]]
        matched_data2 = np.concatenate([merged_data_labels[:, :sim_dim],      # sim scores
                                        merged_data_labels[:, sim_dim + remain_data1.shape[1]:]],
                                       axis=1)

        return [matched_data1, matched_data2], ordered_labels, data_indices
