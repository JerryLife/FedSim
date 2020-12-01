import pandas as pd
import pickle
import os
import sys
import numpy as np

from synthetic.syn_data_generator import TwoPartyClsOne2OneGenerator
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"

# syn_generator = TwoPartyClsOne2OneGenerator.from_pickle(root + "syn_cls_one2one_generator.pkl")
# [X1, X2], y = syn_generator.get_parties()
#
#
# def _collect(array):
#     """
#     Obtained from
#     https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
#     :param array:
#     :return:
#     """
#     idx_sort = np.argsort(array)
#     sorted_array = array[idx_sort]
#
#     # returns the unique values, the index of the first occurrence of a value, and the count for each element
#     vals, idx_start, count = np.unique(sorted_array, return_counts=True, return_index=True)
#
#     # splits the indices into separate arrays
#     res = np.split(idx_sort, idx_start[1:])
#
#     return vals, res
#
#
# sim_scaler = None
#
# def cal_sim_score(key1, key2, seed=0):
#     """
#     :return: numpy array with size (n, 3), table of similarity scores
#              ['index_data1': int, 'index_data2': int, 'sim_score': float]
#     """
#
#     # clustering
#     print("Clustering")
#     kmeans1 = MiniBatchKMeans(n_clusters=100, random_state=seed).fit(key1)
#     kmeans2 = MiniBatchKMeans(n_clusters=100, random_state=seed).fit(key2)
#
#     # filter close cluster centers
#     print("Filter close cluster centers")
#     close_clusters = []
#     for i, center1 in enumerate(kmeans1.cluster_centers_):
#         for j, center2 in enumerate(kmeans2.cluster_centers_):
#             if np.linalg.norm(center1 - center2) < 0.3:
#                 close_clusters.append((i, j))
#     print("Got {} close clusters after filtering".format(len(close_clusters)))
#
#     labels1, indices1 = _collect(kmeans1.labels_)
#     labels2, indices2 = _collect(kmeans2.labels_)
#
#     # compare within the block
#     print("Compare within each cluster")
#     sim_scores = []
#     for (label1, label2) in close_clusters:
#         idx1 = indices1[np.argwhere(labels1 == label1).item()]
#         idx2 = indices2[np.argwhere(labels2 == label2).item()]
#         for i in idx1:
#             for j in idx2:
#                 score = -np.linalg.norm(key1[i] - key2[j])  # reverse distance
#                 sim_scores.append([i, j, score])
#     print("Done calculating similarity scores")
#
#     # scale similarity scores to [0, 1]
#     sim_scores = np.array(sim_scores)
#     if sim_scaler is not None:
#         sim_scores[:, -1] = sim_scaler.transform(sim_scores[:, -1].reshape(-1, 1)).flatten()
#     else:
#         sim_scaler = MinMaxScaler(feature_range=(0, 1))
#         sim_scores[:, -1] = sim_scaler.fit_transform(sim_scores[:, -1].reshape(-1, 1)).flatten()
#     print("Done scaling similarity")
#
#     return np.array(sim_scores)
#
#
# # extract common features from data
# key1 = X1[:, -3:]
# key2 = X2[:, :3]
#
# # calculate similarity scores
# sim_scores = cal_sim_score(key1, key2)
#
# # filter similarity scores (last column) by a threshold
# sim_scores = sim_scores[sim_scores[:, -1] > 0]

sim_scores = pd.read_pickle("cache/sim_scores.pkl")

print("Summarize")
matched = {}
for idx1, idx2, score in sim_scores:
    idx1, idx2 = int(idx1), int(idx2)
    if idx1 in matched:
        matched[idx1].append((idx2, score))
    else:
        matched[idx1] = [(idx2, score)]

print("Sort")
ranks = []
for k, v in matched.items():
    new_v = list(sorted(v, key=lambda x: x[1], reverse=True))
    matched[k] = new_v
    try:
        rank = next(i for i, (idx2, score) in enumerate(new_v) if idx2 == k)
    except StopIteration:
        rank = -1
    ranks.append(rank)

print("Ranks:" + str(ranks))
print("Done")
