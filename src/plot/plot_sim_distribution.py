import matplotlib.pylab
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from model.vertical_fl.FedSimModel import SimDataset
from tqdm import tqdm

import pickle
import os
import sys


def plot_sim_distribution(data_cache_path, sim_dim=1, knn_k=100, threshold=20):
    print("Loading data from cache")
    with open(data_cache_path, 'rb') as f:
        train_dataset, val_dataset, test_dataset, y_scaler, sim_scaler = pickle.load(f)
    print("Done")
    sim_data = train_dataset.data[:, :sim_dim].reshape(-1, knn_k).detach().cpu().numpy()
    scaler = MinMaxScaler([0, 1])
    scaled_sim_data = scaler.fit_transform(sim_data.T).T
    sorted_sim_data = np.sort(scaled_sim_data, axis=1)[:, ::-1]

    split_indices = np.argmax(sorted_sim_data > threshold, axis=1)
    sim_threshold = sorted_sim_data[:, threshold].reshape(-1, 1)
    sorted_sim_data[sorted_sim_data > sim_threshold] = 1 - sorted_sim_data[sorted_sim_data > sim_threshold]
    improve = np.sum(sorted_sim_data, axis=1)
    position = np.std(split_indices / knn_k)
    avg_improve = np.average(improve)

    # marginal_indices = (sorted_sim_data > threshold) & (sorted_sim_data < 1 - threshold)
    # n_marginal = np.count_nonzero(marginal_indices, axis=1)
    # marginal_ratio = np.average(n_marginal) / knn_k
    # avg_splits = []
    # for i, idx_group in tqdm(enumerate(marginal_indices)):
    #     indices = idx_group.nonzero()[0]
    #     if len(indices) > 0:
    #         avg_split = (indices[0] + indices[-1]) / 2      # indices is sorted
    #     else:
    #         large_sim_indices = (sorted_sim_data[i] > 1 - threshold).nonzero()[0]
    #         if len(large_sim_indices) == 0:
    #             continue
    #         avg_split = large_sim_indices[0]
    #     avg_splits.append(avg_split)
    # split_variance = np.std(avg_splits)

    print(f"{data_cache_path=}, {position=}, {avg_improve=}")
    pass


def plot_metric_vs_accuracy(output_path):
    params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
    matplotlib.pylab.rcParams.update(params)

    # delta = [34.05, 14.26, 20.69, 4.14, 1.24]
    # score_top1 = np.asarray([58.54, 256.19, 31.56, 92.71, 8.18])
    # score_fedsim = np.asarray([42.12, 236.28, 27.17, 92.87, 8])
    delta = [34.05, 14.26, 20.69, 4.14, 10.50]
    score_top1 = np.asarray([58.54, 256.19, 31.56, 92.71, 42840.23])
    score_fedsim = np.asarray([42.12, 236.28, 27.17, 92.87, 37083.72])
    dataset_name = ['house', 'taxi', 'hdb', 'game', 'company']
    improve = np.abs(score_fedsim - score_top1) / score_top1 * 100
    plt.scatter(delta, improve)
    plt.annotate('house', (delta[0], improve[0]), fontsize=16,
                 xytext=(delta[0]-3.5, improve[0]-2))
    for i, name in enumerate(dataset_name):
        if i == 0:
            continue
        plt.annotate(name, (delta[i], improve[i]), fontsize=16,
                     xytext=(delta[i]+.5, improve[i]+.5))

    plt.xlabel(r"$\Delta$(Top1Sim)")
    plt.ylabel("Improvement on Top1Sim (%)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../")  # change working directory
    # for k in [1]:
    #     print(f"{k=}")
    #     plot_sim_distribution("cache/beijing_sim.pkl", 1, knn_k=100, threshold=2*k)
    #     plot_sim_distribution("cache/game_sim.pkl", 1, knn_k=50, threshold=k)
    #     plot_sim_distribution("cache/hdb_sim.pkl", 1, knn_k=50, threshold=k)
    #     plot_sim_distribution("cache/song_sim.pkl", 1, knn_k=50, threshold=k)
    #     plot_sim_distribution("cache/ny_sim.pkl", 1, knn_k=50, threshold=k)
    #     plot_sim_distribution("cache/syn_sim_noise_0.2.pkl", 1, knn_k=100, threshold=k)
    #     plot_sim_distribution("cache/boone_sim_noise_0.2.pkl", 1, knn_k=100, threshold=k)
    #     plot_sim_distribution("cache/frog_sim_noise_0.2.pkl", 1, knn_k=100, threshold=k)
    #     plot_sim_distribution("cache/company_subset_sim_p_base_0.1.pkl", 1, knn_k=50, threshold=k)

    plot_metric_vs_accuracy("fig/metric_vs_improve.png")