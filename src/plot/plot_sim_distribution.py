import numpy as np
from sklearn.preprocessing import MinMaxScaler

from model.vertical_fl.FedSimModel import SimDataset
from tqdm import tqdm

import pickle


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


if __name__ == '__main__':
    for k in [1, 5, 10, 20, 30, 40, 50]:
        print(f"{k=}")
        plot_sim_distribution("cache/beijing_sim.pkl", 1, knn_k=100, threshold=2*k)
        plot_sim_distribution("cache/game_sim.pkl", 1, knn_k=50, threshold=k)
        plot_sim_distribution("cache/hdb_sim.pkl", 1, knn_k=50, threshold=k)
        plot_sim_distribution("cache/song_sim.pkl", 1, knn_k=50, threshold=k)
        plot_sim_distribution("cache/ny_sim.pkl", 1, knn_k=50, threshold=k)