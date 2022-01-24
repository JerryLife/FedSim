from model.vertical_fl.FedSimModel import SimDataset

import pickle


def plot_sim_distribution(data_cache_path, sim_dim=1):
    print("Loading data from cache")
    with open(data_cache_path, 'rb') as f:
        train_dataset, val_dataset, test_dataset, y_scaler, sim_scaler = pickle.load(f)
    print("Done")
    sim_data = train_dataset.data[:, :sim_dim]
    pass


if __name__ == '__main__':
    plot_sim_distribution("cache/beijing_sim.pkl", 1)