import os
import sys
from datetime import datetime

from model.vertical_fl.AvgSimModel import AvgSimModel
from preprocess.ml_dataset.two_party_loader import TwoPartyLoader

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
dataset = "MiniBooNE_PID.txt"
num_common_features = 4

data_loader = TwoPartyLoader.from_pickle(root + dataset + "_loader.pkl")
[X1, X2], y = data_loader.load_parties()
name = "boone_sim_merge_combine"
model = AvgSimModel(num_common_features=num_common_features,
                    sim_hidden_sizes=[10, 10],
                    task='binary_cls',
                    dataset_type='syn',
                    merge_mode='sim_model_avg',
                    blocking_method='kdtree',
                    n_classes=2,
                    grid_min=-3.0,
                    grid_max=3.0,
                    grid_width=0.15,
                    knn_k=3,
                    tree_leaf_size=1000,
                    model_name=name + "_" + now_string,
                    val_rate=0.1,
                    test_rate=0.2,
                    drop_key=True,
                    device='cuda:0',
                    hidden_sizes=[200, 200, 100],
                    train_batch_size=4096,
                    test_batch_size=4096,
                    num_epochs=100,
                    learning_rate=1e-2,
                    weight_decay=1e-5,
                    num_workers=4 if sys.gettrace() is None else 0,
                    use_scheduler=False,
                    sche_factor=0.1,
                    sche_patience=10,
                    sche_threshold=0.0001,
                    writer_path="runs/{}_{}".format(name, now_string),
                    model_save_path="ckp/{}_{}.pth".format(name, now_string))
# model.train_combine(X1, X2, y, data_cache_path="cache/{}_data_60k.pkl".format(name))
model.train_combine(X1, X2, y)