import os
import sys
from datetime import datetime

from model.vertical_fl.Top1SimModel import Top1SimModel
from preprocess.ml_dataset.two_party_loader import TwoPartyLoader

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
dataset = "MiniBooNE_PID.txt"
num_common_features = 4

data_loader = TwoPartyLoader.from_pickle(root + dataset + "_loader.pkl")
[X1, X2], y = data_loader.load_parties()
name = "boone_sim_top1_combine"
model = Top1SimModel(num_common_features=num_common_features,
                     dataset_type='syn',
                     task='binary_cls',
                     blocking_method='radius',
                     n_classes=2,
                     grid_min=-10.0,
                     grid_max=10.0,
                     grid_width=1.5,
                     knn_k=10,
                     kd_tree_leaf_size=1000,
                     kd_tree_radius=2,
                     model_name=name + "_" + now_string,
                     val_rate=0.1,
                     test_rate=0.2,
                     drop_key=False,
                     device='cuda:0',
                     hidden_sizes=[100, 100],
                     train_batch_size=4096,
                     test_batch_size=4096,
                     num_epochs=100,
                     learning_rate=1e-3,
                     weight_decay=1e-5,
                     num_workers=8 if sys.gettrace() is None else 0,
                     use_scheduler=False,
                     sche_factor=0.1,
                     sche_patience=10,
                     sche_threshold=0.0001,
                     writer_path="runs/{}_{}".format(name, now_string),
                     model_save_path="ckp/{}_{}.pth".format(name, now_string)
                     )
model.train_combine(X1, X2, y, data_cache_path="cache/{}_data_60k.pkl".format(name))
