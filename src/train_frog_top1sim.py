import os
import sys
from datetime import datetime
import argparse

from model.vertical_fl.Top1SimModel import Top1SimModel
from preprocess.ml_dataset.two_party_loader import TwoPartyLoader

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--noise-scale', type=float, default=0.2)
args = parser.parse_args()

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
dataset = "Frogs_MFCCs.csv"
num_common_features = 16
noise_scale = args.noise_scale

data_loader = TwoPartyLoader.from_pickle(root + dataset + "_scale_{:.2f}".format(noise_scale) + "_loader.pkl")
[X1, X2], y = data_loader.load_parties()
name = "frog_top1sim_noise_{:.2f}".format(noise_scale)

model = Top1SimModel(num_common_features=num_common_features,
                     dataset_type='syn',
                     task='multi_cls',
                     metrics=['accuracy'],
                     blocking_method='knn',
                     n_classes=10,
                     grid_min=-10.0,
                     grid_max=10.0,
                     grid_width=1.5,
                     knn_k=100,
                     kd_tree_leaf_size=1000,
                     kd_tree_radius=2,
                     model_name=name + "_" + now_string,
                     val_rate=0.1,
                     test_rate=0.2,
                     drop_key=True,
                     device='cuda:0',
                     hidden_sizes=[200, 100],
                     train_batch_size=4096,
                     test_batch_size=4096,
                     num_epochs=100,
                     learning_rate=3e-2,
                     weight_decay=1e-4,
                     num_workers=4 if sys.gettrace() is None else 0,
                     use_scheduler=False,
                     sche_factor=0.1,
                     sche_patience=10,
                     sche_threshold=0.0001,
                     writer_path="runs/{}_{}".format(name, now_string),
                     model_save_path="ckp/{}_{}.pth".format(name, now_string),
                     # SplitNN parameters
                     local_hidden_sizes=[[100], [100]],
                     agg_hidden_sizes=[100],
                     cut_dims=[50, 50]
                     )
model.train_splitnn(X1, X2, y)