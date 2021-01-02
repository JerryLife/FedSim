import os
import sys
from datetime import datetime
import argparse

from model.vertical_fl.Top1SimModel import Top1SimModel
from preprocess.nytaxi.ny_loader import NYBikeTaxiLoader

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/nytaxi/"
bike_dataset = "bike_201606_clean_sample_6e5.pkl"
taxi_dataset = "taxi_201606_clean.pkl"

num_common_features = 4
data_loader = NYBikeTaxiLoader(bike_path=root + bike_dataset, taxi_path=root + taxi_dataset, link=True)
[X1, X2], y = data_loader.load_parties()
name = "ny_top1sim_combine"

model = Top1SimModel(num_common_features=num_common_features,
                     task='regression',
                     dataset_type='real',
                     blocking_method='knn',
                     n_classes=2,
                     grid_min=-10.0,
                     grid_max=10.0,
                     grid_width=1.5,
                     knn_k=10,
                     kd_tree_radius=0.01,
                     kd_tree_leaf_size=1000,
                     model_name=name + "_" + now_string,
                     val_rate=0.1,
                     test_rate=0.2,
                     drop_key=True,
                     device='cuda:0',
                     hidden_sizes=[200, 100],
                     train_batch_size=1024 * 4,
                     test_batch_size=1024 * 4,
                     num_epochs=50,
                     learning_rate=3e-3,
                     weight_decay=1e-5,
                     num_workers=8 if sys.gettrace() is None else 0,
                     use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                     writer_path="runs/{}_{}".format(name, now_string),
                     model_save_path="ckp/{}_{}.pth".format(name, now_string),
                     )
model.train_combine(X1, X2, y, data_cache_path="cache/{}.pkl".format(name), scale=True)
# model.train_combine(X1, X2, y, scale=True)
