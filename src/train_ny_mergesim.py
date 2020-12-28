import os
import sys
from datetime import datetime
import argparse

from model.vertical_fl.MergeSimModel import MergeSimModel
from preprocess.nytaxi.ny_loader import NYLoader


now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/nytaxi/"
aribnb_dataset = "airbnb_clean.csv"
taxi_dataset = "taxi_201606_clean_sample_1e6.csv"
# taxi_dataset = "taxi_201606_clean.csv"

num_common_features = 2
data_loader = NYLoader(airbnb_path=root + aribnb_dataset, taxi_path=root + taxi_dataset, link=True)
[X1, X2], y = data_loader.load_parties()
name = "ny_mergesim_combine"

model = MergeSimModel(num_common_features=num_common_features,
                      sim_hidden_sizes=[10, 10],
                      merge_mode='sim_model_avg',
                      task='regression',
                      dataset_type='real',
                      blocking_method='knn',
                      n_classes=2,
                      grid_min=-10.0,
                      grid_max=10.0,
                      grid_width=1.5,
                      knn_k=100,
                      kd_tree_radius=5e-4,
                      kd_tree_leaf_size=400,
                      model_name=name + "_" + now_string,
                      val_rate=0.1,
                      test_rate=0.2,
                      drop_key=True,
                      device='cuda:0',
                      hidden_sizes=[200, 100],
                      train_batch_size=1024 * 4 // 100,
                      test_batch_size=1024 * 4,
                      num_epochs=50,
                      learning_rate=1e-2,
                      weight_decay=1e-5,
                      sim_learning_rate=1e-2,
                      sim_weight_decay=1e-5,
                      sim_batch_size=4096,
                      update_sim_freq=1,
                      num_workers=8 if sys.gettrace() is None else 0,
                      use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                      writer_path="runs/{}_{}".format(name, now_string),
                      model_save_path="ckp/{}_{}.pth".format(name, now_string),
                      sim_model_save_path="ckp/{}_{}_sim.pth".format(name, now_string),
                      )
model.train_combine(X1, X2, y, data_cache_path="cache/{}.pkl".format(name), scale=True)
# model.train_combine(X1, X2, y, scale=True)
