import os
import sys
from datetime import datetime
import argparse

from model.vertical_fl.MergeSimModel import MergeSimModel
from preprocess.nytaxi.ny_loader import NYBikeTaxiLoader


now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../../")  # change working directory
root = "data/nytaxi/"
bike_dataset = "bike_201606_clean_sample_2e5.pkl"
taxi_dataset = "taxi_201606_clean.pkl"
# taxi_dataset = "taxi_201606_clean.csv"

num_common_features = 4
data_loader = NYBikeTaxiLoader(bike_path=root + bike_dataset, taxi_path=root + taxi_dataset, link=True)
[X1, X2], y = data_loader.load_parties()
name = "ny_mergesim"


model = MergeSimModel(num_common_features=num_common_features,
                      sim_hidden_sizes=[10, 10],
                      merge_mode='common_model_avg',
                      feature_wise_sim=True,
                      task='regression',
                      metrics=['r2_score', 'rmse'],
                      dataset_type='real',
                      blocking_method='knn',
                      n_classes=2,
                      grid_min=-10.0,
                      grid_max=10.0,
                      grid_width=1.5,
                      knn_k=50,
                      kd_tree_radius=2e-3,
                      tree_leaf_size=1000,
                      model_name=name + "_" + now_string,
                      val_rate=0.1,
                      test_rate=0.2,
                      drop_key=True,
                      device='cuda:0',
                      hidden_sizes=[200, 100],
                      train_batch_size=64,
                      test_batch_size=1024 * 4,
                      num_epochs=50,
                      learning_rate=3e-3,
                      weight_decay=1e-5,
                      sim_learning_rate=3e-3,
                      sim_weight_decay=1e-5,
                      sim_batch_size=4096,
                      update_sim_freq=1,
                      num_workers=8 if sys.gettrace() is None else 0,
                      use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                      writer_path="runs/{}_{}".format(name, now_string),
                      model_save_path="ckp/{}_{}.pth".format(name, now_string),
                      sim_model_save_path="ckp/{}_{}_sim.pth".format(name, now_string),
                      # SplitNN parameters
                      local_hidden_sizes=[[100], [100]],
                      agg_hidden_sizes=[100],
                      cut_dims=[50, 50]
                      )
model.train_splitnn(X1, X2, y, data_cache_path="cache/ny_sim_multidim.pkl", scale=True)

