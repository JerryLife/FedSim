import os
import sys
from datetime import datetime
import argparse

from model.vertical_fl.MergeSimModel import MergeSimModel
from preprocess.hdb import load_both


now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/hdb/"
hdb_dataset = root + "hdb_clean.csv"
school_dataset = root + "school_clean.csv"

num_common_features = 2
[X1, X2], y = load_both(hdb_path=hdb_dataset, airbnb_path=school_dataset, active_party='hdb')
name = "hdb_avgsim"


model = MergeSimModel(num_common_features=num_common_features,
                      sim_hidden_sizes=[10, 10],
                      merge_mode='avg',
                      feature_wise_sim=False,
                      task='regression',
                      metrics=['r2_score', 'rmse'],
                      dataset_type='real',
                      blocking_method='knn',
                      n_classes=2,
                      grid_min=-10.0,
                      grid_max=10.0,
                      grid_width=1.5,
                      knn_k=50,
                      kd_tree_radius=1e-2,
                      kd_tree_leaf_size=1000,
                      model_name=name + "_" + now_string,
                      val_rate=0.1,
                      test_rate=0.2,
                      drop_key=True,
                      device='cuda:1',
                      hidden_sizes=[200, 100],
                      train_batch_size=128,
                      test_batch_size=1024 * 4,
                      num_epochs=100,
                      learning_rate=1e-3,
                      weight_decay=1e-5,
                      sim_learning_rate=1e-3,
                      sim_weight_decay=1e-5,
                      sim_batch_size=4096,
                      update_sim_freq=1,
                      num_workers=4 if sys.gettrace() is None else 0,
                      use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                      writer_path="runs/{}_{}".format(name, now_string),
                      model_save_path="ckp/{}_{}.pth".format(name, now_string),
                      sim_model_save_path="ckp/{}_{}_sim.pth".format(name, now_string),
                      log_dir="log/{}_{}/".format(name, now_string),
                      # SplitNN parameters
                      local_hidden_sizes=[[200], [200]],
                      agg_hidden_sizes=[400],
                      cut_dims=[100, 100]
                      )
model.train_splitnn(X1, X2, y, data_cache_path="cache/beijing_sim.pkl".format(name), scale=True)

