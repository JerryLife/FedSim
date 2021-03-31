import os
import sys
from datetime import datetime
import argparse
import numpy as np

from model.vertical_fl.FedSimModel import FedSimModel
from preprocess.game import load_both

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/game/"
rawg_dataset = root + "rawg_clean.csv"
steam_dataset = root + "steam_clean.csv"

num_common_features = 1
[X1, X2], y = load_both(rawg_path=rawg_dataset, steam_path=steam_dataset,
                        active_party='steam')
name = "game_fedsim"

model = FedSimModel(num_common_features=num_common_features,
                    raw_output_dim=10,
                    feature_wise_sim=False,
                    task='multi_cls',
                    metrics=['accuracy'],
                    dataset_type='real',
                    blocking_method='knn_priv_str',
                    n_classes=3,
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
                    device='cuda:0',
                    hidden_sizes=[200, 100],
                    train_batch_size=128,
                    test_batch_size=1024 * 4,
                    num_epochs=100,
                    learning_rate=1e-3,
                    weight_decay=1e-5,
                    update_sim_freq=1,
                    num_workers=4 if sys.gettrace() is None else 0,
                    use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                    writer_path="runs/{}_{}".format(name, now_string),
                    model_save_path="ckp/{}_{}.pth".format(name, now_string),
                    sim_model_save_path="ckp/{}_{}_sim.pth".format(name, now_string),
                    log_dir="log/{}_{}/".format(name, now_string),
                    # SplitNN parameters
                    local_hidden_sizes=[[100], [100]],
                    agg_hidden_sizes=[400],
                    cut_dims=[50, 50],

                    # fedsim parameters
                    use_conv=True,
                    merge_hidden_sizes=[100],
                    sim_hidden_sizes=[10],
                    merge_model_save_path="ckp/{}_{}_merge.pth".format(name, now_string),
                    merge_dropout_p=0.2,
                    conv_n_channels=8,
                    conv_kernel_v_size=5,

                    # linkage parameters
                    edit_distance_threshold=1,
                    n_hash_func=10,
                    collision_rate=0.05,
                    qgram_q=2,
                    link_delta=0.1,
                    n_hash_lsh=20,
                    )
model.train_splitnn(X1, X2, y, data_cache_path="cache/game_sim.pkl".format(name), scale=True)
# model.train_splitnn(X1, X2, y, scale=True)
