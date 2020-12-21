import os
import sys
from datetime import datetime

from model.game.GameMergeSimModel import GameMergeSimModel
from preprocess.game.steam_ign_loader import SteamIgnLoader

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
dataset = "game"
num_common_features = 1

steam_data_train_path = root + "steam_data_train.csv"
steam_data_val_path = root + "steam_data_val.csv"
steam_data_test_path = root + "steam_data_test.csv"
ign_data_path = root + "ign_game_clean.csv"
data_loader = SteamIgnLoader(steam_data_train_path, steam_data_val_path, steam_data_test_path, ign_data_path)
steam_train_data, steam_val_data, steam_test_data, ign_data = data_loader.load_parties()

name = "game_mergesim_dlrm"
model = GameMergeSimModel(num_common_features=num_common_features,
                          sim_hidden_sizes=[10, 10],
                          top_mlp_units=[512, 256, 64],
                          dense_mlp_units=[64, 32],
                          emb_dim=16,
                          merge_mode='sim_model_avg',
                          task='binary_cls',
                          dataset_type='syn',
                          blocking_method='radius',
                          n_classes=2,
                          grid_min=-10.0,
                          grid_max=10.0,
                          grid_width=1.5,
                          knn_k=10,
                          kd_tree_radius=2,
                          kd_tree_leaf_size=1000,
                          model_name=name + "_" + now_string,
                          val_rate=0.1,
                          test_rate=0.2,
                          drop_key=True,
                          device='cuda:0',
                          hidden_sizes=[100, 100],
                          train_batch_size=384,
                          test_batch_size=1000,
                          num_epochs=50,
                          learning_rate=1e-2,
                          weight_decay=1e-4,
                          sim_learning_rate=1e-2,
                          sim_weight_decay=1e-4,
                          sim_batch_size=4096,
                          update_sim_freq=1,
                          num_workers=8 if sys.gettrace() is None else 0,
                          use_scheduler=True,
                          sche_factor=0.1,
                          sche_patience=10,
                          sche_threshold=0.0001,
                          writer_path="runs/{}_{}".format(name, now_string),
                          model_save_path="ckp/{}_{}.pth".format(name, now_string),
                          sim_model_save_path="ckp/{}_{}_sim.pth".format(name, now_string),
                          )
# model.train_combine(X1, X2, y, data_cache_path="cache/{}.pkl".format(name))
model.train_dlrm(steam_train_data, steam_val_data, steam_test_data, ign_data,
                 data_cache_path="cache/{}.pkl".format(name),
                 sim_score_cache_path="cache/game_sim_scores.pkl")
