import os
import sys
from datetime import datetime

from model.vertical_fl.MergeSimModelV2 import MergeSimModel
from preprocess.sklearn.syn_data_generator import TwoPartyClsMany2ManyGenerator

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
num_common_features = 5

syn_generator = TwoPartyClsMany2ManyGenerator.from_pickle(
    root + "syn_cls_many2many_generator.pkl")
[X1, X2], y = syn_generator.get_parties()
name = "syn_sim_merge_combine_v2"
model = MergeSimModel(num_common_features=num_common_features,
                      sim_hidden_sizes=[10, 10],
                      merge_mode='sim_model_avg',
                      task='binary_cls',
                      dataset_type='syn',
                      blocking_method='grid',
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
                      drop_key=False,
                      device='cuda:0',
                      hidden_sizes=[100, 100],
                      train_batch_size=64,
                      test_batch_size=4096,
                      num_epochs=100,
                      learning_rate=1e-3,
                      weight_decay=1e-5,
                      sim_learning_rate=1e-3,
                      sim_weight_decay=1e-5,
                      sim_batch_size=4096,
                      update_sim_freq=1,
                      num_workers=8 if sys.gettrace() is None else 0,
                      use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                      writer_path="runs/{}_{}".format(name, now_string),
                      model_save_path="ckp/{}_{}.pth".format(name, now_string),
                      sim_model_save_path="ckp/{}_{}_sim.pth".format(name, now_string)
                      )
model.train_combine(X1, X2, y, data_cache_path="cache/{}.pkl".format(name))

