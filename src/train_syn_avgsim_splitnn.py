import os
import sys
from datetime import datetime

from model.vertical_fl.MergeSimModel import MergeSimModel
from preprocess.sklearn.syn_data_generator import TwoPartyClsMany2ManyGenerator

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
num_common_features = 5
noise_scale = 0.2

syn_generator = TwoPartyClsMany2ManyGenerator.from_pickle(
    root + "syn_cls_many2many_generator_noise_{:.2f}.pkl".format(noise_scale))
[X1, X2], y = syn_generator.get_parties()
name = "syn_sim_avgsim_splitnn"
model = MergeSimModel(num_common_features=num_common_features,
                      sim_hidden_sizes=[10],
                      merge_mode='avg',
                      feature_wise_sim=False,
                      task='binary_cls',
                      metrics=['accuracy'],
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
                      drop_key=True,
                      device='cuda:0',
                      hidden_sizes=[100, 100],
                      train_batch_size=64,
                      test_batch_size=4096,
                      num_epochs=50,
                      learning_rate=1e-3,
                      weight_decay=1e-4,
                      sim_learning_rate=1e-3,
                      sim_weight_decay=1e-4,
                      sim_batch_size=4096,
                      update_sim_freq=1,
                      num_workers=4 if sys.gettrace() is None else 0,
                      use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                      writer_path="runs/{}_{}".format(name, now_string),
                      model_save_path="ckp/{}_{}.pth".format(name, now_string),
                      sim_model_save_path="ckp/{}_{}_sim.pth".format(name, now_string),
                      log_dir="log/{}_{}/".format(name, now_string),
                      # SplitNN parameters
                      local_hidden_sizes=[[100], [100]],
                      agg_hidden_sizes=[100],
                      cut_dims=[50, 50]
                      )
# model.train_combine(X1, X2, y, data_cache_path="cache/{}.pkl".format(name))
model.train_splitnn(X1, X2, y)
