import os
import sys
import argparse
from datetime import datetime

from model.vertical_fl.OrderSimModel import OrderSimModel
from preprocess.sklearn.syn_data_generator import TwoPartyClsMany2ManyGenerator

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--noise-scale', type=float, default=0.0)
args = parser.parse_args()

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
num_common_features = 5
noise_scale = args.noise_scale

syn_generator = TwoPartyClsMany2ManyGenerator.from_pickle(
    root + "syn_cls_many2many_generator_noise_{:.2f}.pkl".format(noise_scale))
[X1, X2], y = syn_generator.get_parties()
name = "syn_ordersim_noise_{:.2f}".format(noise_scale)

model = OrderSimModel(num_common_features=num_common_features,
                      sim_hidden_sizes=[50, 50],
                      raw_output_dim=3,
                      feature_wise_sim=False,
                      task='binary_cls',
                      metrics=['accuracy'],
                      dataset_type='syn',
                      blocking_method='knn',
                      n_classes=2,
                      grid_min=-10.0,
                      grid_max=10.0,
                      grid_width=1.5,
                      knn_k=100,
                      kd_tree_radius=2,
                      tree_leaf_size=1000,
                      model_name=name + "_" + now_string,
                      val_rate=0.1,
                      test_rate=0.2,
                      drop_key=True,
                      device='cuda:0',
                      hidden_sizes=[100, 100],
                      train_batch_size=32,
                      test_batch_size=4096,
                      num_epochs=50,
                      learning_rate=1e-3,
                      weight_decay=1e-4,
                      sim_learning_rate=1e-3,
                      sim_weight_decay=1e-4,
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
                      cut_dims=[50, 50],

                      use_sim=False
                      )
model.train_splitnn(X1, X2, y, data_cache_path="cache/syn_sim_noise_{:.2f}.pkl".format(noise_scale))
# model.train_splitnn(X1, X2, y)