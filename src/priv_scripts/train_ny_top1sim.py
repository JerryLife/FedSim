import os
import sys
from datetime import datetime
import argparse

from model.vertical_fl.Top1SimModel import Top1SimModel
from preprocess.nytaxi.ny_loader import NYBikeTaxiLoader

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../../")  # change working directory
root = "data/nytaxi/"
bike_dataset = "bike_201606_clean_sample_2e5.pkl"
taxi_dataset = "taxi_201606_clean_sample_1e5.pkl"

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

num_common_features = 4
data_loader = NYBikeTaxiLoader(bike_path=root + bike_dataset, taxi_path=root + taxi_dataset, link=True)
[X1, X2], y = data_loader.load_parties()
name = "ny_top1sim"

model = Top1SimModel(num_common_features=num_common_features,
                     task='regression',
                     metrics=['r2_score', 'rmse'],
                     dataset_type='real',
                     blocking_method='knn_priv_float',
                     n_classes=2,
                     grid_min=-10.0,
                     grid_max=10.0,
                     grid_width=1.5,
                     knn_k=50,
                     kd_tree_radius=0.01,
                     tree_leaf_size=1000,
                     model_name=name + "_" + now_string,
                     val_rate=0.1,
                     test_rate=0.2,
                     drop_key=True,
                     device='cuda:{}'.format(args.gpu),
                     hidden_sizes=[200, 100],
                     train_batch_size=4096,
                     test_batch_size=4096,
                     num_epochs=50,
                     learning_rate=3e-3,
                     weight_decay=1e-5,
                     num_workers=8 if sys.gettrace() is None else 0,
                     use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                     writer_path="runs/{}_{}".format(name, now_string),
                     model_save_path="ckp/{}_{}.pth".format(name, now_string),

                     # private link parameters
                     link_epsilon=1e-1,
                     link_delta=1e-1,
                     link_threshold_t=2e-2,
                     sim_leak_p=args.leak_p,
                     link_n_jobs=-1,
                     )
model.train_splitnn(X1, X2, y, data_cache_path="cache/ny_sim_p_base.pkl", scale=True)
