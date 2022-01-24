import os
import sys
from datetime import datetime
import argparse

from model.vertical_fl.ExactModel import ExactModel
from preprocess.game import load_both

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../../")  # change working directory
root = "data/game/"
rawg_dataset = root + "rawg_clean.csv"
steam_dataset = root + "steam_clean.csv"

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

num_common_features = 1
[X1, X2], y = load_both(rawg_path=rawg_dataset, steam_path=steam_dataset,
                        active_party='steam')
name = "game_exact"

model = ExactModel(num_common_features=num_common_features,
                   task='binary_cls',
                   dataset_type='real',
                   metrics=['accuracy'],
                   n_classes=2,
                   model_name=name + "_" + now_string,
                   val_rate=0.1,
                   test_rate=0.2,
                   drop_key=True,
                   device='cuda:{}'.format(args.gpu),
                   hidden_sizes=[200, 100],
                   train_batch_size=1024 * 4,
                   test_batch_size=1024 * 4,
                   num_epochs=200,
                   learning_rate=1e-3,
                   weight_decay=1e-5,
                   num_workers=4 if sys.gettrace() is None else 0,
                   use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                   writer_path="runs/{}_{}".format(name, now_string),
                   model_save_path="ckp/{}_{}.pth".format(name, now_string),

                   # SplitNN parameters
                   local_hidden_sizes=[[100], [100]],
                   agg_hidden_sizes=[400],
                   cut_dims=[50, 50],
                   )
# model.train_splitnn(X1, X2, y, data_cache_path="cache/game_exact.pkl".format(name))
model.train_splitnn(X1, X2, y, scale=True)
