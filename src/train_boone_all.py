import os
import sys
import argparse
from datetime import datetime

import numpy as np

from model.vertical_fl.OnePartyModel import OnePartyModel
from preprocess.ml_dataset.two_party_loader import TwoPartyLoader
from preprocess.ml_dataset.miniboone import load_miniboone

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--noise-scale', type=float, default=0.0)
args = parser.parse_args()

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory

root = "data/"
dataset = "MiniBooNE_PID.txt"
num_features = 50
num_common_features = 4
noise_scale = args.noise_scale

# data_loader = TwoPartyLoader(num_features=num_features,
#                              num_common_features=num_common_features,
#                              common_feature_noise_scale=noise_scale,
#                              data_fmt=load_miniboone, dataset_name=dataset, n_classes=2,
#                              seed=0)
# data_loader.load_parties(root + dataset)
# data_loader.to_pickle(root + dataset + "_scale_{:.2f}".format(noise_scale) + "_loader.pkl")

data_loader = TwoPartyLoader.from_pickle(root + dataset + "_scale_{:.2f}".format(noise_scale) + "_loader.pkl")
X, y = data_loader.load_dataset()
print("X got {} dimensions".format(X.shape[1]))
name = "boone_all"
model = OnePartyModel(model_name=name + "_" + now_string,
                      task='binary_cls',
                      n_classes=2,
                      val_rate=0.1,
                      test_rate=0.2,
                      device='cuda:0',
                      hidden_sizes=[100, 100],
                      train_batch_size=4096,
                      test_batch_size=4096,
                      num_epochs=150,
                      learning_rate=1e-3,
                      weight_decay=1e-5,
                      num_workers=4 if sys.gettrace() is None else 0,
                      use_scheduler=False,
                      sche_factor=0.1,
                      sche_patience=10,
                      sche_threshold=0.0001,
                      writer_path="runs/{}_{}".format(name, now_string),
                      model_save_path="ckp/{}_{}.pth".format(name, now_string)
                      )
model.train_all(X, y)
