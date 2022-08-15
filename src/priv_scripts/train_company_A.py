import os
import sys
from datetime import datetime
import argparse

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import numpy as np

from model.vertical_fl.OnePartyModel import OnePartyModel
from preprocess.company import load_loan

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../../")  # change working directory

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

root = "data/company/"
dataset = "loan_subset_clean.csv"

X, y = load_loan(root + dataset)
print("X got {} dimensions".format(X.shape[1]))
name = "company_A"
# reg = LinearRegression().fit(X, y)
# score = np.sqrt(metrics.mean_squared_error(reg.predict(X), y))
# print(score)

model = OnePartyModel(model_name=name + "_" + now_string,
                      task='regression',
                      metrics=['rmse', 'r2_score', 'mae'],
                      n_classes=2,
                      val_rate=0.1,
                      test_rate=0.2,
                      device='cuda:{}'.format(args.gpu),
                      hidden_sizes=[200, 100],
                      train_batch_size=4096,
                      test_batch_size=4096,
                      num_epochs=100,
                      learning_rate=3e-3,
                      weight_decay=1e-5,
                      num_workers=4 if sys.gettrace() is None else 0,
                      use_scheduler=True,
                      sche_factor=0.1,
                      sche_patience=30,
                      sche_threshold=0.0001,
                      writer_path="runs/{}_{}".format(name, now_string),
                      model_save_path="ckp/{}_{}.pth".format(name, now_string)
                      )
model.train_single(X, y, scale=True)
