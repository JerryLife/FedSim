import os
import sys
from datetime import datetime
import argparse

from model.vertical_fl.ExactModel import ExactModel
from preprocess.company import load_both

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../../")  # change working directory
root = "data/company/"

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

num_common_features = 1
loan_dataset = root + "loan_subset_clean.csv"
company_dataset = root + "company_subset_clean.csv"
[X1, X2], y = load_both(loan_dataset, company_dataset, host_party='loan')
name = "company_exact"

model = ExactModel(num_common_features=num_common_features,
                   task='regression',
                   dataset_type='real',
                   metrics=['rmse', 'r2_score', 'mae'],
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
                   learning_rate=3e-3,
                   weight_decay=1e-5,
                   num_workers=0 if sys.gettrace() is None else 0,
                   use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                   writer_path="runs/{}_{}".format(name, now_string),
                   model_save_path="ckp/{}_{}.pth".format(name, now_string),

                   # SplitNN parameters
                   local_hidden_sizes=[[100], [100]],
                   agg_hidden_sizes=[200],
                   cut_dims=[100, 100],
                   )
model.train_splitnn(X1, X2, y, data_cache_path="cache/company_subset_exact.pkl".format(name), scale=True)
# model.train_splitnn(X1, X2, y, scale=True)
