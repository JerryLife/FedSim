import os
import sys
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

from synthetic.syn_two_party_model import ThresholdSimModel
from synthetic.syn_data_generator import TwoPartyClsMany2ManyGenerator

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
num_common_features = 5

syn_generator = TwoPartyClsMany2ManyGenerator.from_pickle(root + "syn_cls_many2many_generator.pkl")
[X1, X2], y = syn_generator.get_parties()

def run(sim_threshold):
    name = "syn_threshold_{:.2f}_sim_combine".format(sim_threshold)
    model = ThresholdSimModel(num_common_features=num_common_features,
                              task='binary_cls',
                              n_classes=2,
                              sim_threshold=sim_threshold,
                              grid_min=-10.0,
                              grid_max=10.0,
                              grid_width=1.5,
                              model_name=name + "_" + now_string,
                              val_rate=0.1,
                              test_rate=0.2,
                              drop_key=True,
                              device='cuda:0',
                              hidden_sizes=[100, 100],
                              train_batch_size=4096,
                              test_batch_size=4096,
                              num_epochs=100,
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
    return model.train_combine(X1, X2, y, data_cache_path="cache/{}_data_60k.pkl".format(name))

sim_thresholds = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.83, 0.86, 0.9, 0.93, 0.96, 0.99])
results = Parallel(n_jobs=6)(delayed(run)(t) for t in sim_thresholds)
# results = []
# for t in sim_thresholds:
#     result = run(t)
#     results.append(result)
print("All done------------------------------------------------------")
for threshold, (train_sample_acc, val_sample_acc, test_sample_acc,
                train_acc, val_acc, test_acc) in zip(sim_thresholds, results):
    print("Threshold {:.2f}: train sample acc = {:.4f}, val sample acc = {:.4f}, test sample acc = {:.4f}\n"
          "                  train acc = {:.4f}, val acc = {:.4f}, test acc = {:.4f}"
          .format(threshold, train_sample_acc, val_sample_acc, test_sample_acc,
                  train_acc, val_acc, test_acc))

