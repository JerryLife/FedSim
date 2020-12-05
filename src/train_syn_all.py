import os
import sys
from datetime import datetime

import numpy as np

from model.vertical_fl.OnePartyModel import OnePartyModel
from preprocess.sklearn.syn_data_generator import TwoPartyClsMany2ManyGenerator

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
num_common_features = 5

# syn_generator = TwoPartyClsMany2ManyGenerator(num_samples=60000,
#                                               num_features_per_party=[50, 50],
#                                               num_common_features=num_common_features,
#                                               common_feature_noise_scale=0.1,
#                                               common_feature_noise_bias=0.0,
#                                               n_informative=10,
#                                               n_redundant=10,
#                                               n_clusters_per_class=3,
#                                               class_sep=0.3,
#                                               n_classes=2,
#                                               seed=512
#                                               )
# syn_generator.get_parties()
# syn_generator.to_pickle(root + "syn_cls_many2many_generator.pkl")

syn_generator = TwoPartyClsMany2ManyGenerator.from_pickle(root + "syn_cls_many2many_generator.pkl")
X, y = syn_generator.get_global()
print("X got {} dimensions".format(X.shape[1]))
name = "syn_all"
model = OnePartyModel(model_name=name + "_" + now_string,
                      task='binary_cls',
                      n_classes=2,
                      val_rate=0.1,
                      test_rate=0.2,
                      device='cuda:0',
                      hidden_sizes=[100, 100],
                      train_batch_size=4096,
                      test_batch_size=4096,
                      num_epochs=100,
                      learning_rate=1e-3,
                      weight_decay=1e-4,
                      num_workers=4 if sys.gettrace() is None else 0,
                      use_scheduler=False,
                      sche_factor=0.1,
                      sche_patience=10,
                      sche_threshold=0.0001,
                      writer_path="runs/{}_{}".format(name, now_string),
                      model_save_path="ckp/{}_{}.pth".format(name, now_string)
                      )
model.train_all(X, y)
