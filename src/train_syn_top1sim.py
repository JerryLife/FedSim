import os
import sys
from datetime import datetime

from model.vertical_fl.Top1SimModel import Top1SimModel
from preprocess.sklearn.syn_data_generator import TwoPartyClsMany2ManyGenerator

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
num_common_features = 5
noise_scale = 0

syn_generator = TwoPartyClsMany2ManyGenerator.from_pickle(
    root + "syn_cls_many2many_generator.pkl")
[X1, X2], y = syn_generator.get_parties()
name = "syn_sim_top1_combine"
model = Top1SimModel(num_common_features=num_common_features,
                     dataset_type='syn',
                     task='binary_cls',
                     n_classes=2,
                     grid_min=-10.0,
                     grid_max=10.0,
                     grid_width=1.5,
                     model_name=name + "_" + now_string,
                     val_rate=0.1,
                     test_rate=0.2,
                     drop_key=False,
                     device='cuda:0',
                     hidden_sizes=[100, 100],
                     train_batch_size=4096,
                     test_batch_size=4096,
                     num_epochs=100,
                     learning_rate=1e-3,
                     weight_decay=1e-5,
                     num_workers=8 if sys.gettrace() is None else 0,
                     use_scheduler=False,
                     sche_factor=0.1,
                     sche_patience=10,
                     sche_threshold=0.0001,
                     writer_path="runs/{}_{}".format(name, now_string),
                     model_save_path="ckp/{}_{}.pth".format(name, now_string)
                     )
model.train_combine(X1, X2, y, data_cache_path="cache/{}.pkl".format(name))
