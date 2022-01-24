import os
import sys
import argparse
from datetime import datetime

from model.vertical_fl.Top1SimModel import Top1SimModel
from preprocess.sklearn.syn_data_generator import TwoPartyOne2OneLinearGenerator

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--noise-scale', type=float, default=0.0)
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
num_common_features = 5
noise_scale = args.noise_scale

syn_generator = TwoPartyOne2OneLinearGenerator.from_pickle(
    root + "syn_cls_one2one_linear_generator.pkl".format(noise_scale))
[X1, X2], y = syn_generator.get_parties()
name = "random_top1sim".format(noise_scale)

model = Top1SimModel(num_common_features=num_common_features,
                     dataset_type='syn',
                     task='binary_cls',
                     metrics=['accuracy'],
                     blocking_method='knn',
                     knn_k=50,
                     n_classes=2,
                     grid_min=-10.0,
                     grid_max=10.0,
                     grid_width=1.5,
                     model_name=name + "_" + now_string,
                     val_rate=0.1,
                     test_rate=0.2,
                     drop_key=True,
                     device='cuda:{}'.format(args.gpu),
                     hidden_sizes=[100, 100],
                     train_batch_size=32,
                     test_batch_size=4096,
                     num_epochs=100,
                     learning_rate=1e-3,
                     weight_decay=1e-5,
                     # IMPORTANT: Set num_workers to 0 to prevent deadlock on RTX3090 for unknown reason.
                     num_workers=0,
                     use_scheduler=False,
                     sche_factor=0.1,
                     sche_patience=10,
                     sche_threshold=0.0001,
                     writer_path="runs/{}_{}".format(name, now_string),
                     model_save_path="ckp/{}_{}.pth".format(name, now_string),
                     # SplitNN parameters
                     local_hidden_sizes=[[50], [50]],
                     agg_hidden_sizes=[100],
                     cut_dims=[20, 20],

                     # private link parameters
                     link_epsilon=0.1,
                     link_delta=0.1,
                     link_threshold_t=0.1,
                     sim_leak_p=args.leak_p
                     )
model.train_splitnn(X1, X2, y, data_cache_path="cache/random_fedsim.pkl".format(noise_scale))
# model.train_splitnn(X1, X2, y)
