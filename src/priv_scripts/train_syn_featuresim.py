import os
import sys
import argparse
from datetime import datetime

from model.vertical_fl.FeatureSimModel import FeatureSimModel
from preprocess.sklearn.syn_data_generator import TwoPartyClsMany2ManyGenerator

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--noise-scale', type=float, default=0.0)
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('-k', '--top-k', type=int, default=None)
args = parser.parse_args()

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../../")  # change working directory
root = "data/"
num_common_features = 5
noise_scale = args.noise_scale

syn_generator = TwoPartyClsMany2ManyGenerator.from_pickle(
    root + "syn_cls_many2many_generator_noise_{:.1f}.pkl".format(noise_scale))
[X1, X2], y = syn_generator.get_parties()
name = "syn_featuresim_noise_{:.1f}".format(noise_scale)

model = FeatureSimModel(num_common_features=num_common_features,
                        feature_wise_sim=False,
                        task='binary_cls',
                        metrics=['accuracy'],
                        dataset_type='syn',
                        blocking_method='knn_priv_str',
                        n_classes=2,
                        grid_min=-10.0,
                        grid_max=10.0,
                        grid_width=1.5,
                        knn_k=100,
                        filter_top_k=args.top_k,
                        kd_tree_radius=2,
                        tree_leaf_size=1000,
                        model_name=name + "_" + now_string,
                        val_rate=0.1,
                        test_rate=0.2,
                        drop_key=True,
                        device='cuda:{}'.format(args.gpu),
                        hidden_sizes=[100, 100],
                        train_batch_size=32,
                        test_batch_size=4096,
                        num_epochs=50,
                        learning_rate=1e-3,
                        weight_decay=1e-4,
                        num_workers=4 if sys.gettrace() is None else 0,
                        use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                        writer_path="runs/{}_{}".format(name, now_string),
                        model_save_path="ckp/{}_{}.pth".format(name, now_string),

                        # SplitNN parameters
                        local_hidden_sizes=[[100], [100]],
                        agg_hidden_sizes=[100],
                        cut_dims=[50, 50],

                        # private link parameters
                        link_epsilon=2e-3,
                        link_delta=2e-3,
                        link_threshold_t=5e-2,
                        sim_leak_p=args.leak_p,
                        link_n_jobs=-1,
                        )
model.train_splitnn(X1, X2, y, data_cache_path="cache/syn_sim_noise_{:.1f}_p_base.pkl".format(noise_scale))
# model.train_splitnn(X1, X2, y)
