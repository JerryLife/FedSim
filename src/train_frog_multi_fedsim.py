import os
import sys
import argparse
from datetime import datetime

from model.vertical_fl.FedSimModel import FedSimModel
from preprocess.ml_dataset.two_party_loader import ThreePartyLoader
from preprocess.ml_dataset.frog import load_frog

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--noise-scale', type=float, default=0.0)
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('-k', '--top-k', type=int, default=None)
args = parser.parse_args()

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/"
dataset = "Frogs_MFCCs.csv"
num_features = 22
num_common_features = 16
noise_scale = args.noise_scale

data_loader = ThreePartyLoader(num_features=num_features,
                               num_common_features=num_common_features,
                               common_feature_noise_scale=noise_scale,
                               data_fmt=load_frog, dataset_name=dataset, n_classes=2,
                               seed=0)
data_loader.load_parties(root + dataset)
data_loader.to_pickle(root + dataset + "_scale_{:.1f}_party_3".format(noise_scale) + "_loader.pkl")

data_loader = ThreePartyLoader.from_pickle(root + dataset + "_scale_{:.1f}_party_3".format(noise_scale) + "_loader.pkl")
[X1, X2, X3], y = data_loader.load_parties()
name = "frog_fedsim_noise_{:.1f}_party_3".format(noise_scale)

model = FedSimModel(num_common_features=num_common_features,
                    raw_output_dim=10,
                    feature_wise_sim=False,
                    task='multi_cls',
                    metrics=['accuracy'],
                    dataset_type='syn',
                    blocking_method='knn',
                    n_classes=10,
                    grid_min=-10.0,
                    grid_max=10.0,
                    grid_width=1.5,
                    knn_k=100,
                    filter_top_k=args.top_k,
                    kd_tree_radius=2,
                    tree_leaf_size=1000,
                    model_name=name + "_" + now_string,
                    val_rate=0.2,
                    test_rate=0.2,
                    drop_key=True,
                    device='cuda:{}'.format(args.gpu),
                    hidden_sizes=[100, 100],
                    train_batch_size=32,
                    test_batch_size=32,
                    num_epochs=100,
                    learning_rate=3e-3,
                    weight_decay=1e-3,
                    sim_learning_rate=3e-3,
                    sim_weight_decay=1e-3,
                    update_sim_freq=1,
                    num_workers=4 if sys.gettrace() is None else 0,
                    use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                    writer_path="runs/{}_{}".format(name, now_string),
                    model_save_path="ckp/{}_{}.pth".format(name, now_string),
                    sim_model_save_path="ckp/{}_{}_sim.pth".format(name, now_string),
                    log_dir=None,
                    # SplitNN parameters
                    local_hidden_sizes=[[50], [50], [50]],
                    agg_hidden_sizes=[100],
                    cut_dims=[10, 10, 10],

                    # fedsim parameters
                    merge_hidden_sizes=[400],
                    sim_hidden_sizes=[10],
                    use_conv=False,
                    merge_model_save_path="ckp/{}_{}_merge.pth".format(name, now_string),
                    merge_dropout_p=0.7,
                    conv_n_channels=4,
                    conv_kernel_v_size=7,

                    # private link parameters
                    link_epsilon=0.1,
                    link_delta=0.1,
                    link_threshold_t=0.1,
                    sim_leak_p=args.leak_p,
                    link_n_jobs=-1,
                    )
model.train_splitnn_party3(X1, X2, X3, y, data_cache_path="cache/frog_sim_{:.1f}_party_3.pkl".format(noise_scale))
# model.train_splitnn(X1, X2, y)
