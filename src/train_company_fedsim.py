import os
import sys
from datetime import datetime
import argparse
import numpy as np

from model.vertical_fl.FedSimModel import FedSimModel
from preprocess.company import load_both

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/company/"
company_dataset = root + "company_subset_clean.csv"
loan_dataset = root + "loan_subset_clean.csv"

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('-k', '--top-k', type=int, default=None)
args = parser.parse_args()

num_common_features = 1
[X1, X2], y = load_both(company_path=company_dataset, loan_path=loan_dataset,
                        host_party='loan')
name = "company_fedsim"

model = FedSimModel(num_common_features=num_common_features,
                    raw_output_dim=10,
                    feature_wise_sim=False,
                    task='regression',
                    metrics=['rmse', 'mae', 'r2_score'],
                    dataset_type='real',
                    blocking_method='knn_priv_str',
                    n_classes=2,
                    grid_min=-10.0,
                    grid_max=10.0,
                    grid_width=1.5,
                    knn_k=50,
                    filter_top_k=args.top_k,
                    kd_tree_radius=1e-2,
                    tree_leaf_size=1000,
                    model_name=name + "_" + now_string,
                    val_rate=0.1,
                    test_rate=0.2,
                    drop_key=True,
                    device='cuda:{}'.format(args.gpu),
                    hidden_sizes=[200, 100],
                    train_batch_size=32,
                    test_batch_size=1024 * 4,
                    num_epochs=100,
                    learning_rate=1e-3,
                    weight_decay=1e-5,
                    update_sim_freq=1,
                    num_workers=4 if sys.gettrace() is None else 0,
                    use_scheduler=False, sche_factor=0.1, sche_patience=10, sche_threshold=0.0001,
                    writer_path="runs/{}_{}".format(name, now_string),
                    model_save_path="ckp/{}_{}.pth".format(name, now_string),
                    sim_model_save_path="ckp/{}_{}_sim.pth".format(name, now_string),
                    log_dir="log/{}_{}/".format(name, now_string),
                    # SplitNN parameters
                    local_hidden_sizes=[[100], [100]],
                    agg_hidden_sizes=[100],
                    cut_dims=[50, 50],

                    # fedsim parameters
                    use_conv=True,
                    merge_hidden_sizes=[200],
                    sim_hidden_sizes=[10],
                    merge_model_save_path="ckp/{}_{}_merge.pth".format(name, now_string),
                    merge_dropout_p=0.7,
                    conv_n_channels=4,
                    conv_kernel_v_size=7,

                    # linkage parameters
                    edit_distance_threshold=10,
                    n_hash_func=50,
                    collision_rate=1e-2,
                    qgram_q=4,
                    link_delta=1e-2,
                    n_hash_lsh=50,
                    psig_p=8,
                    sim_leak_p=args.leak_p,
                    )
model.train_splitnn(X1, X2, y, data_cache_path="cache/company_subset_sim_p_base_0.1.pkl".format(name), scale=True)
# model.train_splitnn(X1, X2, y)
