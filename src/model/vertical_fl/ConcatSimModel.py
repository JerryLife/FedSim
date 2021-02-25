import os
import pickle
import gc
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import deprecation
from tqdm import tqdm
from torchsummaryX import summary
import torch_optimizer as adv_optim

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from .SimModel import SimModel
from model.base import *
from .SimModel import SimDataset
from utils import get_split_points


class ConcatSimModel(SimModel):
    def __init__(self, num_common_features, sim_hidden_sizes=None,
                 sim_model_save_path=None, update_sim_freq=1, raw_output_dim=1,
                 sim_learning_rate=1e-3, sim_weight_decay=1e-5, sim_batch_size=128,
                 log_dir=None, use_sim=True,
                 **kwargs):
        super().__init__(num_common_features, **kwargs)
        self.use_sim = use_sim
        self.raw_output_dim = raw_output_dim
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.update_sim_freq = update_sim_freq
        self.sim_model_save_path = sim_model_save_path
        self.sim_batch_size = sim_batch_size
        self.sim_weight_decay = sim_weight_decay
        self.sim_learning_rate = sim_learning_rate
        self.sim_model = None

        if sim_hidden_sizes is None:
            self.sim_hidden_sizes = [10]
        else:
            self.sim_hidden_sizes = sim_hidden_sizes

        self.data1_shape = None
        self.data2_shape = None

        assert self.blocking_method == 'knn'    # the pairing must be consistent

    def train_splitnn(self, data1, data2, labels, data_cache_path=None, scale=False, torch_seed=None):
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            # For CUDA >= 10.2 only
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

            torch.set_deterministic(True)

        start_time = datetime.now()
        train_dataset, val_dataset, test_dataset, y_scaler = \
            self.prepare_train_combine(data1, data2, labels, data_cache_path, scale)
        time_duration_sec = (datetime.now() - start_time).seconds
        print("Preparing time (sec): {}".format(time_duration_sec))

        start_time = datetime.now()
        print("Initializing dataloader")
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                  num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context,
                                  collate_fn=self.var_collate_fn)

        print("Done")
        self.data1_shape = data1.shape
        self.data2_shape = data2.shape
        if self.drop_key:
            num_features = data1.shape[1] + data2.shape[1] - 2 * self.num_common_features
        else:
            num_features = data1.shape[1] + data2.shape[1]

        print("Prepare for training")
        num_parties = 2
        input_dims = [self.data1_shape[1] - self.num_common_features,
                      self.data2_shape[1] - self.num_common_features]

        if self.task == 'binary_cls':
            output_dim = 1
            local_models = [MLP(input_size=input_dims[i], hidden_sizes=self.local_hidden_sizes[i],
                                output_size=self.cut_dims[i], activation=None) for i in range(num_parties)]
            agg_model = MLP(input_size=sum(self.cut_dims), hidden_sizes=self.merge_hidden_sizes,
                            output_size=self.raw_output_dim, activation='sigmoid')
            self.model = SplitNN(local_models, input_dims, agg_model)
            criterion = nn.BCELoss()
            val_criterion = nn.BCELoss()
        elif self.task == 'multi_cls':
            output_dim = self.n_classes
            local_models = [MLP(input_size=input_dims[i], hidden_sizes=self.local_hidden_sizes[i],
                                output_size=self.cut_dims[i], activation=None) for i in range(num_parties)]
            agg_model = MLP(input_size=sum(self.cut_dims), hidden_sizes=self.merge_hidden_sizes,
                            output_size=self.raw_output_dim, activation=None)
            self.model = SplitNN(local_models, input_dims, agg_model)
            criterion = nn.CrossEntropyLoss()
            val_criterion = nn.CrossEntropyLoss()
        elif self.task == 'regression':
            output_dim = 1
            local_models = [MLP(input_size=input_dims[i], hidden_sizes=self.local_hidden_sizes[i],
                                output_size=self.cut_dims[i], activation=None) for i in range(num_parties)]
            agg_model = MLP(input_size=sum(self.cut_dims), hidden_sizes=self.merge_hidden_sizes,
                            output_size=self.raw_output_dim, activation='sigmoid')
            self.model = SplitNN(local_models, input_dims, agg_model)
            criterion = nn.MSELoss()
            val_criterion = nn.MSELoss()
        else:
            assert False, "Unsupported task"
        self.model = self.model.to(self.device)
        if self.feature_wise_sim:
            self.sim_model = MLP(input_size=(self.num_common_features + self.raw_output_dim) * self.knn_k,
                                 hidden_sizes=self.sim_hidden_sizes,
                                 output_size=output_dim, activation='sigmoid').to(self.device)
        else:
            self.sim_model = MLP(input_size=(self.raw_output_dim + 1) * self.knn_k,
                                 hidden_sizes=self.sim_hidden_sizes,
                                 output_size=output_dim, activation='sigmoid').to(self.device)
        optimizer = adv_optim.Lamb(list(self.model.parameters()) + list(self.sim_model.parameters()),
                                   lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.sche_factor,
                                                             patience=self.sche_patience,
                                                             threshold=self.sche_threshold)
        else:
            scheduler = None

        best_train_metric_scores = [m.worst for m in self.metrics_f]
        best_val_metric_scores = [m.worst for m in self.metrics_f]
        best_test_metric_scores = [m.worst for m in self.metrics_f]
        print("Start training")
        summary(self.model, torch.zeros([self.train_batch_size, num_features]).to(self.device))
        summary(self.sim_model, torch.zeros([1, (self.num_common_features + self.raw_output_dim) * self.knn_k
                if self.feature_wise_sim else (self.raw_output_dim + 1) * self.knn_k])
                .to(self.device))
        print(str(self))
        # # debug
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.num_epochs):
            # train
            train_loss = 0.0
            n_train_batches = 0
            self.sim_model.train()
            self.model.train()
            all_preds = np.zeros((0, 1))
            all_labels = np.zeros(0)
            for data_batch, labels, weights, idx1, idx1_unique in tqdm(train_loader, desc="Train Main"):
                data_batch = data_batch.to(self.device).float()
                labels = labels.to(self.device).float()

                if self.feature_wise_sim:
                    data = data_batch[:, self.num_common_features:]
                    if self.use_sim:
                        sim_scores = data_batch[:, :self.num_common_features]
                    else:
                        sim_scores = torch.ones([data_batch.shape[0], self.num_common_features]).to(self.device)
                else:
                    data = data_batch[:, 1:]
                    if self.use_sim:
                        sim_scores = data_batch[:, 0].reshape(-1, 1)
                    else:
                        sim_scores = torch.ones([data_batch.shape[0], 1]).to(self.device)

                # train main model
                optimizer.zero_grad()
                outputs = self.model(data)

                outputs_batch = torch.zeros([0, output_dim]).to(self.device)
                idx1_split_points = get_split_points(idx1, idx1.shape[0])
                labels_sim = torch.zeros(0).to(self.device)
                for i in range(idx1_unique.shape[0]):
                    start = idx1_split_points[i]
                    end = idx1_split_points[i + 1]

                    outputs_flat = outputs[start:end].flatten()
                    sim_scores_flat = sim_scores[start:end].flatten()
                    output_i = self.sim_model(torch.cat([outputs_flat, sim_scores_flat]))

                    # # bound threshold
                    # output_i[output_i > 1.] = 1.
                    # output_i[output_i < 0.] = 0.

                    outputs_batch = torch.cat([outputs_batch, output_i.reshape(-1, 1)], dim=0)
                    labels_sim = torch.cat([labels_sim, labels[i].repeat(end - start)], dim=0)
                if self.task == 'binary_cls':
                    outputs_batch = outputs_batch.flatten()
                    loss = criterion(outputs_batch, labels)
                    preds = outputs_batch > 0.5
                elif self.task == 'multi_cls':
                    loss = criterion(outputs_batch, labels.long())
                    preds = torch.argmax(outputs_batch, dim=1)
                elif self.task == 'regression':
                    outputs_batch = outputs_batch.flatten()
                    loss = criterion(outputs_batch, labels)
                    preds = outputs_batch
                else:
                    assert False, "Unsupported task"

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = preds.reshape(-1, 1).detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                n_train_batches += 1
                if self.task == 'regression' and y_scaler is not None:
                    preds = y_scaler.inverse_transform(preds.reshape(-1, 1))
                    labels = y_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()

                all_preds = np.concatenate([all_preds, preds])
                all_labels = np.concatenate([all_labels, labels])

            train_loss /= n_train_batches
            train_metric_scores = []
            for metric_f in self.metrics_f:
                train_metric_scores.append(metric_f(all_preds, all_labels))

            # validation and test
            val_loss, val_metric_scores = self.eval_merge_score(val_dataset, val_criterion,
                                                                'Val', y_scaler=y_scaler)
            test_loss, test_metric_scores = self.eval_merge_score(test_dataset, val_criterion,
                                                                  'Test', y_scaler=y_scaler)
            if self.use_scheduler:
                scheduler.step(val_loss)

            # The first metric determines early stopping
            if self.metrics[0] in ['accuracy', 'r2_score']:
                is_best = (val_metric_scores[0] > best_val_metric_scores[0])
            elif self.metrics[0] in ['rmse']:
                is_best = (val_metric_scores[0] < best_val_metric_scores[0])
            else:
                assert False, "Unsupported metric"

            if is_best:
                best_train_metric_scores = train_metric_scores
                best_val_metric_scores = val_metric_scores
                best_test_metric_scores = test_metric_scores
                if self.model_save_path is not None:
                    torch.save(self.model.state_dict(), self.model_save_path)
                    torch.save(self.sim_model.state_dict(), self.sim_model_save_path)

            print("Epoch {}: {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}"
                  .format(epoch + 1, "Loss:", train_loss, val_loss, test_loss))
            self.writer.add_scalars('Loss', {'Train': train_loss,
                                             'Validation': val_loss,
                                             'Test': test_loss}, epoch + 1)
            for i in range(len(self.metrics)):
                print("          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}"
                      .format(self.metrics_f[i].name, train_metric_scores[i],
                              val_metric_scores[i], test_metric_scores[i]))
                self.writer.add_scalars(self.metrics_f[i].name, {'Train': train_metric_scores[i],
                                                                 'Validation': val_metric_scores[i],
                                                                 'Test': test_metric_scores[i]}, epoch + 1)
            print("Best:")
            for i in range(len(self.metrics)):
                print("          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}"
                      .format(self.metrics_f[i].name, best_train_metric_scores[i],
                              best_val_metric_scores[i], best_test_metric_scores[i]))

        time_duration_sec = (datetime.now() - start_time).seconds
        print("Training time (sec): {}".format(time_duration_sec))

        return best_test_metric_scores

    def eval_merge_score(self, val_dataset, loss_criterion, name='Val', y_scaler=None):
        assert self.model is not None, "Model has not been initialized"

        val_loader = DataLoader(val_dataset, batch_size=self.train_batch_size, shuffle=True,
                                num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context,
                                collate_fn=self.var_collate_fn)

        val_loss = 0.0
        n_val_batches = 0
        output_dim = 1 if self.task in ['binary_cls', 'regression'] else self.n_classes
        all_preds = np.zeros((0, 1))
        all_labels = np.zeros(0)
        with torch.no_grad():
            self.model.eval()
            self.sim_model.eval()
            for data_batch, labels, weights, idx1, idx1_unique in tqdm(val_loader, desc=name):
                data_batch = data_batch.to(self.device).float()
                labels = labels.to(self.device).float()

                if self.feature_wise_sim:
                    data = data_batch[:, self.num_common_features:]
                    if self.use_sim:
                        sim_scores = data_batch[:, :self.num_common_features]
                    else:
                        sim_scores = torch.ones([data_batch.shape[0], self.num_common_features]).to(self.device)
                else:
                    data = data_batch[:, 1:]
                    if self.use_sim:
                        sim_scores = data_batch[:, 0].reshape(-1, 1)
                    else:
                        sim_scores = torch.ones([data_batch.shape[0], 1]).to(self.device)

                outputs = self.model(data)

                outputs_batch = torch.zeros([0, output_dim]).to(self.device)
                idx1_split_points = get_split_points(idx1, idx1.shape[0])
                labels_sim = torch.zeros(0).to(self.device)
                for i in range(idx1_unique.shape[0]):
                    start = idx1_split_points[i]
                    end = idx1_split_points[i + 1]

                    outputs_flat = outputs[start:end].flatten()
                    sim_scores_flat = sim_scores[start:end].flatten()
                    try:
                        output_i = self.sim_model(torch.cat([outputs_flat, sim_scores_flat]))
                    except Exception:
                        pass

                    # # bound threshold
                    # output_i[output_i > 1.] = 1.
                    # output_i[output_i < 0.] = 0.

                    outputs_batch = torch.cat([outputs_batch, output_i.reshape(-1, 1)], dim=0)
                    labels_sim = torch.cat([labels_sim, labels[i].repeat(end - start)], dim=0)
                if self.task == 'binary_cls':
                    outputs_batch = outputs_batch.flatten()
                    loss = loss_criterion(outputs_batch, labels)
                    preds = outputs_batch > 0.5
                elif self.task == 'multi_cls':
                    loss = loss_criterion(outputs_batch, labels.long())
                    preds = torch.argmax(outputs_batch, dim=1)
                elif self.task == 'regression':
                    outputs_batch = outputs_batch.flatten()
                    loss = loss_criterion(outputs_batch, labels)
                    preds = outputs_batch
                else:
                    assert False, "Unsupported task"

                preds = preds.reshape(-1, 1).detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                val_loss += loss.item()
                n_val_batches += 1

                if self.task == 'regression' and y_scaler is not None:
                    preds = y_scaler.inverse_transform(preds.reshape(-1, 1))
                    labels = y_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()

                all_preds = np.concatenate([all_preds, preds])
                all_labels = np.concatenate([all_labels, labels])

            metric_scores = []
            for metric_f in self.metrics_f:
                metric_scores.append(metric_f(all_preds, all_labels))

            val_loss /= n_val_batches

        return val_loss, metric_scores
