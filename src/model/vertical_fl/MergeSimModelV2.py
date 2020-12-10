import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchlars
from tqdm import tqdm
from torchsummaryX import summary

from .SimModel import SimModel
from model.base.MLP import MLP
from .MergeSimModel import AvgMergeDataset as AvgMergeDatasetV1


class AvgMergeDataset(Dataset):
    def __init__(self, data1, data2, labels, data_idx):
        assert data1.shape[0] == data2.shape[0] == data_idx.shape[0]
        # remove similarity scores in data1 (at column 0)
        data1_labels = np.concatenate([data1[:, 1:], labels.reshape(-1, 1)], axis=1)

        print("Grouping data")
        grouped_data1 = {}
        grouped_data2 = {}
        for i in range(data_idx.shape[0]):
            idx1, idx2 = data_idx[i]
            new_data2 = np.concatenate([idx2.reshape(1, 1), data2[i].reshape(1, -1)], axis=1)
            if idx1 in grouped_data2:
                grouped_data2[idx1] = np.concatenate([grouped_data2[idx1], new_data2], axis=0)
            else:
                grouped_data2[idx1] = new_data2
            grouped_data1[idx1] = data1_labels[i]
        print("Done")

        group1_data_idx = np.array(list(grouped_data1.keys()))
        group1_data1_labels = np.array(list(grouped_data1.values()))
        group2_data_idx = np.array(list(grouped_data2.keys()))
        group2_data2 = np.array(list(grouped_data2.values()), dtype='object')

        print("Sorting data")
        group1_order = group1_data_idx.argsort()
        group2_order = group2_data_idx.argsort()

        group1_data_idx = group1_data_idx[group1_order]
        group1_data1_labels = group1_data1_labels[group1_order]
        group2_data_idx = group2_data_idx[group2_order]
        group2_data2 = group2_data2[group2_order]
        assert (group1_data_idx == group2_data_idx).all()
        print("Done")

        self.data1_idx: np.ndarray = group1_data_idx
        data1: np.ndarray = group1_data1_labels[:, :-1]
        self.labels: torch.Tensor = torch.from_numpy(group1_data1_labels[:, -1])
        data2: list = group2_data2

        print("Retrieve data")
        data_list = []
        weight_list = []
        data_idx_list = []
        self.data_idx_split_points = [0]
        for i in range(self.data1_idx.shape[0]):
            d2 = torch.from_numpy(data2[i].astype(np.float)[:, 1:])  # remove index
            d1 = torch.from_numpy(np.repeat(data1[i].reshape(1, -1), d2.shape[0], axis=0))
            d = torch.cat([d2[:, 0].reshape(-1, 1), d1, d2[:, 1:]], dim=1)  # move similarity to index 0
            data_list.append(d)

            weight = torch.ones(d2.shape[0]) / d2.shape[0]
            weight_list.append(weight)

            idx = torch.from_numpy(np.repeat(self.data1_idx[i].item(), d2.shape[0], axis=0))
            data_idx_list.append(idx)
            self.data_idx_split_points.append(self.data_idx_split_points[-1] + idx.shape[0])
        print("Done")

        self.data = torch.cat(data_list, dim=0)
        self.weights = torch.cat(weight_list, dim=0)
        self.data_idx = torch.cat(data_idx_list, dim=0)

    def __len__(self):
        return self.data1_idx.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start = self.data_idx_split_points[idx]
        end = self.data_idx_split_points[idx+1]

        return self.data[start:end], self.labels[idx], self.weights[start:end], \
               self.data_idx[start:end], self.data1_idx[idx]


class SimScoreModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 10)
        self.fc2 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        out = self.fc1(X)
        # out = self.dropout(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out


class MergeSimModel(SimModel):
    def __init__(self, num_common_features, sim_hidden_sizes=None, merge_mode='sim_model_avg', **kwargs):
        super().__init__(num_common_features, **kwargs)
        assert merge_mode in ['sim_model_avg', 'avg', 'sim_avg', 'common_model_avg']
        self.merge_mode = merge_mode
        self.sim_model = None
        if sim_hidden_sizes is None:
            self.sim_hidden_sizes = [10]
        else:
            self.sim_hidden_sizes = sim_hidden_sizes

        self.data1_shape = None
        self.data2_shape = None

    @staticmethod
    def var_collate_fn(batch):
        data = torch.cat([item[0] for item in batch], dim=0)
        labels = torch.stack([item[1] for item in batch])
        weights = torch.cat([item[2] for item in batch], dim=0)
        idx = torch.cat([item[3] for item in batch], dim=0)
        idx_unique = np.array([item[4] for item in batch], dtype=np.int)
        return data, labels, weights, idx, idx_unique

    def prepare_train_combine(self, data1, data2, labels, data_cache_path=None):
        if data_cache_path and os.path.isfile(data_cache_path):
            print("Loading data from cache")
            with open(data_cache_path, 'rb') as f:
                train_dataset, val_dataset, test_dataset = pickle.load(f)
            print("Done")
        else:
            print("Splitting data")
            train_data1, val_data1, test_data1, train_labels, val_labels, test_labels, train_idx1, val_idx1, test_idx1 = \
                self.split_data(data1, labels, val_rate=self.val_rate, test_rate=self.test_rate)
            if self.dataset_type == 'syn':
                train_data2 = data2[train_idx1]
                val_data2 = data2[val_idx1]
                test_data2 = data2[test_idx1]
            elif self.dataset_type == 'real':
                train_data2 = data2
                val_data2 = data2
                test_data2 = data2
            else:
                assert False, "Not supported dataset type"
            print("Matching training set")
            self.sim_scaler = None  # scaler will fit train_Xs and transform val_Xs, test_Xs
            preserve_key = not self.drop_key
            train_Xs, train_y, train_idx = self.match(train_data1, train_data2, train_labels, idx=train_idx1,
                                                      preserve_key=preserve_key, grid_min=self.grid_min,
                                                      grid_max=self.grid_max, grid_width=self.grid_width,
                                                      knn_k=self.knn_k, kd_tree_leaf_size=self.kd_tree_leaf_size,
                                                      radius=self.kd_tree_radius)
            assert self.sim_scaler is not None
            print("Matching validation set")
            val_Xs, val_y, val_idx = self.match(val_data1, val_data2, val_labels, idx=val_idx1,
                                                preserve_key=preserve_key, grid_min=self.grid_min,
                                                grid_max=self.grid_max, grid_width=self.grid_width, knn_k=self.knn_k,
                                                kd_tree_leaf_size=self.kd_tree_leaf_size,
                                                radius=self.kd_tree_radius)
            assert self.sim_scaler is not None
            print("Matching test set")
            test_Xs, test_y, test_idx = self.match(test_data1, test_data2, test_labels, idx=test_idx1,
                                                   preserve_key=preserve_key, grid_min=self.grid_min,
                                                   grid_max=self.grid_max, grid_width=self.grid_width, knn_k=self.knn_k,
                                                   kd_tree_leaf_size=self.kd_tree_leaf_size,
                                                   radius=self.kd_tree_radius)

            for train_X, val_X, test_X in zip(train_Xs, val_Xs, test_Xs):
                print("Replace NaN with mean value")
                col_mean = np.nanmean(train_X, axis=0)
                train_indices = np.where(np.isnan(train_X))
                train_X[train_indices] = np.take(col_mean, train_indices[1])
                print("Train done.")
                val_indices = np.where(np.isnan(val_X))
                val_X[val_indices] = np.take(col_mean, val_indices[1])
                print("Validation done.")
                test_indices = np.where(np.isnan(test_X))
                test_X[test_indices] = np.take(col_mean, test_indices[1])
                print("Test done.")

            train_dataset = AvgMergeDataset(train_Xs[0], train_Xs[1], train_y, train_idx)
            val_dataset = AvgMergeDatasetV1(val_Xs[0], val_Xs[1], val_y, val_idx)
            test_dataset = AvgMergeDatasetV1(test_Xs[0], test_Xs[1], test_y, test_idx)

            if data_cache_path:
                print("Saving data to cache")
                with open(data_cache_path, 'wb') as f:
                    pickle.dump([train_dataset, val_dataset, test_dataset], f)
                print("Saved")

        return train_dataset, val_dataset, test_dataset

    def merge_pred(self, pred_all: list, idx=None):
        pred_array = np.array(pred_all).T
        weights = pred_array[1] + 1e-6  # prevent zero-division
        avg_pred = np.average(pred_array[0], weights=weights)
        if self.task == 'binary_cls':
            return avg_pred > 0.5
        else:
            assert False, "Not Implemented"

    @staticmethod
    def get_split_points(array, size):
        assert size > 1

        prev = array[0]
        split_points = [0]
        for i in range(1, size):
            if prev != array[i]:
                prev = array[i]
                split_points.append(i)

        split_points.append(size)
        return split_points

    def train_combine(self, data1, data2, labels, data_cache_path=None):
        train_dataset, val_dataset, test_dataset = \
            self.prepare_train_combine(data1, data2, labels, data_cache_path)

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
        # self.sim_model = MLP(input_size=1, hidden_sizes=self.sim_hidden_sizes, output_size=1, activation='sigmoid')
        if self.task == 'binary_cls':
            output_dim = 1
            self.model = MLP(input_size=num_features, hidden_sizes=self.hidden_sizes, output_size=output_dim,
                             activation='sigmoid')
            criterion = nn.BCELoss()
            val_criterion = nn.BCELoss()
        elif self.task == 'multi_cls':
            output_dim = self.n_classes
            self.model = MLP(input_size=num_features, hidden_sizes=self.hidden_sizes, output_size=output_dim,
                             activation=None)
            criterion = nn.CrossEntropyLoss()
            val_criterion = nn.CrossEntropyLoss()
        else:
            assert False, "Unsupported task"
        self.model = self.model.to(self.device)
        if self.merge_mode == 'common_model_avg':
            self.sim_model = SimScoreModel(num_features=self.num_common_features)
            params = list(self.model.parameters()) + list(self.sim_model.parameters())
        elif self.merge_mode == 'sim_model_avg':
            self.sim_model = SimScoreModel(num_features=1)
            params = list(self.model.parameters()) + list(self.sim_model.parameters())
        else:
            self.sim_model = None
            params = list(self.model.parameters())
        self.sim_model = self.sim_model.to(self.device)
        optimizer = torchlars.LARS(optim.Adam(params,
                                              lr=self.learning_rate, weight_decay=self.weight_decay))
        if self.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.sche_factor,
                                                             patience=self.sche_patience,
                                                             threshold=self.sche_threshold)
        else:
            scheduler = None

        best_train_sample_acc = 0
        best_val_sample_acc = 0
        best_test_sample_acc = 0
        best_train_acc = 0
        best_val_acc = 0
        best_test_acc = 0
        print("Start training")
        summary(self.model, torch.zeros([self.train_batch_size, num_features]).to(self.device))
        print(str(self))
        for epoch in range(self.num_epochs):
            # train
            train_loss = 0.0
            train_sample_correct = 0
            train_total_samples = 0
            n_train_batches = 0
            train_pred_all = {}
            self.model.train()
            self.sim_model.train()
            for data_batch, labels, weights, idx1, idx1_unique in tqdm(train_loader, desc="Train"):
                data_batch = data_batch.to(self.device).float()
                labels = labels.to(self.device).float()
                weights = weights.to(self.device).float()

                sim_scores = data_batch[:, 0]
                data = data_batch[:, 1:]

                optimizer.zero_grad()
                outputs = self.model(data)
                if self.merge_mode == 'sim_model_avg':
                    sim_weights = self.sim_model(sim_scores.reshape(-1, 1))
                else:
                    assert False, "Unsupported merge mode"

                outputs_batch = torch.zeros([0, output_dim]).to(self.device)
                idx1_split_points = self.get_split_points(idx1, idx1.shape[0])
                for i in range(idx1_unique.shape[0]):
                    start = idx1_split_points[i]
                    end = idx1_split_points[i+1]
                    output_i = torch.sum(outputs[start:end] * sim_weights[start:end])\
                               / torch.sum(sim_weights[start:end])
                    outputs_batch = torch.cat([outputs_batch, output_i.reshape(-1, 1)], dim=0)
                outputs_batch = outputs_batch.to(self.device)
                outputs_batch[outputs_batch > 1.] = 1.  # bound threshold
                outputs_batch[outputs_batch < 0.] = 0.

                if self.task == 'binary_cls':
                    outputs_batch = outputs_batch.flatten()
                    loss = criterion(outputs_batch, labels)
                    preds = outputs_batch > 0.5
                elif self.task == 'multi_cls':
                    loss = criterion(outputs_batch, labels.long())
                    preds = torch.argmax(outputs_batch, dim=1)
                else:
                    assert False, "Unsupported task"
                n_correct = torch.count_nonzero(preds == labels).item()

                loss.backward()

                optimizer.step()

                train_loss += loss.item()
                train_sample_correct += n_correct
                train_total_samples += idx1_unique.shape[0]
                n_train_batches += 1

            train_loss /= n_train_batches
            train_acc = train_sample_acc = train_sample_correct / train_total_samples

            # validation and test
            val_loss, val_sample_acc, val_acc = self.eval_merge_score(val_dataset, val_criterion, 'Val')
            test_loss, test_sample_acc, test_acc = self.eval_merge_score(test_dataset, val_criterion, 'Test')
            if self.use_scheduler:
                scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_train_acc = train_acc
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_train_sample_acc = train_sample_acc
                best_val_sample_acc = val_sample_acc
                best_test_sample_acc = test_sample_acc
                if self.model_save_path is not None:
                    torch.save(self.model.state_dict(), self.model_save_path)

            print("Epoch {}: {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                  "          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                  "          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                  "          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                  "          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}\n"
                  .format(epoch + 1, "Loss:", train_loss, val_loss, test_loss,
                          "Sample Acc:", train_sample_acc, val_sample_acc, test_sample_acc,
                          "Best Sample Acc:", best_train_sample_acc, best_val_sample_acc, best_test_sample_acc,
                          "Acc:", train_acc, val_acc, test_acc,
                          "Best Acc", best_train_acc, best_val_acc, best_test_acc))

            self.writer.add_scalars('Loss', {'Train': train_loss,
                                             'Validation': val_loss,
                                             'Test': test_loss}, epoch + 1)
            self.writer.add_scalars('Sample Accuracy', {'Train': train_sample_acc,
                                                        'Validation': val_sample_acc,
                                                        'Test': test_sample_acc}, epoch + 1)
            self.writer.add_scalars('Accuracy', {'Train': train_acc,
                                                 'Validation': val_acc,
                                                 'Test': test_acc}, epoch + 1)
        return best_train_sample_acc, best_val_sample_acc, best_test_sample_acc, \
               best_train_acc, best_val_acc, best_test_acc

    def eval_merge_score(self, val_dataset, loss_criterion=None, name='Val'):
        assert self.model is not None, "Model has not been initialized"

        val_loader = DataLoader(val_dataset, batch_size=self.test_batch_size, shuffle=False,
                                num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context,
                                )

        val_idx = val_dataset.data1_idx.detach().cpu().numpy()
        val_labels = val_dataset.data1_labels.detach().cpu().int().numpy()
        answer_all = dict(zip(val_idx, val_labels))
        val_pred_all = {}

        val_loss = 0.0
        val_sample_correct = 0
        val_total_samples = 0
        n_val_batches = 0
        with torch.no_grad():
            self.model.eval()
            self.sim_model.eval()
            for data, labels, weights, idx1, idx2 in tqdm(val_loader, desc=name):

                data = data.to(self.device).float()
                labels = labels.to(self.device).float()
                sim_scores = data[:, 0]
                data = data[:, 1:]

                outputs = self.model(data)
                if self.merge_mode == 'sim_model_avg':
                    sim_weights = self.sim_model(sim_scores.reshape(-1, 1))
                elif self.merge_mode == 'avg':
                    sim_weights = weights
                elif self.merge_mode == 'sim_avg':
                    sim_weights = sim_scores
                elif self.merge_mode == 'common_model_avg':
                    assert self.drop_key is False, "Not common keys to train"
                    common_features = data[:, self.data1_shape[1] - self.num_common_features:
                                              self.data1_shape[1] + self.num_common_features]
                    dists = common_features[:, :self.num_common_features] - common_features[:,
                                                                            self.num_common_features:]
                    sim_weights = self.sim_model(dists)
                else:
                    assert False

                if self.task == 'binary_cls':
                    outputs = outputs.flatten()
                    if loss_criterion is not None:
                        loss = loss_criterion(outputs, labels)
                        val_loss += loss.item()
                    preds = outputs > 0.5
                elif self.task == 'multi_cls':
                    if loss_criterion is not None:
                        loss = loss_criterion(outputs, labels.long())
                        val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                else:
                    assert False, "Unsupported task"
                n_correct = torch.count_nonzero(preds == labels).item()

                val_sample_correct += n_correct
                val_total_samples += data.shape[0]
                n_val_batches += 1

                # calculate final predictions
                sim_weights = sim_weights.detach().cpu().numpy().flatten()
                idx1 = idx1.detach().cpu().numpy()
                idx2 = idx2.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                # noinspection PyUnboundLocalVariable
                for i1, i2, score, out in zip(idx1, idx2, sim_weights, outputs):
                    # noinspection PyUnboundLocalVariable
                    if i1 in val_pred_all:
                        val_pred_all[i1].append((out, score, i2))
                    else:
                        val_pred_all[i1] = [(out, score, i2)]

        val_correct = 0
        # only = 0
        # include = 0
        for i, pred_all in val_pred_all.items():
            pred = self.merge_pred(pred_all, i)
            if answer_all[i] == pred:
                val_correct += 1

            # idx2 = np.array(pred_all).T[-1]
            # if i in idx2:
            #     include += 1
            # if idx2.shape[0] == 0 and idx2.item() == i:
            #     only += 1
        val_acc = val_correct / len(val_pred_all)
        # print("Only {}, include {}".format(only / len(val_pred_all), include / len(val_pred_all)))

        val_sample_acc = val_sample_correct / val_total_samples
        if loss_criterion is not None:
            val_loss /= n_val_batches
        else:
            val_loss = -1.

        return val_loss, val_sample_acc, val_acc
