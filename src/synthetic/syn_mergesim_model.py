import os
import sys
import abc
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torchlars
from tqdm import tqdm
from torchsummaryX import summary

from .syn_two_party_model import SimModel
from .model import MLP


class AvgMergeDataset(Dataset):
    def __init__(self, data1, data2, labels, data_idx):
        assert data1.shape[0] == data2.shape[0] == data_idx.shape[0]
        # remove similarity scores in data1 (at column 0)
        data1_labels = np.concatenate([data1[:, 1:], labels.reshape(-1, 1)], axis=1)

        grouped_data1 = {}
        grouped_data2 = {}
        for i in range(data_idx.shape[0]):
            idx1 = data_idx[i]
            if idx1 in grouped_data2:
                grouped_data2[idx1] = np.concatenate([grouped_data2[idx1], data2[i].reshape(1, -1)], axis=0)
            else:
                grouped_data2[idx1] = data2[i].reshape(1, -1)
            grouped_data1[idx1] = data1_labels[i]

        group1_data_idx = np.array(list(grouped_data1.keys()))
        group1_data1_labels = np.array(list(grouped_data1.values()))
        group2_data_idx = np.array(list(grouped_data2.keys()))
        group2_data2 = np.array(list(grouped_data2.values()), dtype='object')

        group1_order = group1_data_idx.argsort()
        group2_order = group2_data_idx.argsort()

        group1_data_idx = group1_data_idx[group1_order]
        group1_data1_labels = group1_data1_labels[group1_order]
        group2_data_idx = group2_data_idx[group2_order]
        group2_data2 = group2_data2[group2_order]
        assert (group1_data_idx == group2_data_idx).all()

        self.data1_idx: torch.Tensor = torch.from_numpy(group1_data_idx)
        self.data1: torch.Tensor = torch.from_numpy(group1_data1_labels[:, :-1])
        self.data1_labels: torch.Tensor = torch.from_numpy(group1_data1_labels[:, -1])
        data2: list = group2_data2

        final_data = []
        final_weights = []
        final_labels = []
        final_idx = []
        for i in range(self.data1.shape[0]):
            d1 = self.data1[i]
            for j in range(data2[i].shape[0]):
                d2 = torch.from_numpy(data2[i][j])
                weight = 1 / data2[i].shape[0]
                d_line = torch.cat([d2[0].reshape(1), d1, d2[1:]], dim=0)

                final_data.append(d_line)
                final_weights.append(weight)
                final_labels.append(self.data1_labels[i])
                final_idx.append(self.data1_idx[i])

        self.data = torch.stack(final_data)
        self.weights = torch.tensor(final_weights)
        self.labels = torch.tensor(final_labels)
        self.data_idx = torch.tensor(final_idx)

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx], self.labels[idx], self.weights[idx], self.data_idx[idx]


class MergeSimModel(SimModel):
    def __init__(self, num_common_features, **kwargs):
        super().__init__(num_common_features, **kwargs)

    @staticmethod
    def var_collate_fn(batch):
        data1 = torch.stack([item[0] for item in batch])
        data2 = [item[1] for item in batch]
        labels = torch.stack([item[2] for item in batch])
        idx = torch.stack([item[3] for item in batch])
        return data1, data2, labels, idx

    def prepare_train_combine(self, data1, data2, labels, data_cache_path=None):
        if data_cache_path and os.path.isfile(data_cache_path):
            print("Loading data from cache")
            with open(data_cache_path, 'rb') as f:
                train_dataset, val_dataset, test_dataset = pickle.load(f)
        else:
            print("Splitting data")
            train_data1, val_data1, test_data1, train_labels, val_labels, test_labels, train_idx1, val_idx1, test_idx1 = \
                self.split_data(data1, labels, val_rate=self.val_rate, test_rate=self.test_rate)
            print("Matching training set")
            self.sim_scaler = None  # scaler will fit train_Xs and transform val_Xs, test_Xs
            train_Xs, train_y, train_idx = self.match(train_data1, data2, train_labels, idx=train_idx1, preserve_key=self.drop_key)
            print("Matching validation set")
            val_Xs, val_y, val_idx = self.match(val_data1, data2, val_labels, idx=val_idx1, preserve_key=self.drop_key)
            print("Matching test set")
            test_Xs, test_y, test_idx = self.match(test_data1, data2, test_labels, idx=test_idx1, preserve_key=self.drop_key)

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

            train_dataset = AvgMergeDataset(train_Xs[0], train_Xs[1], train_y, train_idx[:, 0])
            val_dataset = AvgMergeDataset(val_Xs[0], val_Xs[1], val_y, val_idx[:, 0])
            test_dataset = AvgMergeDataset(test_Xs[0], test_Xs[1], test_y, test_idx[:, 0])

            if data_cache_path:
                print("Saving data to cache")
                with open(data_cache_path, 'wb') as f:
                    pickle.dump([train_dataset, val_dataset, test_dataset], f)

        return train_dataset, val_dataset, test_dataset

    def merge_pred(self, pred_all: list):
        pred_array = np.array(pred_all).T
        # weights = pred_array[1] + 1e-6        # prevent zero-division
        avg_pred = np.average(pred_array[0])
        if self.task == 'binary_cls':
            return avg_pred > 0.5
        else:
            assert False, "Not Implemented"

    def train_combine(self, data1, data2, labels, data_cache_path=None):
        train_dataset, val_dataset, test_dataset = \
            self.prepare_train_combine(data1, data2, labels, data_cache_path)

        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                  num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context)
        num_features = next(iter(train_loader))[0].shape[1] - 1     # remove similarity

        print("Prepare for training")
        if self.task == 'binary_cls':
            model = MLP(input_size=num_features, hidden_sizes=self.hidden_sizes, output_size=1,
                        activation='sigmoid')
            criterion = nn.BCELoss(reduction='none')
            val_criterion = nn.BCELoss()
        elif self.task == 'multi_cls':
            model = MLP(input_size=num_features, hidden_sizes=self.hidden_sizes, output_size=self.n_classes,
                        activation=None)
            criterion = nn.BCELoss(reduction='none')
            val_criterion = nn.CrossEntropyLoss()
        else:
            assert False, "Unsupported task"
        model = model.to(self.device)
        self.model = model
        optimizer = torchlars.LARS(optim.Adam(model.parameters(),
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
        train_pred_all = {}
        train_idx = train_dataset.data1_idx.detach().cpu().numpy()
        train_labels = train_dataset.data1_labels.detach().cpu().int().numpy()
        answer_all = dict(zip(train_idx, train_labels))
        print("Start training")
        summary(self.model, next(iter(train_loader))[0][:, 1:].to(self.device).float())
        print(str(self))
        for epoch in range(self.num_epochs):
            # train
            train_loss = 0.0
            train_sample_correct = 0
            train_total_samples = 0
            n_train_batches = 0
            model.train()
            for data, labels, weights, idx in tqdm(train_loader, desc="Train"):
                weights = weights.to(self.device).float()
                data = data.to(self.device).float()
                labels = labels.to(self.device).float()
                sim_scores = data[:, 0]
                data = data[:, 1:]
                optimizer.zero_grad()

                outputs = model(data)
                if self.task == 'binary_cls':
                    outputs = outputs.flatten()
                    losses = criterion(outputs, labels)
                    preds = outputs > 0.5
                elif self.task == 'multi_cls':
                    losses = criterion(outputs, labels.long())
                    preds = torch.argmax(outputs, dim=1)
                else:
                    assert False, "Unsupported task"
                # noinspection PyTypeChecker
                n_correct = torch.count_nonzero(preds == labels).item()

                # adjust gradients
                # adjusted_grad = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
                # for i in range(losses.shape[0]):
                #     losses[i].backward(retain_graph=True)
                #     for name, param in model.named_parameters():
                #         adjusted_grad[name] += param.grad * weights[i] * sim_scores[0] / losses.shape[0]
                #     model.zero_grad()
                # for name, param in model.named_parameters():
                #     param.grad = adjusted_grad[name]

                loss = torch.mean(losses * weights)
                loss.backward()

                optimizer.step()

                train_loss += torch.mean(losses).item()
                train_sample_correct += n_correct
                train_total_samples += data.shape[0]
                n_train_batches += 1

                # calculate final prediction
                sim_scores = sim_scores.detach().cpu().numpy()
                # noinspection PyUnboundLocalVariable
                idx = idx.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                # noinspection PyUnboundLocalVariable
                for i, score, out in zip(idx, sim_scores, outputs):
                    # noinspection PyUnboundLocalVariable
                    if i in train_pred_all:
                        train_pred_all[i].append((out, score))
                    else:
                        train_pred_all[i] = [(out, score)]

            train_loss /= n_train_batches
            train_sample_acc = train_sample_correct / train_total_samples

            train_correct = 0
            for i, pred_all in train_pred_all.items():
                pred = self.merge_pred(pred_all)
                # noinspection PyUnboundLocalVariable
                if answer_all[i] == pred:
                    train_correct += 1
                assert len(answer_all) == len(train_pred_all)
            train_acc = train_correct / len(train_pred_all)

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
                                num_workers=self.num_workers, multiprocessing_context=self.multiprocess_context)

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
            for data, labels, weights, idx in tqdm(val_loader, desc=name):

                data = data.to(self.device).float()
                labels = labels.to(self.device).float()
                sim_scores = data[:, 0]
                data = data[:, 1:]

                outputs = self.model(data)

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
                sim_scores = sim_scores.detach().cpu().numpy()
                idx = idx.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                # noinspection PyUnboundLocalVariable
                for i, score, out in zip(idx, sim_scores, outputs):
                    # noinspection PyUnboundLocalVariable
                    if i in val_pred_all:
                        val_pred_all[i].append((out, score))
                    else:
                        val_pred_all[i] = [(out, score)]

        val_correct = 0
        for i, pred_all in val_pred_all.items():
            pred = self.merge_pred(pred_all)
            # noinspection PyUnboundLocalVariable
            if answer_all[i] == pred:
                val_correct += 1
        val_acc = val_correct / len(val_pred_all)

        val_sample_acc = val_sample_correct / val_total_samples
        if loss_criterion is not None:
            val_loss /= n_val_batches
        else:
            val_loss = -1.

        return val_loss, val_sample_acc, val_acc