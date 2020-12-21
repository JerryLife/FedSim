import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummaryX import summary

from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import torch_optimizer as optim


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


seed = 0
set_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='exact_aligned_game_unmatch', type=str, help='Training dataset ')
parser.add_argument('--setting', default='two', type=str, help='Training setting')
parser.add_argument('--batch-size', default=1024 * 16, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-2, type=float, help='Starting learning rate')
parser.add_argument('--epochs', default=30, type=int, help='Number of epochs to run for')
parser.add_argument('--weight-decay', default=1e-3, type=float, help='Weight decay of optimizer')
args = parser.parse_args()
for arg in vars(args):
    print("{}: {}".format(arg, getattr(args, arg)))

root = "data/"
os.chdir(sys.path[0] + "/../")  # change working directory

dataset = args.dataset
setting = args.setting
print(dataset, setting)

trn_df = pd.read_csv(root + dataset + '_train.csv')
vld_df = pd.read_csv(root + dataset + '_val.csv')
tst_df = pd.read_csv(root + dataset + '_test.csv')
print(trn_df.shape, vld_df.shape, tst_df.shape)

def std_scale_col(column):
    scaler = StandardScaler()
    trn_df[[column]] = scaler.fit_transform(trn_df[[column]])
    vld_df[[column]] = scaler.transform(vld_df[[column]])
    tst_df[[column]] = scaler.transform(tst_df[[column]])

std_scale_col('price')

trn_df = trn_df.drop(columns=['title'])
vld_df = vld_df.drop(columns=['title'])
tst_df = tst_df.drop(columns=['title'])


def fillna_col(column: str, fill: str):
    if fill == 'mode':
        trn_df[column] = trn_df[column].fillna(trn_df[column].mode()[0])
        vld_df[column] = vld_df[column].fillna(trn_df[column].mode()[0])
        tst_df[column] = tst_df[column].fillna(trn_df[column].mode()[0])
    elif fill == 'mean':
        trn_df[column] = trn_df[column].fillna(trn_df[column].mean())
        vld_df[column] = vld_df[column].fillna(trn_df[column].mean())
        tst_df[column] = tst_df[column].fillna(trn_df[column].mean())
    else:
        assert False

cols = ['appid', 'steamid', 'type', 'release_year', 'required_age', 'is_multiplayer']
if setting == 'two':
    if dataset == 'exact_aligned_game_unmatch':
        fillna_col('score_phrase', fill='mode')
        fillna_col('editors_choice', fill='mode')
        fillna_col('score', fill='mean')

    std_scale_col('score')
    fillna_col('genre', fill='mode')
    cols += ['score_phrase', 'genre', 'editors_choice']
elif setting == 'one':
    trn_df = trn_df.drop(columns=['score_phrase', 'score', 'genre', 'editors_choice'])
    vld_df = vld_df.drop(columns=['score_phrase', 'score', 'genre', 'editors_choice'])
    tst_df = tst_df.drop(columns=['score_phrase', 'score', 'genre', 'editors_choice'])

# category to int
counts = []
for col in cols:
    cats = sorted(trn_df[col].unique().tolist())
    cat2i = {cat: i for i, cat in enumerate(cats)}
    counts.append(len(cat2i))
    trn_df[col] = trn_df[col].transform(lambda cat: cat2i[cat])
    vld_df[col] = vld_df[col].transform(lambda cat: cat2i[cat])
    tst_df[col] = tst_df[col].transform(lambda cat: cat2i[cat])
print("Counts: {}".format(counts))

batch_size = args.batch_size

if setting == 'two':
    x_trn = [trn_df.appid, trn_df.steamid, trn_df.type, trn_df.release_year, trn_df.required_age,
             trn_df.is_multiplayer, trn_df.score_phrase, trn_df.genre, trn_df.editors_choice,
             trn_df[['price']].astype('float32'), trn_df[['score']].astype('float32')]
    x_vld = [vld_df.appid, vld_df.steamid, vld_df.type, vld_df.release_year, vld_df.required_age,
             vld_df.is_multiplayer, vld_df.score_phrase, vld_df.genre, vld_df.editors_choice,
             vld_df[['price']].astype('float32'), vld_df[['score']].astype('float32')]
    x_tst = [tst_df.appid, tst_df.steamid, tst_df.type, tst_df.release_year, tst_df.required_age,
             tst_df.is_multiplayer, tst_df.score_phrase, tst_df.genre, tst_df.editors_choice,
             tst_df[['price']].astype('float32'), tst_df[['score']].astype('float32')]
elif setting == 'one':
    x_trn = [trn_df.appid, trn_df.steamid, trn_df.type, trn_df.release_year,
             trn_df.required_age, trn_df.is_multiplayer, trn_df[['price']].astype('float32')]
    x_vld = [vld_df.appid, vld_df.steamid, vld_df.type, vld_df.release_year,
             vld_df.required_age, vld_df.is_multiplayer, vld_df[['price']].astype('float32')]
    x_tst = [tst_df.appid, tst_df.steamid, tst_df.type, tst_df.release_year,
             tst_df.required_age, tst_df.is_multiplayer, tst_df[['price']].astype('float32')]
else:
    assert False

x_trn = [torch.from_numpy(col.to_numpy()) for col in x_trn]
x_vld = [torch.from_numpy(col.to_numpy()) for col in x_vld]
x_tst = [torch.from_numpy(col.to_numpy()) for col in x_tst]
y_trn = torch.from_numpy(trn_df.label.astype('float32').to_numpy())
y_vld = torch.from_numpy(vld_df.label.astype('float32').to_numpy())
y_tst = torch.from_numpy(tst_df.label.astype('float32').to_numpy())

trn_set = torch.utils.data.TensorDataset(*x_trn, y_trn)
vld_set = torch.utils.data.TensorDataset(*x_vld, y_vld)
tst_set = torch.utils.data.TensorDataset(*x_tst, y_tst)
print(len(trn_set), len(vld_set), len(tst_set))

trn_loader = torch.utils.data.DataLoader(trn_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
vld_loader = torch.utils.data.DataLoader(vld_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
tst_loader = torch.utils.data.DataLoader(tst_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
print(len(trn_loader), len(vld_loader), len(tst_loader))


class DLRM(nn.Module):
    def __init__(self, top_mlp_units, dense_mlp_units, cat_mlp_units, emb_dim, counts, denses):
        super().__init__()
        num_fea = len(counts) + len(denses)
        self.num_cat = len(counts)
        self.num_dense = len(denses)

        embs = [nn.Embedding(cnt, emb_dim) for cnt in counts]
        self.embs = nn.ModuleList(embs)

        #         cat_mlps = []
        #         for _ in range(self.num_cat):
        #             mlp = []
        #             prev = emb_dim
        #             for unit in cat_mlp_units:
        #                 mlp.append(nn.Linear(prev, unit))
        #                 mlp.append(nn.LeakyReLU())
        #                 prev = unit
        #             mlp.append(nn.Linear(prev, emb_dim))
        #             mlp.append(nn.LeakyReLU())
        #             cat_mlps.append(nn.Sequential(*mlp))
        #         self.cat_mlps = nn.ModuleList(cat_mlps)

        dense_mlps = []
        for d in denses:
            mlp = []
            prev = d
            for unit in dense_mlp_units:
                mlp.append(nn.Linear(prev, unit))
                mlp.append(nn.LeakyReLU())
                prev = unit
            mlp.append(nn.Linear(prev, emb_dim))
            mlp.append(nn.LeakyReLU())
            dense_mlps.append(nn.Sequential(*mlp))
        self.dense_mlps = nn.ModuleList(dense_mlps)

        top_mlp = []
        # prev =
        prev = emb_dim * self.num_dense + int(num_fea * (num_fea - 1) / 2)
        for unit in top_mlp_units:
            top_mlp.append(nn.Linear(prev, unit))
            top_mlp.append(nn.LeakyReLU())
            prev = unit
        top_mlp.append(nn.Dropout(0.5))
        top_mlp.append(nn.Linear(prev, 1))
        top_mlp.append(nn.Sigmoid())
        self.top_mlp = nn.Sequential(*top_mlp)

    def forward(self, inputs):
        cat_embs = []
        dense_embs = []

        for i in range(self.num_cat):
            emb = self.embs[i](inputs[i])
            # emb = self.cat_mlps[i](emb)
            cat_embs.append(emb)

        for i in range(self.num_dense):
            emb = self.dense_mlps[i](inputs[self.num_cat + i])
            dense_embs.append(emb)

        # out = torch.cat(cat_embs + dense_embs, dim=1)
        out = self.interact_features(dense_embs, cat_embs)
        out = self.top_mlp(out)
        out = torch.flatten(out)

        return out

    def interact_features(self, x, ly):
        # concatenate dense and sparse features
        (batch_size, d) = x[0].shape
        T = torch.cat(x + ly, dim=1).view((batch_size, -1, d))

        # perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        _, ni, nj = Z.shape
        offset = 0
        li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
        lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
        Zflat = Z[:, li, lj]

        # concatenate dense features and interactions
        R = torch.cat(x + [Zflat], dim=1)
        return R

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.05, 0.05)


# top_mlp_units = [256, 128, 64]
# dense_mlp_units = [32]
top_mlp_units = [512, 256, 64]
dense_mlp_units = [64, 32]
cat_mlp_units = []
emb_dim = 16
if setting == 'two':
    denses = [1, 1]
elif setting == 'one':
    denses = [1]
lr = args.lr

model = DLRM(top_mlp_units, dense_mlp_units, cat_mlp_units, emb_dim, counts, denses)
summary(model, next(iter(trn_loader))[:-1])

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Lamb(model.parameters(), lr=lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=10, verbose=True, threshold=1e-4)

writer = SummaryWriter("runs/run_{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), flush_secs=60)

def train():
    model.train()
    trn_loss = 0.0
    trn_total = 0
    trn_correct = 0
    for data in trn_loader:
        inputs = [x.to(device) for x in data[:-1]]
        labels = data[-1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        cnt = inputs[0].size(0)
        trn_total += cnt
        trn_loss += loss.item() * cnt

        predicted = (outputs.detach().cpu().numpy() > 0.5)
        trn_correct += np.sum(labels.detach().cpu().numpy() == predicted)

    trn_loss /= trn_total
    return trn_loss, 100 * trn_correct / trn_total


# min_loss = float('inf')
# wait = 0

def test(data_loader, status):
    model.eval()
    vld_loss = 0.0
    vld_total = 0
    vld_correct = 0
    with torch.no_grad():
        for data in data_loader:
            inputs = [x.to(device) for x in data[:-1]]
            labels = data[-1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            cnt = inputs[0].size(0)
            vld_total += cnt
            vld_loss += loss.item() * cnt

            predicted = (outputs.detach().cpu().numpy() > 0.5)
            vld_correct += np.sum(labels.detach().cpu().numpy() == predicted)

    vld_loss /= vld_total
    if status == 'vld':
        scheduler.step(vld_loss)

    #         global min_loss
    #         global wait
    #         if min_loss - vld_loss >= 1e-4:
    #             min_loss = vld_loss
    #             wait = 0
    #         else:
    #             wait += 1
    #             print(wait)
    return vld_loss, 100 * vld_correct / vld_total


loss = []
acc = []
epochs = args.epochs
best_vld_acc = 0.0
final_tst_acc = 0.0
final_trn_acc = 0.0
for epoch in range(epochs):
    start_t = time.time()
    trn_loss, trn_acc = train()
    vld_loss, vld_acc = test(vld_loader, 'vld')
    tst_loss, tst_acc = test(tst_loader, 'tst')
    end_t = time.time()
    print('Epoch %d trn_loss: %.4f trn_acc: %.2f vld_loss: %.4f vld_acc: %.2f tst_loss: %.4f tst_acc: %.2f Time: %d s' %
          (epoch, trn_loss, trn_acc, vld_loss, vld_acc, tst_loss, tst_acc, end_t - start_t))
    writer.add_scalars('Loss', {
            'Train': trn_loss,
            'Validation': vld_loss,
            'Test': tst_loss
        }, epoch)
    writer.add_scalars('Accuracy', {
        'Train': trn_acc,
        'Validation': vld_acc,
        'Test': tst_acc
    }, epoch)
    loss.append(tst_loss)
    acc.append(tst_acc)
    if vld_acc > best_vld_acc:
        best_vld_acc = vld_acc
        final_tst_acc = tst_acc
        final_trn_acc = trn_acc
print("----------------------------------------------------------")
print("Final train accuracy={}".format(final_trn_acc))
print("Final validation accuracy={}".format(best_vld_acc))
print("Final test accuracy={}".format(final_tst_acc))
