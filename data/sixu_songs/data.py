import numpy as np
import torch
import pickle

class Dataset():
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.min_val = np.amin([d[-1] for d in self.data])
        self.max_val = np.amax([d[-1] for d in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.Tensor(self.data[idx][1]), torch.Tensor([self.data[idx][2]]) - (self.min_val + self.max_val) / 2)

    def input_size(self):
        return len(self.data[0][1])

class FMADataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir + '/fma.pkl')

class FMAAlignedDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir + '/fma_aligned.pkl')

class MSDDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir + '/msd.pkl')

class MSDAlignedDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir + '/fma_aligned.pkl')

class FMAUnionDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir + '/fma_union.pkl')

class MSDUnionDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir + '/msd_union.pkl')




class CombinedDataset(Dataset):
    def __init__(self, data_path, join_data=True):
        super().__init__(data_path)
        self.join_data = join_data

    def __getitem__(self, idx):
        if self.join_data:
            return (torch.Tensor(self.data[idx][1] + self.data[idx][2]), torch.Tensor([self.data[idx][3]]) - (self.min_val + self.max_val) / 2)
        else:
            return torch.Tensor(self.data[idx][1]), torch.Tensor(self.data[idx][2]), torch.Tensor([self.data[idx][3]]) - (self.min_val + self.max_val) / 2

    def input_size(self):
        return len(self.data[0][1] + self.data[0][2])

    def input_sizes(self):
        return [len(self.data[0][1]), len(self.data[0][2])]

class AlignedDataset(CombinedDataset):
    def __init__(self, data_dir, join_data=True):
        super().__init__(data_dir + '/both.pkl', join_data)

class UnionDataset(CombinedDataset):
    def __init__(self, data_dir, join_data=True):
        super().__init__(data_dir + '/union.pkl', join_data)


