import torch
import numpy as np

def one_hot(data, n_cols):
    data_onehot = np.zeros((data.shape[0], n_cols))
    data_onehot[range(data.shape[0]), data] = 1.
    return data_onehot

def to_tensor(torch_type, device):
    def tensor(data):
        return torch.tensor(data, dtype=torch_type).to(device)
    return tensor
