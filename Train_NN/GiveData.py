import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def give_data(df):
    idx = np.random.permutation(len(df))

    num_obs = len(df)
    num_train = int(0.7 * num_obs)
    num_val   = int(0.15 * num_obs)
    train_idx = idx[:num_train]
    val_idx   = idx[num_train:num_train+num_val]
    test_idx  = idx[num_train+num_val:]

    print(df.shape)
    print(df.columns)
    
    num_in = 124  # number of input features
    X = df.iloc[:, :num_in].to_numpy(dtype=np.float32)
    y = df.iloc[:, num_in].to_numpy(dtype=np.float32)  # assuming column num_in+1 is target

    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)
    X = (X - mean) / std

    norm_params = pd.DataFrame(np.vstack([mean, std]), index=["mean", "std"])
    norm_params.to_excel("NormalizationParams.xlsx", sheet_name="Sheet1")

    train_data = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
    val_data   = TensorDataset(torch.tensor(X[val_idx]),   torch.tensor(y[val_idx]))
    test_data  = TensorDataset(torch.tensor(X[test_idx]),  torch.tensor(y[test_idx]))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=64)
    test_loader  = DataLoader(test_data,  batch_size=64)

    return train_loader, val_loader, test_loader, num_in