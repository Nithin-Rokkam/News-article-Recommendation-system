
"""

This module contains the code for loading and preprocessing data, initializing the dataloaders,
and training the news recommendation model using Matrix Factorization implemented in PyTorch Lightning.

Functions:
    load_data(filepath): Loads and preprocesses the data from the specified file.
    prepare_dataloader(raw_behaviour): Prepares the dataloader from the preprocessed data.
    train_model(dataloader, num_users, num_items, epochs=10): Trains the NewsMF model.
"""

import torch
from torch.utils.data import Dataset, DataLoader

class MindDataset(Dataset):
    # A fairly simple torch dataset module that can take a pandas dataframe (as above),
    # and convert the relevant fields into a dictionary of arrays that can be used in a dataloader
    def __init__(self, df):
        # Create a dictionary of tensors out of the dataframe
        self.data = {
            'userIdx' : torch.tensor(df.userIdx.values),
            'click' : torch.tensor(df.click.values),
            'noclick' : torch.tensor(df.noclick.values)
        }
    def __len__(self):
        return len(self.data['userIdx'])
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}


# Build datasets and dataloaders of train and validation dataframes:

def dataloader(data_train, data_valid):
    bs = 1024
    ds_train = MindDataset(data_train)
    train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True)
    ds_valid = MindDataset(data_valid)
    valid_loader = DataLoader(ds_valid, batch_size=bs, shuffle=False)

    return train_loader, valid_loader

