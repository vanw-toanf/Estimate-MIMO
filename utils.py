import torch

def collate_fn(batch):
    """
    Custom collate function for CNN data.
    Input items are already (channels, height, width) tensors from MIMODataset.
    xs: (batch, 2, pilot_length, Nr)
    ys: (batch, 2, Nr, Nt)
    """
    xs = torch.stack([item[0] for item in batch])
    ys = torch.stack([item[1] for item in batch])
    return xs, ys