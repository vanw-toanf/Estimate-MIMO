import os
import shutil
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CNN_MIMO
from dataset import MIMODataset
from torch.utils.data import random_split, DataLoader
from pprint import pprint
import argparse
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from utils import collate_fn


def get_arg():
    parse = argparse.ArgumentParser(description='Train MIMO model')
    parse.add_argument('-b', '--batch_size', type=int, default=128)
    parse.add_argument('-e', '--epochs', type=int, default=100)
    parse.add_argument('-l', '--lr', type=float, default=1e-3)
    parse.add_argument('-c', '--checkpoint_path', type=str, default=None)
    parse.add_argument('-t', '--tensorboard_path', type=str, default="tensorboard")
    parse.add_argument('-r', '--train_path', type=str, default="weights")
    args = parse.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset = MIMODataset(num_samples=10000, snr_db=10, Nt=2, Nr=2)
    dataset = MIMODataset(num_samples=10000, Nt=2, Nr=2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CNN_MIMO(Nt=2, Nr=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)


    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
    else:
        start_epoch = 0
        best_val_loss = 0

    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)
    if not os.path.isdir(args.train_path):
        os.mkdir(args.train_path)
    writer = SummaryWriter(args.tensorboard_path)

    num_iters = len(train_loader)
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        # TRAIN
        model.train()
        losses = []
        progress_bar = tqdm(train_loader, colour="cyan")

        for iter, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)

            # Forward pass
            loss = criterion(model(x), y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()

            progress_bar.set_description("Epoch {}/{}. Loss value: {:.4f}".format(epoch + 1, args.epochs, loss_value))
            losses.append(loss_value)
            writer.add_scalar("Train/Loss", np.mean(losses), iter + epoch * num_iters)

        # VALIDATION
        model.eval()
        losses = []
        with torch.no_grad():
            for iter, (x_val, y_val) in enumerate(val_loader):
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_pred = model(x_val)
                loss = criterion(y_pred, y_val)
                losses.append(loss.item())

        val_loss = np.mean(losses)
        writer.add_scalar("Val/Loss", val_loss, epoch)

        # Save the model
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, os.path.join(args.train_path, "last.pt"))

        if val_loss < best_val_loss:
            torch.save(checkpoint, os.path.join(args.train_path, "best.pt"))
            best_val_loss = val_loss

        scheduler.step()


if __name__ == "__main__":
    args = get_arg()
    train(args)