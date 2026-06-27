import json
import os
import torch
from torch.utils.data import DataLoader
from braindecode.models import EEGNetv4
import datetime
import argparse

from train.dataset import EEGDataset
from torch.utils.data import random_split


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/datasets/csv/",
        help="Path to the dataset directory.",
    )
    parser.add_argument(  # accept list of subject sub-01 sub-02 ...
        "--subjects",
        type=str,
        nargs="+",
        default=["sub-01"],
        help="List of subject (e.g., sub-01 sub-02 ...).",
    )
    parser.add_argument(
        "--n_times",
        type=int,
        default=1,
        help="Number of times to run the training.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="Number of epochs to train the model.",
    )
    return parser.parse_args()


def batch_to_device(batch, device):
    """Move the batch to the specified device."""
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    return x, y


def test(model, loader, loss_fn, device):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch_to_device(batch, device)
            y_pred = model(x)
            loss +=  loss_fn(y_pred, y).item()

            # TODO: compute 
            
    loss /= len(loader)
    print(f"Test loss: {loss:.4f}")
    return loss


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    for batch in loader:
        x, y = batch_to_device(batch, device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()


def main_train_loop(model, n_epochs, train_loader, test_loader, loss_fn, optimizer, device):
    for epoch in range(n_epochs):
        train(model, train_loader, loss_fn, optimizer, device)
        test(model, test_loader, loss_fn, device)

    return model


def evaluate(model, test_loader, loss_fn, device):
    pass


def main(data_path, n_times, batch_size=32):

    # parse command line arguments
    args = parse_args()

    # create the dataset
    ds = EEGDataset(args.data_path, args.subjects, test_split=0.2)
    train_ds, test_ds = ds.get_torch_dataset(
        time_window=0.5, overlap=0.50, balance=1.0
    )

    # split train_ds into train and val sets
    val_ratio = 0.2
    train_len = int((1 - val_ratio) * len(train_ds))
    val_len = len(train_ds) - train_len
    train_ds, val_ds = random_split(train_ds, [train_len, val_len])

    # create the dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # train on raw data
    x = train_ds[0][0]
    n_channels, n_times = x.shape

    # create the model

    model_kwargs = {
        "n_chans": n_channels,
        "n_classes": 2,
        "n_times": n_times,
    }
    model = EEGNetv4(**model_kwargs)

    # define the loss function and optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save results
    class_name = model.__class__.__name__
    ckpt_dir = "checkpoints/train_nn/"
    model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id += f"_{class_name}"
    model_dir = os.path.join(ckpt_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # train the model
    model = main_train_loop(model, args.n_epochs, train_loader, val_loader, loss_fn, optimizer, device)

    # save the model
    print(f"Saving model to {model_dir}...")
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
    
    # save model kwargs
    model_kwargs_path = os.path.join(model_dir, "model_kwargs.json")
    with open(model_kwargs_path, "w") as f:
        json.dump(model_kwargs, f)

    # evaluate the model
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
    )
    evaluate(model, test_loader, loss_fn, device)



if __name__ == "__main__":
   main()


