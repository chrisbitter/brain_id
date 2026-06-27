import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset



class EEGDataset:
    """Class to handle EEG dataset loading """

    def __init__(self, data_dir, subjects, test_split=0.2):
        self.data_dir = data_dir
        self.subjects = subjects
        X, y = load_data(data_dir, subjects)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42
        )
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    
    def get_dataset(self, time_window=1.0, sfreq=250, overlap=0.1, balance:float=None):
        """Return the dataset."""

        X_train, y_train = create_dataset(self.X_train, self.y_train, 
                                           time_window, sfreq, overlap)
        X_test, y_test = create_dataset(self.X_test, self.y_test, 
                                           time_window, sfreq, overlap)

        if balance is not None:
            X_train, y_train = balance_classes(X_train, y_train, balance)

        return X_train, y_train, X_test, y_test

    def get_torch_dataset(self, time_window=1.0, sfreq=250, overlap=0.1, balance:float=None):
        """Return the dataset as a PyTorch dataset."""
        
        X_train, y_train, X_test, y_test = self.get_dataset(
            time_window, sfreq, overlap, balance
        )

        train_dataset = EEGDatasetTorch(X_train, y_train)
        test_dataset = EEGDatasetTorch(X_test, y_test)

        return train_dataset, test_dataset



class EEGDatasetTorch(Dataset):

    """ Class to handle EEG dataset loading """

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y


def load_data(data_dir, subjects):
    """ Load the data from the specified path and subjects. """
    
    df = None
    for sub in subjects:
        sub_dir = os.path.join(data_dir, sub)

        for run_file in sorted(os.listdir(sub_dir)):
            if not run_file.endswith(".csv"):
                continue

            run_path = os.path.join(sub_dir, run_file)
            print(f"Loading {run_path}...")

            # Load and append the data
            if df is None:
                df = pd.read_csv(run_path)
            else:
                df = pd.concat([df, pd.read_csv(run_path)], ignore_index=True)
            
    # select channels and stimulus
    ch_names = [f'ch{i}' for i in range(1, 9)]
    X = df[ch_names].to_numpy()
    y = df['stimulus'].to_numpy()

    # encode labels
    y = LabelEncoder().fit_transform(y)

    # sanity checks
    assert X.shape[1] == 8, "X must have 8 channels"
    assert len(X) > 0, "X must not be empty"
    assert len(X) == len(y), "X and y must have the same length"

    return X, y


def create_dataset(X, y, time_window=1.0, sfreq=250, overlap=0.1):
    """Create a dataset from the given data.

    Args:
        X (np.ndarray): sample data
        y (np.ndarray): target data
        time_window (float, optional): length of each time window in seconds. Defaults to 1.0.
        sfreq (int, optional): sampling frequency of the data in Hz. Defaults to 250.
        overlap (float, optional): fraction of overlap between consecutive windows. Defaults to 0.1.
    
    Returns:
        np.ndarray: X segments
        np.ndarray: y segments
    """
    assert 0 <= overlap < 1, "overlap must be between 0 and 1"

    n_samples = X.shape[0]
    time_steps = int(sfreq * time_window)
    step_size = int(time_steps * (1 - overlap))

    X_windows = []
    y_windows = []

    for start in range(0, n_samples - time_steps + 1, step_size):
        end = start + time_steps
        
        y_window = y[start:end]
        unique_classes = np.unique(y_window[y_window != 0])
        y_win_cls = unique_classes[0] if len(unique_classes) > 0 else 0

        X_windows.append(X[start:end])
        y_windows.append(y_win_cls)

    return np.array(X_windows), np.array(y_windows)


def balance_classes(X, y, max_ratio=3.0):
    """Balance the classes in the dataset by removing samples from the larger classes.

    Args:
        X (np.ndarray): sample data
        y (np.ndarray): target data
        max_ratio (float, optional): maximum ratio of min and max class size. Defaults to 3.0.
    
    Returns:
        np.ndarray: balanced X
        np.ndarray: balanced y
    """
    assert max_ratio >= 1, "max_ratio must be greater than or equal to 1"

    # count the number of samples in each class
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = min(counts)
    max_count = min_count * max_ratio

    # for each class remove samples until the ratio is met
    rm_idxs = []
    for cls, cls_count in zip(unique_classes, counts):
        if cls_count > max_count:
            # remove samples from this class
            cls_idxs = np.where(y == cls)[0]
            np.random.shuffle(cls_idxs)
            rm_idxs.extend(cls_idxs[:- int(max_count)])

    X = np.delete(X, rm_idxs, axis=0)
    y = np.delete(y, rm_idxs, axis=0)

    # sanity checks
    min_count = min(np.unique(y, return_counts=True)[1])
    max_count = max(np.unique(y, return_counts=True)[1])
    assert max_count / min_count <= max_ratio, f"max ratio of {max_ratio} exceeded: {max_count / min_count}"
    assert len(np.unique(y)) == len(unique_classes), "y must have the same number of classes as before"

    return X, y
