import enum
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from typing import Tuple

def create_ds_from_df(df: pd.DataFrame) -> torch.utils.data.TensorDataset:
    label = df["label"].values
    if any(label < 0):
        label = (label > 0).astype(int)

    return torch.utils.data.TensorDataset(
        torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32),
        torch.tensor(label, dtype=torch.long),
    )


def preprocess(path, batch_size=32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Reads a CSV file and returns its train, validation and test subsets as dataloaders."""
    dataset = pd.read_csv(path)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    train_df = dataset[:train_size]
    val_df = dataset[train_size : train_size + val_size]
    test_df = dataset[train_size + val_size :]

    train_ds = create_ds_from_df(train_df)
    val_ds = create_ds_from_df(val_df)
    test_ds = create_ds_from_df(test_df)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def preprocess_know_paths(
    train_path, val_path, test_path, batch_size=32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Reads 3 given CSV files and returns their respective train, validation and test data as dataloaders."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_ds = create_ds_from_df(train_df)
    val_ds = create_ds_from_df(val_df)
    test_ds = create_ds_from_df(test_df)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    multiclass: bool,
) -> Tuple[float, float, float]:
    """
    Calculates f1 (weighted), roc_auc (weighted) and accuracy.

    Args:
        logits: raw model outputs (before softmax/sigmoid), shape (N, C) or (N, 1)
        labels: ground-truth class indices, shape (N,)
        multiclass: True for CrossEntropy problems, False for binary BCE
    """
    labels_np = labels.cpu().numpy()

    if multiclass:
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        auc = roc_auc_score(labels_np, probs, average="weighted", multi_class="ovr")
    else:
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(labels_np, probs, average="weighted")

    f1 = f1_score(labels_np, preds, average="weighted")
    acc = accuracy_score(labels_np, preds)

    return f1, auc, acc


class RegularizationType(enum.Enum):
    NONE = "none"
    DROPOUT = "dropout"
    L2 = "l2"



