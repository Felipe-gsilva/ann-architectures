import enum
from pathlib import Path
from typing import List, Tuple
from torch.optim import SGD, Adam
import torch.nn as nn
import pandas as pd
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm

DROPOUT_RATE = 0.1
L2_LAMBDA = 0.001
PATIENCE = 20
EPOCHS = 100


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


class MLP(nn.Module):
    def __init__(
        self,
        name: str,
        layers: List[torch.uint16],
        optimizer=SGD,
        activation_fn: nn.Module = nn.ReLU,
        regularization: RegularizationType = RegularizationType.NONE,
        multiclass: bool = False,
    ) -> None:
        super(MLP, self).__init__()
        self.name = name
        hidden_layers = self.init_hidden_layers(layers, activation_fn, regularization)
        hidden_layers.append(self.init_output_layer(layers[-2], layers[-1]))
        self.layers = nn.ModuleList(hidden_layers)
        self.patience = PATIENCE
        lr = 0.01 if optimizer == SGD else 0.001
        self.optim = (
            optimizer(self.parameters(), lr=lr, weight_decay=L2_LAMBDA)
            if regularization == RegularizationType.L2
            else optimizer(self.parameters(), lr=lr)
        )
        self.criterion = nn.CrossEntropyLoss() if multiclass else nn.BCEWithLogitsLoss()
        self.multiclass = multiclass

    def init_hidden_layers(
        self,
        shape_list: List[torch.uint16],
        activation_fn: nn.Module = nn.ReLU,
        regularization: RegularizationType = RegularizationType.NONE,
    ) -> List[nn.Sequential]:
        """Initializes the hidden layers of the MLP based on the provided shape list, activation function, and regularization type."""
        hidden: List[nn.Sequential] = []
        for i in range(len(shape_list) - 2):
            input_feat = shape_list[i]
            output_feat = shape_list[i + 1]
            hidden.append(
                nn.Sequential(
                    nn.Linear(input_feat, output_feat),
                    activation_fn(),
                    nn.Dropout(DROPOUT_RATE)
                    if regularization == RegularizationType.DROPOUT
                    else nn.Identity(),
                )
            )
        return hidden

    def init_output_layer(self, in_shape, out_shape) -> nn.Sequential:
        """Initializes the output layer of the MLP based on the provided input and output shapes."""
        return nn.Sequential(
            nn.Linear(in_shape, out_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

    def plot_error(self, error, stage="train"):
        """Plots the error over epochs and saves the plot as an image file in the assets/mlp directory."""
        plt.figure()
        plt.plot(error, color="blue", marker="o", markersize=3)
        plt.xlabel("Épocas")
        plt.xlim(0, EPOCHS)
        plt.ylabel("Erro de Classificação")
        plt.ylim(0, max(error) * 1.1)
        plt.title(f"Erro x Época\n({self.name})")
        plt.tight_layout()
        path = Path("assets/mlp")
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"{self.name}_{stage}_error_plot.png")
        plt.close()

    def plot_accuracy(self, accuracy, stage="train"):
        """Plots the accuracy over epochs and saves the plot as an image file in the assets/mlp directory."""
        plt.figure()
        plt.plot(accuracy, color="green", marker="o", markersize=3)
        plt.xlabel("Épocas")
        plt.xlim(0, EPOCHS)
        plt.ylabel("Acurácia")
        plt.ylim(0, 1.0)
        plt.title(f"Acurácia x Época\n({self.name})")
        plt.tight_layout()
        path = Path("assets/mlp")
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"{self.name}_{stage}_accuracy_plot.png")
        plt.close()

    def plot_decision_boundary(self, data: DataLoader, stage: str = "test"):
        """
        Plots the decision boundary for 2-feature datasets.
        Adapted from the Adaline implementation.
        Only works when the input has exactly 2 features.
        """
        all_x = torch.cat([x for x, _ in data]).numpy()
        all_y = torch.cat([y for _, y in data]).numpy()

        if all_x.shape[1] != 2:
            print(
                f"Skipping decision boundary plot for {self.name}: "
                f"Data is {all_x.shape[1]}D (requires 2D)."
            )
            return

        x_min, x_max = all_x[:, 0].min() - 1, all_x[:, 0].max() + 1
        y_min, y_max = all_x[:, 1].min() - 1, all_x[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.01),
            np.arange(y_min, y_max, 0.01),
        )

        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        Z = self.predict(grid).numpy().reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.8, cmap="RdYlBu")
        plt.scatter(
            all_x[:, 0],
            all_x[:, 1],
            c=all_y,
            cmap="RdYlBu",
            edgecolors="k",
            marker="o",
        )
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"Decision Boundary\n({self.name})")

        path = Path("assets/mlp")
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"{self.name}_{stage}_decision_boundary.png")
        plt.close()

    def fit(
        self, train_dataloader: DataLoader, epochs: int = 10, val_df: DataLoader = None
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        nn.Module.train(self)
        best_loss = float("inf")
        epochs_without_improvement = 0
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        pbar = tqdm(total=epochs, desc=f"Training {self.name}", unit="epoch")
        for e in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for x, y in train_dataloader:
                output = self.forward(x)
                y_input = y.float().unsqueeze(1) if not self.multiclass else y
                loss = self.criterion(output, y_input)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.multiclass:
                    preds = torch.argmax(output, dim=1)
                else:
                    preds = (torch.sigmoid(output) >= 0.5).squeeze().long()

                correct += (preds == y).sum().item()
                total += y.size(0)
                epoch_loss += loss.item()

            epoch_loss /= len(train_dataloader)
            epoch_acc = correct / total
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

            if val_df is not None:
                val_loss, val_acc, _ = self.evaluate(val_df)
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)
                nn.Module.train(self)

            pbar.update(1)
            if epochs_without_improvement >= self.patience:
                print(
                    f"Early stopping triggered after {epochs_without_improvement + 1} epochs without improvement."
                )
                break

        pbar.close()
        return train_loss_history, train_acc_history, val_loss_history, val_acc_history

    def evaluate(self, data: DataLoader) -> Tuple[float, float, Tuple]:
        """Returns loss, accuracy and extended metrics (f1, auc, acc)."""
        self.eval()
        all_logits = []
        all_labels = []
        loss = 0.0

        with torch.no_grad():
            for x, y in data:
                output = self.forward(x)
                y_input = y.float().unsqueeze(1) if not self.multiclass else y
                loss += self.criterion(output, y_input).item()
                all_logits.append(output)
                all_labels.append(y)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        f1, auc, acc = get_metrics(all_logits, all_labels, self.multiclass)
        avg_loss = loss / len(data)

        
        return avg_loss, acc, (f1, auc, acc)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            if self.multiclass:
                predicted = torch.argmax(output, dim=1)
            else:
                predicted = (torch.sigmoid(output) >= 0.5).squeeze().long()
        return predicted


def main(path: str | List[str], batch_size=32, test_num=None):
    if isinstance(path, list) and len(path) == 3:
        (train_dataloader, val_dataloader, test_dataloader) = preprocess_know_paths(
            *path, batch_size=batch_size
        )
    elif isinstance(path, str):
        (train_dataloader, val_dataloader, test_dataloader) = preprocess(
            path, batch_size=batch_size
        )
    else:
        print(
            "Invalid path input. Please provide either a single CSV file path or three CSV file paths for train, val, and test."
        )
        return

    activ_fns = [nn.ReLU, nn.Tanh]
    regularization_types = ["none", "dropout", "l2"]
    num_classes = len(train_dataloader.dataset.tensors[1].unique())
    is_multiclass = num_classes > 2
    output_size = num_classes if is_multiclass else 1

    optimizers = [SGD, Adam]

    architectures = [
        [train_dataloader.dataset.tensors[0].shape[1], 64, output_size],
        [train_dataloader.dataset.tensors[0].shape[1], 128, 64, output_size],
        [train_dataloader.dataset.tensors[0].shape[1], 256, 128, 64, output_size],
    ]

    results = []
    hyperparams = product(optimizers, architectures, activ_fns, regularization_types)

    for h in hyperparams:
        model_name = (
            f"Optimizer-{h[0].__name__}_Arch-{len(h[1]) - 2}hidden"
            f"_Act-{h[2].__name__}_Reg-{h[3]}"
        )
        m = MLP(
            model_name,
            optimizer=h[0],
            layers=h[1],
            activation_fn=h[2],
            regularization=RegularizationType(h[3]),
            multiclass=is_multiclass,
        )
        train_loss, train_acc, val_loss, val_acc = m.fit(
            train_dataloader, EPOCHS, val_dataloader
        )
        m.plot_error(train_loss, stage="train")
        m.plot_accuracy(train_acc, stage="train")
        if val_loss:
            m.plot_error(val_loss, stage="val")
            m.plot_accuracy(val_acc, stage="val")

        # decision boundary (only plots if data is 2D)
        m.plot_decision_boundary(test_dataloader, stage="test")

        test_loss, test_acc, (f1, auc, acc) = m.evaluate(test_dataloader)
        print(
            f"Test Loss: {test_loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
        )

        results.append(
            {
                "model": model_name,
                "optimizer": h[0].__name__,
                "architecture": str(h[1]),
                "activation": h[2].__name__,
                "regularization": h[3],
                "test_loss": round(test_loss, 4),
                "test_acc": round(test_acc, 4),
                "test_f1": round(f1, 4),
                "test_auc": round(auc, 4),
                "final_train_loss": round(train_loss[-1], 4),
                "final_train_acc": round(train_acc[-1], 4),
                "final_val_loss": round(val_loss[-1], 4) if val_loss else None,
                "final_val_acc": round(val_acc[-1], 4) if val_acc else None,
            }
        )

    # save comparison table
    results_df = pd.DataFrame(results)
    out_path = Path("assets/mlp")
    out_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path / f"{test_num}_results.csv", index=False)
    print(f"\nResults saved to {out_path / 'results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP on a dataset.")
    parser.add_argument("--path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument(
        "--list_path", nargs=3, help="Paths to the train, val, and test CSV files."
    )
    parser.add_argument(
        "--tests", action="store_true", help="Run predefined tests with known paths."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation.",
    )
    args = parser.parse_args()

    if args.tests:
        test_paths = [
            [
                "assets/train_dataset1.csv",
                "assets/val_dataset1.csv",
                "assets/test_dataset1.csv",
            ],
            [
                "assets/train_dataset2.csv",
                "assets/val_dataset2.csv",
                "assets/test_dataset2.csv",
            ],
            [
                "assets/train_dataset.csv",
                "assets/validation_dataset.csv",
                "assets/test_dataset.csv",
            ],
        ]
        i = 0
        for paths in test_paths:
            print(f"\nRunning test {i + 1} with paths: {paths}")
            main(paths, args.batch_size, i)
            i += 1

        exit(1)

    if args.path is None and args.list_path is None:
        print("Please provide the path to the dataset CSV file using --path.")
        exit(1)

    if (args.path and not args.path.endswith(".csv")) or (
        args.list_path and any(not path.endswith(".csv") for path in args.list_path)
    ):
        print("Please provide a valid CSV file path.")
        exit(1)

    if args.list_path:
        main(args.list_path, args.batch_size)
    else:
        main(args.path, args.batch_size)
