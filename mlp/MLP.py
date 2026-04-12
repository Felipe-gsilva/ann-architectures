import enum
from pathlib import Path
from typing import List, Self, Tuple
from torch.optim import SGD, Adam
import torch.nn as nn
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader
from itertools import product
import matplotlib.pyplot as plt

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

    def train(self, mode: bool = True) -> Self:
        return super().train(mode)

    def eval(self) -> Self:
        return super().eval()

    def plot_error(self, error, stage="train"):
        """Plots the error over epochs and saves the plot as an image file in the assets/mlp directory."""
        plt.figure()
        plt.plot(error, color="blue", marker="o", markersize=3)
        plt.xlabel("Épocas")
        plt.ylabel("Erro de Classificação")
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
        plt.ylabel("Acurácia")
        plt.title(f"Acurácia x Época\n({self.name})")
        plt.tight_layout()
        path = Path("assets/mlp")
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"{self.name}_{stage}_accuracy_plot.png")
        plt.close()

    def fit(
        self, train_dataloader: DataLoader, epochs: int = 10, val_df: DataLoader = None
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        self.train()
        best_loss = float("inf")
        epochs_without_improvement = 0
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        for e in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            for x, y in train_dataloader:
                output = self.forward(x)
                loss = self.criterion(output, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if self.multiclass:
                    acc = (torch.argmax(output, dim=1) == y).float().mean().item()
                else:
                    acc = ((output > 0.5).squeeze() == y).float().mean().item()

                epoch_loss += loss.item()
                epoch_acc += acc

            epoch_loss /= len(train_dataloader)
            epoch_acc /= len(train_dataloader)
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

            if val_df is not None:
                val_loss, acc = self.evaluate(val_df)
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                val_loss_history.append(val_loss)
                val_acc_history.append(acc)
                self.train()

            print(f"Epoch {e + 1}/{epochs} completed.")
            if val_df is not None and epochs_without_improvement >= self.patience:
                print(
                    f"Early stopping triggered after {epochs_without_improvement + 1} epochs without improvement."
                )
                break

        return train_loss_history, train_acc_history, val_loss_history, val_acc_history

    def evaluate(self, data: DataLoader) -> Tuple[float, float]:
        """Returns Vaidation Loss and Accuracy"""
        self.eval()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for x, y in data:
                output = self.forward(x)
                predicted = torch.argmax(output, dim=1)
                total += y.shape[0]
                correct += (predicted == y).sum().item()
                loss += self.criterion(output, y).item()
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}")
        return loss / len(data), accuracy

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            predicted = torch.argmax(output, dim=1)
        return predicted


def main(path):
    if isinstance(path, list) and len(path) == 3:
        (train_dataloader, val_dataloader, test_dataloader) = preprocess_know_paths(
            *path
        )
    elif isinstance(path, str):
        (train_dataloader, val_dataloader, test_dataloader) = preprocess(path)
    else:
        print(
            "Invalid path input. Please provide either a single CSV file path or three CSV file paths for train, val, and test."
        )
        return
    optimizers = [SGD, Adam]

    architectures = [
        [
            train_dataloader.dataset.tensors[0].shape[1],
            64,
            len(train_dataloader.dataset.tensors[1].unique()),
        ],
        [
            train_dataloader.dataset.tensors[0].shape[1],
            128,
            64,
            len(train_dataloader.dataset.tensors[1].unique()),
        ],
        [
            train_dataloader.dataset.tensors[0].shape[1],
            256,
            128,
            64,
            len(train_dataloader.dataset.tensors[1].unique()),
        ],
    ]

    activ_fns = [nn.ReLU, nn.Tanh]
    regularization_types = ["none", "dropout", "l2"]

    hyperparams = product(
        optimizers,
        architectures,
        activ_fns,
        regularization_types,
    )
    for h in hyperparams:
        m = MLP(
            f"Optimizer: {h[0].__name__}, Architecture: {h[1]}, Activation: {h[2].__name__}, Regularization: {h[3]}",
            optimizer=h[0],
            layers=h[1],
            activation_fn=h[2],
            regularization=RegularizationType(h[3]),
            multiclass=len(train_dataloader.dataset.tensors[1].unique()) > 2,
        )
        train_loss, train_acc, val_loss, val_acc = m.fit(
            train_dataloader, EPOCHS, val_dataloader
        )
        m.plot_error(train_loss, stage="train")
        m.plot_accuracy(train_acc, stage="train")
        if val_loss:
            m.plot_error(val_loss, stage="val")
            m.plot_accuracy(val_acc, stage="val")

        test_loss, test_acc = m.evaluate(test_dataloader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP on a dataset.")
    parser.add_argument("--path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument(
        "--list_path", nargs=3, help="Paths to the train, val, and test CSV files."
    )
    args = parser.parse_args()
    if args.path is None and args.list_path is None:
        print("Please provide the path to the dataset CSV file using --path.")
        exit(1)

    if (args.path and not args.path.endswith(".csv")) or (
        args.list_path and any(not path.endswith(".csv") for path in args.list_path)
    ):
        print("Please provide a valid CSV file path.")
        exit(1)

    if args.list_path:
        main(args.list_path)
    else:
        main(args.path)
