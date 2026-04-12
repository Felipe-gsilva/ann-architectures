import enum
from typing import List, Self, Tuple
from torch.optim import SGD
import torch.nn as nn
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader


def create_ds_from_df(df: pd.DataFrame) -> torch.utils.data.TensorDataset:
    return torch.utils.data.TensorDataset(
        torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32),
        torch.tensor(df["label"].values, dtype=torch.long),
    )


# Preprocess the dataset by splitting it into train, validation, and test dataloaders.
def preprocess(path, batch_size=32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = pd.read_csv(path)
    # Shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    # Split the dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    train_df = dataset[:train_size]
    val_df = dataset[train_size : train_size + val_size]
    test_df = dataset[train_size + val_size :]

    train_ds = create_ds_from_df(train_df)
    val_ds = create_ds_from_df(val_df)
    test_ds = create_ds_from_df(test_df)

    # Convert the dataframes to PyTorch dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def preprocess_know_paths(
    train_path, val_path, test_path, batch_size=32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # create the datasets
    train_ds = create_ds_from_df(train_df)
    val_ds = create_ds_from_df(val_df)
    test_ds = create_ds_from_df(test_df)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class MLP(nn.Module):
    def __init__(
        self,
        layers: List[torch.uint16],
        LR=0.01,
        multiclass: bool = False,
        patience: int = 5,
        dropout_rate: float = 0.5,
    ) -> None:
        super(MLP, self).__init__()
        hidden_layers = self.init_hidden_layers(layers, multiclass, dropout_rate)
        output_layer = self.init_output_layer(layers[-2], layers[-1])
        hidden_layers.append(output_layer)
        self.layers = nn.ModuleList(hidden_layers)
        # TODO implement early stopping
        self.patience = patience
        self.optim = SGD(self.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss() if multiclass else nn.BCEWithLogitsLoss()

    def init_hidden_layers(
        self,
        shape_list: List[torch.uint16],
        multiclass: bool = False,
        dropout_rate: float = 0.0,
    ) -> List[nn.Sequential]:
        hidden: List[nn.Sequential] = []
        for i in range(len(shape_list) - 2):
            input_feat = shape_list[i]
            output_feat = shape_list[i + 1]
            hidden.append(
                nn.Sequential(
                    nn.Linear(input_feat, output_feat),
                    nn.ReLU() if not multiclass else nn.Tanh(),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                )
            )
        return hidden

    def init_output_layer(self, in_shape, out_shape) -> nn.Sequential:
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

    def fit(
        self, train_dataloader: DataLoader, epochs: int = 10, val_df: DataLoader = None
    ):
        self.train()
        for _ in range(epochs):
            for x, y in train_dataloader:
                output = self.forward(x)
                loss = self.criterion(output, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if val_df is not None:
                self.evaluate(val_df)
                self.train()

    def evaluate(self, data: DataLoader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data:
                output = self.forward(x)
                predicted = torch.argmax(output, dim=1)
                total += y.shape[0]
                correct += (predicted == y).sum().item()
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}")

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

    shape_list = [train_dataloader.dataset.tensors[0].shape[1] - 1, 64, 32, 16, 8, 4]
    epochs = 20
    m = MLP(shape_list, LR=0.01, multiclass=True, patience=5, dropout_rate=0.5)
    m.fit(train_dataloader, epochs, val_dataloader)
    m.evaluate(test_dataloader)


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
