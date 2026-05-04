import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms

PATIENCE = 5
EPOCHS = 30

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class CNN(nn.Module):
    def __init__(
        self,
        name: str,
        num_conv_layers: int,
        num_filters: int,
        optimizer_class,
        dropout_rate: float,
        in_channels: int = 3,
        kernel_size: int = 3,
        padding: int = 1,
        pooling_kernel_size: int = 2,
        pooling_stride: int = 2,
        lr: float = 0.001,
    ):
        super(CNN, self).__init__()
        self.name = name
        self.patience = PATIENCE
        self.multiclass = True

        layers = []
        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv2d(
                    in_channels, num_filters, kernel_size=kernel_size, padding=padding
                )
            )
            layers.append(nn.ReLU())
            layers.append(
                nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride)
            )
            in_channels = num_filters

        self.feature_extractor = nn.Sequential(*layers)

        spatial_size = 32 // (2**num_conv_layers)
        flattened_size = num_filters * spatial_size * spatial_size

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 200),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(200, 10),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optim = optimizer_class(self.parameters(), lr=lr)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.head(features)
        return logits

    def plot_metric(self, metric_data, metric_name="Erro", stage="train"):
        plt.figure()
        plt.plot(
            metric_data,
            color="blue" if metric_name == "Erro" else "green",
            marker="o",
            markersize=3,
        )
        plt.xlabel("Épocas")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} x Época\n({self.name})")
        plt.tight_layout()
        path = Path("assets/cnn")
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"{self.name}_{stage}_{metric_name.lower()}_plot.png")
        plt.close()

    def fit(
        self,
        train_dataloader: DataLoader,
        epochs: int,
        val_dataloader: Optional[DataLoader] = None,
    ):
        best_loss = float("inf")
        epochs_without_improvement = 0
        train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []

        pbar = tqdm(total=epochs, desc=f"Treinando {self.name}", unit="epoch")
        for _ in range(epochs):
            self.train()
            epoch_loss, correct, total = 0.0, 0, 0

            for x, y in train_dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                output = self.forward(x)
                loss = self.criterion(output, y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                preds = torch.argmax(output, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                epoch_loss += loss.item()

            train_loss_hist.append(epoch_loss / len(train_dataloader))
            train_acc_hist.append(correct / total)

            if val_dataloader is not None:
                val_loss, val_acc = self.evaluate(val_dataloader)
                val_loss_hist.append(val_loss)
                val_acc_hist.append(val_acc)

                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            pbar.update(1)
            if epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {len(train_loss_hist)}")
                break

        pbar.close()
        return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist

    def evaluate(self, data: DataLoader) -> Tuple[float, float]:
        self.eval()
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in data:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = self.forward(x)
                loss += self.criterion(output, y).item()
                preds = torch.argmax(output, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return loss / len(data), correct / total

    def show_predictions(self, test_loader: DataLoader, classes: List[str]):
        """Requisito: Mostrar algumas imagens com as classes preditas."""
        self.eval()
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            outputs = self(images)
            _, preds = torch.max(outputs, 1)

        fig = plt.figure(figsize=(15, 3))
        for idx in range(5):
            ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])
            img = images[idx].cpu().numpy().transpose((1, 2, 0))
            img = img * 0.5 + 0.5
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(
                f"Pred: {classes[preds[idx]]}\nReal: {classes[labels[idx]]}",
                color=("green" if preds[idx] == labels[idx] else "red"),
            )

        path = Path("assets/cnn")
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"best_model_predictions.png")
        plt.close()

    def plot_filters_and_maps(self, image: torch.Tensor):
        """Requisito: Plotar alguns filtros e mapas de ativação de camadas."""
        self.eval()

        first_conv = self.feature_extractor[0]
        filtros = first_conv.weight.data.clone().cpu()

        fig, axes = plt.subplots(1, min(6, filtros.size(0)), figsize=(12, 2))
        for i, ax in enumerate(axes):
            f = filtros[i].numpy().transpose(1, 2, 0)
            f = (f - f.min()) / (f.max() - f.min())
            ax.imshow(f)
            ax.axis("off")
        plt.suptitle(f"Filtros da 1ª Camada - {self.name}")
        fig.savefig(Path("assets/cnn") / f"best_model_filters.png")
        plt.close()

        with torch.no_grad():
            image = image.unsqueeze(0).to(DEVICE)
            mapas_ativacao = first_conv(image).cpu()[0]

        fig, axes = plt.subplots(1, min(6, mapas_ativacao.size(0)), figsize=(12, 2))
        for i, ax in enumerate(axes):
            ax.imshow(mapas_ativacao[i].numpy(), cmap="viridis")
            ax.axis("off")
        plt.suptitle(f"Mapas de Ativação (1ª Camada) - {self.name}")
        fig.savefig(Path("assets/cnn") / f"best_model_feature_maps.png")
        plt.close()


def main(batch_size=64):
    print(f"Usando dispositivo: {DEVICE}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    full_train_data = CIFAR10(
        root="./assets/data", train=True, download=True, transform=transform
    )
    test_data = CIFAR10(
        root="./assets/data", train=False, download=True, transform=transform
    )

    val_size = int(0.1 * len(full_train_data))
    train_size = len(full_train_data) - val_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(
        f"Treino: {len(train_data)} | Validação: {len(val_data)} | Teste: {len(test_data)}"
    )

    opts = [SGD, Adam]
    conv_layers = [2, 3, 4]
    filters = [32, 64, 128]
    dropouts = [0.1, 0.3]

    results = []
    hyperparams = list(product(opts, conv_layers, filters, dropouts))

    print(f"Total de arquiteturas para testar: {len(hyperparams)}")

    # Variáveis para armazenar o melhor modelo
    global_best_acc = 0.0
    best_model_state = None
    best_hyperparams = None

    for opt, n_conv, n_filt, drop in hyperparams:
        model_name = f"{opt.__name__}_Conv{n_conv}_Filt{n_filt}_Drop{drop}"

        m = CNN(
            name=model_name,
            num_conv_layers=n_conv,
            num_filters=n_filt,
            optimizer_class=opt,
            dropout_rate=drop,
        )

        t_loss, t_acc, v_loss, v_acc = m.fit(train_loader, EPOCHS, val_loader)

        m.plot_metric(t_loss, "Erro", "train")
        m.plot_metric(t_acc, "Acuracia", "train")
        m.plot_metric(v_loss, "Erro", "val")
        m.plot_metric(v_acc, "Acuracia", "val")

        test_loss, test_acc = m.evaluate(test_loader)
        print(f"[{model_name}] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")

        # Verifica se é o melhor modelo até agora e salva os pesos
        if test_acc > global_best_acc:
            global_best_acc = test_acc
            best_model_state = m.state_dict()
            best_hyperparams = (model_name, n_conv, n_filt, opt, drop)

        results.append(
            {
                "model": model_name,
                "optimizer": opt.__name__,
                "conv_layers": n_conv,
                "filters": n_filt,
                "dropout": drop,
                "test_loss": round(test_loss, 4),
                "test_acc": round(test_acc, 4),
                "epochs_run": len(t_loss),
            }
        )

    results_df = pd.DataFrame(results)
    out_path = Path("assets/cnn")
    out_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path / "cifar10_results.csv", index=False)
    print(f"\nResultados salvos em {out_path / 'cifar10_results.csv'}")

    if best_hyperparams is None or best_model_state is None:
        print("Nenhum modelo foi treinado com sucesso.")
        return

    print(f"\nGerando visualizações para o melhor modelo: {best_hyperparams[0]}...")

    # Recriar a arquitetura vencedora
    best_m = CNN(
        name=best_hyperparams[0],
        num_conv_layers=best_hyperparams[1],
        num_filters=best_hyperparams[2],
        optimizer_class=best_hyperparams[3],
        dropout_rate=best_hyperparams[4],
    )
    # Carregar os pesos (state_dict) do melhor treino
    best_m.load_state_dict(best_model_state)

    classes_cifar10 = [
        "Avião",
        "Carro",
        "Pássaro",
        "Gato",
        "Cervo",
        "Cachorro",
        "Sapo",
        "Cavalo",
        "Navio",
        "Caminhão",
    ]

    # Gerar imagem com predições reais vs preditas usando o MELHOR modelo
    best_m.show_predictions(test_loader, classes_cifar10)

    # Pegar uma imagem aleatória do teste para plotar mapas usando o MELHOR modelo
    img_teste, _ = next(iter(test_loader))
    best_m.plot_filters_and_maps(img_teste[0])

    print("Imagens de filtros, mapas e predições salvas na pasta assets/cnn!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNNs on CIFAR-10.")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (default: 64)"
    )
    args = parser.parse_args()

    main(args.batch_size)
