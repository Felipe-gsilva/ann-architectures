import enum
from numpy.typing import NDArray
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class WeightsInitializationType(enum.Enum):
    RANDOM = "random"
    XAVIER = "xavier"
    HE = "he"


class SimpleModel:
    """Base class for simple machine learning models, providing common functionality for weight initialization, data loading, and plotting."""

    weights: NDArray
    activation_function: function

    def __init__(self, name):
        self.name = name

    def init_weights(self, type: WeightsInitializationType, size: int, seed: int = 42):
        """Initialize model weights with random values."""
        if type == WeightsInitializationType.RANDOM:
            self.weights = np.random.default_rng(seed).random(size)
        else:
            print(f"Unsupported weights initialization type: {type}")
            exit(1)

    def load_df(self, data_path: Path) -> pd.DataFrame:
        """Load data from a CSV file, add a bias column if it doesn't exist, and return the DataFrame."""
        try:
            if not data_path.exists():
                raise FileNotFoundError(f"Data path does not exist: {data_path}")

            df = pd.read_csv(data_path)
            if "bias" not in df.columns:
                df.insert(0, "bias", 1)

            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            exit(1)

    def plot_decision_boundary(self, x, labels, step: str = "train"):
        assert "bias" in x.columns, (
            "DataFrame must contain a 'bias' column for plotting decision boundary."
        )
        assert len(x.columns) == len(self.weights), (
            "Number of features (including bias) must match the number of weights."
        )
        assert self.weights is not None, (
            "Model weights must be initialized before plotting decision boundary."
        )

        if len(self.weights) > 3:
            print(
                f"Skipping decision boundary plot for {self.name}: Data is {len(self.weights) - 1}D."
            )
            return

        x_min, x_max = x.iloc[:, 1].min() - 1, x.iloc[:, 1].max() + 1
        y_min, y_max = x.iloc[:, 2].min() - 1, x.iloc[:, 2].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
        )

        grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
        Z = np.where(np.dot(grid, self.weights) >= 0, 1, -1)
        Z = Z.reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.8, cmap="RdYlBu")
        plt.scatter(
            x.iloc[:, 1],
            x.iloc[:, 2],
            c=labels,
            cmap="RdYlBu",
            edgecolors="k",
            marker="o",
        )
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"Decision Boundary of {self.name}")

        Path("../assets/perceptron").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"../assets/perceptron/{step}_decision_boundary_{self.name}.png")
        plt.close()

    def plot_error(self, classification_error):
        assert len(classification_error) > 0, (
            "Classification error list cannot be empty for plotting."
        )
        plt.figure()
        plt.plot(classification_error, color="blue", marker="o", markersize=3)
        plt.xlabel("Épocas")
        plt.ylabel("Erro de Classificação")
        plt.title(f"Erro de Classificação x Época\n({self.name})")
        plt.tight_layout()
        plt.savefig(f"../assets/perceptron/erro_{self.name}.png")
        plt.close()
