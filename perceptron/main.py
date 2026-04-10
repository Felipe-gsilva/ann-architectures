import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.typing import NDArray
from itertools import product


class Perceptron:
    weights: NDArray
    name: str

    def __init__(self, name: str = "Perceptron"):
        self.name = name

    def preprocess(self, data_path: Path) -> pd.DataFrame:
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

    def sign(self, u):
        return 1 if u >= np.float32(0) else -1

    def get_delta(self, learning_rate, error, x):
        return learning_rate * error * x

    def plot_decision_boundary(self, x, labels, step: str = "train"):
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
        plt.figure()
        plt.plot(classification_error, color="blue", marker="o", markersize=3)
        plt.xlabel("Épocas")
        plt.ylabel("Erro de Classificação")
        plt.title(f"Erro de Classificação x Época\n({self.name})")
        plt.tight_layout()
        plt.savefig(f"../assets/perceptron/erro_{self.name}.png")
        plt.close()

    def train(self, data: pd.DataFrame, learning_rate=0.01, epochs=1000, seed=42):
        labels = data["label"]
        x = data.drop("label", axis=1)
        size = len(x.columns)

        self.weights = np.random.default_rng(seed).random(size)
        classification_error = []

        x_vals = x.values
        label_vals = labels.values
        epochs_run = 0

        for e in range(epochs):
            count = 0
            for i in range(len(data)):
                u = np.dot(self.weights, x_vals[i])
                y = self.sign(u)
                error = label_vals[i] - y
                if error != 0:
                    count += 1

                delta = self.get_delta(learning_rate, error, x_vals[i])
                self.weights = self.weights + delta

            classification_error.append(count / len(data))
            epochs_run += 1

            if count == 0:
                print(f"Converged at epoch {epochs_run}")
                break

        train_acc = 1 - classification_error[-1]

        print(f"Final weights: {self.weights}")
        print(f"Final Training Error: {classification_error[-1]:.4f}")
        print(f"Final Training Accuracy: {train_acc:.4f}")
        self.plot_error(classification_error)
        self.plot_decision_boundary(x, labels, step="train")

        return epochs_run, train_acc

    def test(self, data):
        labels = data["label"]
        x = data.drop("label", axis=1)

        u = np.dot(x.values, self.weights)
        preds = np.where(u >= 0, 1, -1)

        error = np.sum(labels.values != preds)
        accuracy = 1 - (error / len(data))

        print(f"Test Accuracy: {accuracy:.4f}")
        self.plot_decision_boundary(x, labels, step="test")

        return accuracy


if __name__ == "__main__":
    hyperparams = {"epochs": 100, "LR": 0.1, "seed": 42}
    metrics_log = []

    try:
        for i in range(1, 4):
            train_path = Path(f"../assets/train_dataset{i}.csv")
            test_path = Path(f"../assets/test_dataset{i}.csv")

            if not train_path.exists() or not test_path.exists():
                print(f"Skipping dataset {i}: Data paths do not exist")
                continue

            preprocessor = Perceptron()
            df = preprocessor.preprocess(train_path)
            df_test = preprocessor.preprocess(test_path)

            if i != 3:
                model = Perceptron(f"Perceptron_DS{i}")
                print(f"\n=== Training Dataset {i} ===")
                ep, tr_acc = model.train(
                    df,
                    learning_rate=hyperparams["LR"],
                    epochs=hyperparams["epochs"],
                    seed=hyperparams["seed"],
                )
                print("--- Testing ---")
                te_acc = model.test(df_test)

                metrics_log.append(
                    {
                        "Dataset": i,
                        "LR": hyperparams["LR"],
                        "Max_Epochs": hyperparams["epochs"],
                        "Epochs_Run": ep,
                        "Train_Acc": tr_acc,
                        "Test_Acc": te_acc,
                    }
                )

            if i == 3:
                learning_rates = [0.1, 0.001, 0.0001]
                epochs = [100, 200]

                for lr, max_ep in product(learning_rates, epochs):
                    model = Perceptron(f"Perceptron_DS3_LR{lr}_Epoch{max_ep}")
                    print(
                        f"\n=== Training Dataset 3 with LR: {lr}, Max Epochs: {max_ep} ==="
                    )
                    ep, tr_acc = model.train(
                        df, learning_rate=lr, epochs=max_ep, seed=hyperparams["seed"]
                    )
                    print("--- Testing ---")
                    te_acc = model.test(df_test)

                    metrics_log.append(
                        {
                            "Dataset": i,
                            "LR": lr,
                            "Max_Epochs": max_ep,
                            "Epochs_Run": ep,
                            "Train_Acc": tr_acc,
                            "Test_Acc": te_acc,
                        }
                    )

        if metrics_log:
            metrics_df = pd.DataFrame(metrics_log)
            metrics_df = metrics_df.round(4)
            log_path = Path("../assets/perceptron/metrics_log.csv")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(log_path, index=False)
            print(
                f"\nTodas as métricas do Perceptron foram salvas com sucesso em: {log_path}"
            )

    except KeyboardInterrupt:
        print("\nExiting...")
        exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
