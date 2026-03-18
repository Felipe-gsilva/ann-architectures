from numpy.typing import NDArray
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class Perceptron: 
    weights: NDArray
    name: str

    def __init__(self, name: str ="Perceptron"): 
        self.name = name

    def preprocess(self, data_path: Path) -> pd.DataFrame: 
        try: 
            if not data_path.exists(): 
                raise Exception("Data path does not exist")
                
            df = pd.read_csv(data_path)
            df.insert(0, 'bias', 1)
            return df
        except Exception as e: 
            print(e)
            exit(0)

    def sign(self, u): 
        return 1 if u >= np.float32(0) else -1

    def get_delta(self, learning_rate, error, x):
        return learning_rate * error * x

    def plot_error(self, classification_error):
        plt.plot(classification_error)
        plt.xlabel('Epochs')
        plt.ylabel('Classification Error')
        plt.title('Classification Error over Epochs')
        plt.savefig('classification_error_{self.name}.png')
        plt.show()

    def plot_decision_boundary(self, x, labels, step: str = "train"):
        x_min, x_max = x.iloc[:, 1].min() - 1, x.iloc[:, 1].max() + 1
        y_min, y_max = x.iloc[:, 2].min() - 1, x.iloc[:, 2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = np.array([self.sign(np.dot(self.weights, [1, xx[i][j], yy[i][j]])) for i in range(xx.shape[0]) for j in range(xx.shape[1])])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(x.iloc[:, 1], x.iloc[:, 2], c=labels, edgecolors='k', marker='o')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Decision Boundary of {self.name}')
        plt.savefig(f'../assets/output/{step}_decision_boundary_{self.name}.png')
        plt.show()

    def train(self, data: pd.DataFrame, learning_rate=0.01, epochs=1000, seed=42):
        labels = data['label']
        x = data.drop('label', axis=1)
        size = len(x.columns)
        
        self.weights = np.random.default_rng(seed).random(size)
        classification_error = []
        
        for _ in range(epochs): 
            count = 0
            for i in range(len(data)): 
                u = np.dot(self.weights, x.iloc[i].values)
                y = self.sign(u)
                error = labels.iloc[i] - y 
                if error != 0: 
                    count += 1
                
                delta = self.get_delta(learning_rate, error, x.iloc[i].values)
                self.weights = self.weights + delta
                
            classification_error.append(count / len(data))

        print(f"Final weights: {self.weights}")
        print(f"Final Training Error: {classification_error[-1]}") 
        print(f"Final Training Accuracy: {1 - classification_error[-1]}")
        self.plot_error(classification_error)
        self.plot_decision_boundary(x, labels, step="train")

    def test(self, data):
        preds = []
        error = 0
        labels = data['label']
        x = data.drop('label', axis=1)

        for i in range(len(data)): 
            u = np.dot(self.weights, x.iloc[i].values)
            y = self.sign(u)
            preds.append(y)
            if labels.iloc[i] - y != 0: 
                error += 1

        print(f"Test Predictions: {preds}")
        print(f"Test Error: {error / len(data)}")
        print(f"Test Accuracy: {1 - (error / len(data))}")
        self.plot_decision_boundary(x, labels, step="test")
        return


if __name__ == "__main__": 
    models = [Perceptron(str(i)) for i in range(3)]
    hyperparams = {"epochs": 100, "LR": 0.1, "seed": 42}
    
    try:
        if not Path("../assets").exists(): 
            raise Exception("Assets path does not exist")
        for i in range(1, 3):
            model = models[i]
            print("--- Training ---")
            df = model.preprocess(Path(f"../assets/train_dataset{i}.csv"))
            model.train(df, hyperparams["LR"], hyperparams["epochs"], hyperparams["seed"])
            print("\n--- Testing ---")
            df_test = model.preprocess(Path(f"../assets/test_dataset{i}.csv"))
            model.test(df_test)

    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
    except Exception as e:
        print(e)
        exit(0)
