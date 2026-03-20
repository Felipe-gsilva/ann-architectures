import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.typing import NDArray
from itertools import product

class Adaline: 
    weights: NDArray
    name: str

    def __init__(self, name: str ="Adaline"): 
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

    def plot_decision_boundary(self, x, labels, step: str = "train"): 
        if len(self.weights) > 3:
            print(f"Skipping decision boundary plot for {self.name}: Data is {len(self.weights)-1}D.")
            return
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
        plt.savefig(f'../assets/adaline/{step}_decision_boundary_{self.name}.png')
        plt.show()
        plt.close()

    def plot_error(self, classification_error):
        # cria uma nova figura limpa para este gráfico específico
        plt.figure() 
        
        plt.plot(classification_error, color='blue')
        plt.xlabel('épocas')
        plt.ylabel('erro de classificação')
        plt.title(f'erro de classificação x época\n({self.name})')
        plt.tight_layout()
        plt.savefig(f'../assets/adaline/erro_{self.name}.png')
        plt.show() 
        plt.close()

    def plot_eqm(self, eqms): 
        # cria uma nova figura limpa para este gráfico específico
        plt.figure() 
        
        plt.plot(eqms, color='red')
        plt.xlabel('épocas')
        plt.ylabel('erro quadrático médio')
        plt.title(f'erro quadrático médio x época\n({self.name})')
        plt.tight_layout()
        plt.savefig(f'../assets/adaline/eqm_{self.name}.png')
        plt.show() 
        plt.close()

    def eqm(self, x, d): 
        w = self.weights
        eqm = 0.0

        for i in range(len(x)): 
            u = np.dot(x.iloc[i], w)
            eqm += ((d[i] - u) ** 2)

        return eqm / len(x)

    def train(self, data: pd.DataFrame, learning_rate=0.01, precision=0.01, seed=42):
        labels = data['label']
        x = data.drop('label', axis=1)
        size = len(x.columns)
        
        self.weights = np.random.default_rng(seed).random(size)
        classification_error = []
        epoch = 0
        preds = []
        eqms = [0.0]

        while True: 
            count = 0
            for i in range(len(data)): 
                u = np.dot(self.weights, x.iloc[i].values)
                y = self.sign(u)
                error = labels.iloc[i] - u 
                self.weights = self.weights + self.get_delta(learning_rate, error, x.iloc[i].values)
                preds.append(y)
                # classification_error calc
                if labels.iloc[i] - y != 0: 
                    count += 1
                
            classification_error.append(count / len(data))
            eqms.append(self.eqm(x, labels))
            epoch += 1

            if abs(eqms[-1] - eqms[-2]) < precision or epoch > 100:
                break
            else:
                preds = []

        print(f"Final weights: {self.weights}")
        print(f"Final Training Accuracy: {1 - classification_error[-1]}")
        print(f"Final eqm {eqms[-1]}")
        self.plot_error(classification_error)
        self.plot_eqm(eqms)
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

        print(f"Test Accuracy: {1 - (error / len(data))}")
        self.plot_decision_boundary(x, labels, step="test")
        return


if __name__ == "__main__": 
    models = [Adaline(f"Perceptron_{str(i)}") for i in range(3)]
    hyperparams = {"epsilon": 0.001, "LR": 0.1, "seed": 42}
    
    try:
        for i in range(1, 4):
            model = models[i - 1]
            train_path = Path(f"../assets/train_dataset{i}.csv")
            test_path = Path(f"../assets/test_dataset{i}.csv")

            if not train_path.exists() or not test_path.exists(): 
                raise FileExistsError("Train or Test dataset does not exist")

            df = model.preprocess(train_path)
            df_test = model.preprocess(test_path)
            if i != 3:
                print("--- Training ---")
                model.train(df, hyperparams["LR"], hyperparams["epsilon"], hyperparams["seed"])
                print("\n--- Testing ---")
                model.test(df_test)

            if i == 3: 
                learning_rates = [0.01, 0.001, 0.0001]
                epsilon = [0.1, 0.00001]
                for (lr, precision) in product(learning_rates, epsilon): 
                    model = Adaline(f"Perceptron_LR{lr}_Precision{precision}")
                    print(f"\n--- Training with LR: {lr}, precision: {precision} ---")
                    model.train(df, lr, precision, hyperparams["seed"])
                    print("\n--- Testing ---")
                    model.test(df_test)

    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
    except Exception as e:
        print(e)
        exit(0)
