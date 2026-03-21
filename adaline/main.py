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
                raise FileNotFoundError(f"Data path does not exist: {data_path}")
                
            df = pd.read_csv(data_path)
            if 'bias' not in df.columns:
                df.insert(0, 'bias', 1)
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
            print(f"Skipping decision boundary plot for {self.name}: Data is {len(self.weights)-1}D.")
            return
        x_min, x_max = x.iloc[:, 1].min() - 1, x.iloc[:, 1].max() + 1
        y_min, y_max = x.iloc[:, 2].min() - 1, x.iloc[:, 2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        
        # Fast vectorized calculation
        grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
        Z = np.where(np.dot(grid, self.weights) >= 0, 1, -1)
        Z = Z.reshape(xx.shape)
        
        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
        plt.scatter(x.iloc[:, 1], x.iloc[:, 2], c=labels, cmap='RdYlBu', edgecolors='k', marker='o')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Decision Boundary of {self.name}')
        
        Path('../assets/adaline').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'../assets/adaline/{step}_decision_boundary_{self.name}.png')
        plt.close()

    def plot_error(self, classification_error):
        plt.figure() 
        plt.plot(classification_error, color='blue', marker='o', markersize=3)
        plt.xlabel('Épocas')
        plt.ylabel('Erro de Classificação')
        plt.title(f'Erro de Classificação x Época\n({self.name})')
        plt.tight_layout()
        plt.savefig(f'../assets/adaline/erro_{self.name}.png')
        plt.close()

    def plot_eqm(self, eqms): 
        plt.figure() 
        plt.plot(eqms, color='red', marker='o', markersize=3)
        plt.xlabel('Épocas')
        plt.ylabel('Erro Quadrático Médio')
        plt.title(f'Erro Quadrático Médio x Época\n({self.name})')
        plt.tight_layout()
        plt.savefig(f'../assets/adaline/eqm_{self.name}.png')
        plt.close()

    def eqm(self, x, d): 
        u = np.dot(x.values, self.weights)
        return np.mean((d.values - u) ** 2)

    def train(self, data: pd.DataFrame, learning_rate=0.01, batch=False, max_epochs=100, precision=0.01, seed=42):
        labels = data['label']
        x = data.drop('label', axis=1)
        size = len(x.columns)
        p = len(data) 
        
        self.weights = np.random.default_rng(seed).random(size)
        classification_error = []
        epoch = 0
        eqms = []

        x_vals = x.values
        label_vals = labels.values

        while epoch < max_epochs: 
            count = 0
            delta_w_sum = np.zeros(size)

            for i in range(p): 
                u = np.dot(self.weights, x_vals[i])
                y = self.sign(u)
                error = label_vals[i] - u 
                
                if batch:
                    delta_w_sum += error * x_vals[i]
                else:
                    self.weights += self.get_delta(learning_rate, error, x_vals[i])
                
                if label_vals[i] != y: 
                    count += 1
            
            if batch:
                self.weights += learning_rate * (1/p) * delta_w_sum
                
            classification_error.append(count / p)
            eqms.append(self.eqm(x, labels))
            epoch += 1

            if len(eqms) > 1 and abs(eqms[-1] - eqms[-2]) < precision:
                print(f"Converged at epoch {epoch}")
                break

        train_acc = 1 - classification_error[-1]
        final_eqm = eqms[-1]

        print(f"Final weights: {self.weights}")
        print(f"Final Training Accuracy: {train_acc:.4f}")
        print(f"Final eqm {final_eqm:.4f}")
        self.plot_error(classification_error)
        self.plot_eqm(eqms)
        self.plot_decision_boundary(x, labels, step="train")
        
        return epoch, train_acc, final_eqm

    def test(self, data):
        labels = data['label']
        x = data.drop('label', axis=1)
        
        u = np.dot(x.values, self.weights)
        preds = np.where(u >= 0, 1, -1)
        
        error = np.sum(labels.values != preds)
        accuracy = 1 - (error / len(data))

        print(f"Test Accuracy: {accuracy:.4f}")
        self.plot_decision_boundary(x, labels, step="test")
        return accuracy


if __name__ == "__main__": 
    hyperparams_ds1_2 = {"epsilon": 0.001, "LR": 0.01, "max_epochs": 100, "seed": 42}
    metrics_log = []
    
    try:
        for i in range(1, 4):
            train_path = Path(f"../assets/train_dataset{i}.csv")
            test_path = Path(f"../assets/test_dataset{i}.csv")

            if not train_path.exists() or not test_path.exists(): 
                print(f"Skipping dataset {i}: Data paths do not exist")
                continue

            preprocessor = Adaline()
            df = preprocessor.preprocess(train_path)
            df_test = preprocessor.preprocess(test_path)
            
            if i != 3:
                # --- Amostra ---
                model = Adaline(f"Adaline_DS{i}_Amostra")
                print(f"\n=== Training Dataset {i} (Por Amostra) ===")
                ep, tr_acc, f_eqm = model.train(df, learning_rate=hyperparams_ds1_2["LR"], batch=False, 
                            max_epochs=hyperparams_ds1_2["max_epochs"], precision=hyperparams_ds1_2["epsilon"], 
                            seed=hyperparams_ds1_2["seed"])
                print("--- Testing ---")
                te_acc = model.test(df_test)
                
                metrics_log.append({"Dataset": i, "Abordagem": "Amostra", "LR": hyperparams_ds1_2["LR"], 
                                    "Precision": hyperparams_ds1_2["epsilon"], "Epochs": ep, 
                                    "Final_EQM": f_eqm, "Train_Acc": tr_acc, "Test_Acc": te_acc})
                
                # --- Batch ---
                model_batch = Adaline(f"Adaline_DS{i}_Batch")
                print(f"\n=== Training Dataset {i} (Batch) ===")
                ep, tr_acc, f_eqm = model_batch.train(df, learning_rate=hyperparams_ds1_2["LR"], batch=True, 
                                  max_epochs=hyperparams_ds1_2["max_epochs"], precision=hyperparams_ds1_2["epsilon"], 
                                  seed=hyperparams_ds1_2["seed"])
                print("--- Testing ---")
                te_acc = model_batch.test(df_test)
                
                metrics_log.append({"Dataset": i, "Abordagem": "Batch", "LR": hyperparams_ds1_2["LR"], 
                                    "Precision": hyperparams_ds1_2["epsilon"], "Epochs": ep, 
                                    "Final_EQM": f_eqm, "Train_Acc": tr_acc, "Test_Acc": te_acc})

            if i == 3: 
                learning_rates = [0.01, 0.001, 0.0001]
                epsilon = [0.1, 0.0001]
                max_epochs = 100
                
                for (lr, precision) in product(learning_rates, epsilon): 
                    # --- Amostra ---
                    model = Adaline(f"Adaline_DS3_LR{lr}_Prec{precision}")
                    print(f"\n=== Training Dataset 3 (Amostra) with LR: {lr}, precision: {precision} ===")
                    ep, tr_acc, f_eqm = model.train(df, learning_rate=lr, batch=False, max_epochs=max_epochs, 
                                precision=precision, seed=hyperparams_ds1_2["seed"])
                    print("--- Testing ---")
                    te_acc = model.test(df_test)
                    
                    metrics_log.append({"Dataset": 3, "Abordagem": "Amostra", "LR": lr, 
                                        "Precision": precision, "Epochs": ep, 
                                        "Final_EQM": f_eqm, "Train_Acc": tr_acc, "Test_Acc": te_acc})

                    # --- Batch ---
                    model_batch = Adaline(f"Adaline_DS3_LR{lr}_Prec{precision}_Batch")
                    print(f"\n=== Training Dataset 3 (Batch) with LR: {lr}, precision: {precision} ===")
                    ep, tr_acc, f_eqm = model_batch.train(df, learning_rate=lr, batch=True, max_epochs=max_epochs, 
                                            precision=precision, seed=hyperparams_ds1_2["seed"])
                    print("--- Testing ---")
                    te_acc = model_batch.test(df_test)
                    
                    metrics_log.append({"Dataset": 3, "Abordagem": "Batch", "LR": lr, 
                                        "Precision": precision, "Epochs": ep, 
                                        "Final_EQM": f_eqm, "Train_Acc": tr_acc, "Test_Acc": te_acc})

        if metrics_log:
            metrics_df = pd.DataFrame(metrics_log)
            metrics_df = metrics_df.round(4)
            log_path = Path('../assets/adaline/metrics_log.csv')
            metrics_df.to_csv(log_path, index=False)
            print(f"\nTodas as métricas foram salvas com sucesso em: {log_path}")

    except KeyboardInterrupt:
        print("\nExiting...")
        exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
