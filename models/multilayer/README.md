# MLP (`models/multilayer`)

Este módulo treina e avalia uma **rede neural MLP** com busca de combinações de:
- otimizador (`SGD`, `Adam`)
- arquitetura
- função de ativação (`ReLU`, `Tanh`)
- regularização (`none`, `dropout`, `l2`)

Os resultados e gráficos são salvos em `assets/mlp/`.

## Como executar (professor)

> Rode os comandos a partir da **raiz do repositório** (`ann-architectures`), pois os caminhos dos datasets são relativos a essa pasta.

### 1) Instalar dependências

Com `pip`:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ou com `uv`:

```sh
uv sync
```

### 2) Rodar os testes pré-definidos do MLP

Com `pip`:

```sh
python models/multilayer/MLP.py --tests
```

Com `uv`:

```sh
uv run python models/multilayer/MLP.py --tests
```

Esse comando executa 3 testes com os datasets:
- `assets/train_dataset1.csv`, `assets/val_dataset1.csv`, `assets/test_dataset1.csv`
- `assets/train_dataset2.csv`, `assets/val_dataset2.csv`, `assets/test_dataset2.csv`
- `assets/train_dataset.csv`, `assets/validation_dataset.csv`, `assets/test_dataset.csv`

## Execução manual (opcional)

Dataset único (o script faz split interno em treino/validação/teste):

```sh
python models/multilayer/MLP.py --path assets/train_dataset.csv
```

Datasets já separados:

```sh
python models/multilayer/MLP.py \
  --list_path assets/train_dataset1.csv assets/val_dataset1.csv assets/test_dataset1.csv
```

Também é possível ajustar batch size:

```sh
python models/multilayer/MLP.py --tests --batch_size 32
```
