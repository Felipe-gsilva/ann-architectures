# ANN Architectures

Repositório com implementações de:
- **1 camada:** Perceptron e Adaline
- **N camadas:** MLP

## Objetivo do projeto

O objetivo é evoluir este repositório para uma **biblioteca funcional de implementações padrão** de arquiteturas de redes neurais, com foco em:
- reprodutibilidade de experimentos
- comparação entre abordagens e hiperparâmetros
- organização dos resultados para análise

## Setup

Você pode usar `pip` ou `uv`.

Com `pip`:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Com `uv`:

```sh
uv sync
```

## Execução do MLP (multilayer)

Da raiz do projeto:

```sh
python models/multilayer/MLP.py --tests
```

Ou com `uv`:

```sh
uv run python models/multilayer/MLP.py --tests
```

Os resultados (CSV + gráficos) são gerados em `assets/mlp/`.

Documentação detalhada do MLP: `models/multilayer/README.md`.

## Relatórios e artefatos

Os relatórios e gráficos gerados pelos modelos ficam na pasta `assets/`, em subpastas por arquitetura:
- `assets/mlp/`: resultados do MLP (ex.: `*_results.csv` e gráficos de erro/acurácia/fronteira)
- `assets/adaline/`: métricas e gráficos do Adaline
- `assets/perceptron/`: métricas e gráficos do Perceptron
