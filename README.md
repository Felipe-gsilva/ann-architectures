# ANN Architectures

Arquiteturas disponíveis neste repositório:
- **Perceptron**
- **Adaline**
- **MLP**
- **CNN**

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

## Padrão de estrutura de arquivos

Para evitar repetição entre arquiteturas, o projeto segue este padrão:

```sh
src/models/
├── simple/
│   ├── perceptron.py
│   └── adaline.py
└── multilayer/
    ├── MLP.py
    └── CNN.py

assets/
└── <arquitetura>/
    ├── *_results.csv
    └── gráficos

docs/
└── <arquitetura>/
    ├── <arquitetura>.tex
    └── <arquitetura>.pdf
```

Onde:
- o código-fonte dos modelos fica em `src/models/`
- os resultados dos experimentos ficam em `assets/<arquitetura>/`
- a documentação de cada arquitetura fica em `docs/<arquitetura>/`

## Execução (exemplo com MLP)

Da raiz do projeto:

```sh
python src/models/multilayer/MLP.py --tests
```

Ou com `uv`:

```sh
uv run python src/models/multilayer/MLP.py --tests
```
