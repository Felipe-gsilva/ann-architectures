from dataclasses import dataclass
import enum
from typing import List, Self
from torch import Tensor
import torch
from torch.optim import SGD
import torch.nn as nn


class init_type(enum.Enum):
    random: str
    xavier: str
    he: str


class activation_fn(enum.Enum):
    relu: nn.ReLU
    tanh: nn.Tanh


@dataclass
class layer:
    dim: int
    activ: activation_fn


class MLP(nn.Module):
    def __init__(
        self, df: Tensor, layers: List[layer], LR=0.01, classify: bool = False
    ) -> None:
        super(MLP, self).__init__()
        self.weights = self.init_weights(layers)
        self.optim = SGD(df, lr=LR) if not classify else nn.BCEWithLogitsLoss()
        self.hidden = nn.Sequential()

        current_dim = layers[0].dim
        for l in layers[1:]:
            self.hidden.append(nn.Linear(current_dim, l.dim))
            self.hidden.append(l.activ.value)
            current_dim = l.dim

    def init_weights(
        self, layers: List[layer],
        _type: init_type
    ) -> List[Tensor]:
        weights = []
        # TODO check for init types (random, xavier, etc.)
        for i in range(len(layers) - 1):
            # only random for now, but can be changed to other types of initialization
            w = torch.rand(layers[i].dim, layers[i + 1].dim)
            weights.append(w)
        return weights

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.hidden) // 2):
            # linear transformation
            x = self.hidden[2 * i](x)
            # activation function
            x = self.hidden[2 * i + 1](x)
        return x

    def layers_len(self):
        return len(self.hidden) + 1

    def train(self, mode: bool = True) -> Self:
        return super().train(mode)

    def eval(self) -> Self:
        return super().eval()


def main():
    print("Hello from mlp!")


if __name__ == "__main__":
    main()
