import torch
from torch import nn
from typing import Callable, Union

class DenseLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.linear(input)
        y = self.activation(y)
        return y