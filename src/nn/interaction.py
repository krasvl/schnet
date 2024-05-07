import torch
from torch import nn

from gnn.src.nn.activation import shifted_softplus
from gnn.src.nn.dense import DenseLayer
from gnn.src.nn.filter import FilterLayer


class InteractionLayer(nn.Module):

    def __init__(self, n_features: int, n_filters: int):
        super().__init__()

        activation = shifted_softplus

        self.aw1 = DenseLayer(n_features, n_features)
        self.aw2 = DenseLayer(n_features, n_features, activation=activation)
        self.aw3 = DenseLayer(n_features, n_features)

        self.f = FilterLayer(n_filters, n_features)

    #x: n*n_features
    #r: n*3
    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        #x: n*n_features
        x = self.aw1(x)
        #d: n*n*3
        d = torch.abs(r.unsqueeze(-3) - r.unsqueeze(-2))
        #w: n*n*n_features
        w = self.f(d)
        x = x.unsqueeze(1)
        #conv
        #x: n*n_features
        x = torch.sum(w*x, dim=-2)
        
        #x: n*n_features
        x = self.aw2(x)
        #x: n*n_features
        x = self.aw3(x)
        return x