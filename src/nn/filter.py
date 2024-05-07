import torch
from torch import nn

from gnn.src.nn.activation import shifted_softplus
from gnn.src.nn.dense import DenseLayer
from gnn.src.nn.rbf import RBFLayer

class PoolLayer(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        
        self.pool = torch.nn.MaxPool3d((1, 3, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = torch.squeeze(x, -2)
        return x

    
class FilterLayer(nn.Module):
    def __init__(self, n_rbf: int, n_filters: int) -> None:
        super().__init__()

        activation = shifted_softplus

        self.rbf = RBFLayer(n_rbf, 5)
        self.dense = DenseLayer(n_rbf, n_filters, activation=activation)
        self.pool = PoolLayer()
    
    #d: n*n*3
    def forward(self, d: torch.Tensor) -> torch.Tensor:
        #x: n*n*3*rbf_features
        x = self.rbf(d)
        #x: n*n*3*filter_number
        x = self.dense(x)
        #x: n*n*filter_number
        x = self.pool(x)
        return x