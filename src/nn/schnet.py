import torch
from torch import nn

from gnn.src.nn.activation import shifted_softplus
from gnn.src.nn.dense import DenseLayer
from gnn.src.nn.embedding import EmbeddingLayer
from gnn.src.nn.interaction import InteractionLayer


class SchNet(nn.Module):

    def __init__(self, n_interactions = 3, n_features = 64, n_filters = 64):
        super().__init__()

        activation = shifted_softplus

        self.embedding = EmbeddingLayer(n_features)

        self.n_interactions = n_interactions
        self.interaction = InteractionLayer(n_features, n_filters)

        self.aw1 = DenseLayer(n_features, n_features, activation=activation)
        self.aw2 = DenseLayer(n_features, 1)

    #z: n
    #r: n*3
    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        #x: n*n_features
        x = self.embedding(z)

        for i in range(self.n_interactions):
            #x: n*n_features
            x = self.interaction(x, r)

        #x: n*n_features
        x = self.aw1(x)
        #x: n
        x = self.aw2(x)
        return x
