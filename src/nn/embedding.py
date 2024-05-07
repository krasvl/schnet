import torch
from torch import nn


e_conf = [
  # Z  1s 2s 2p
  [ 0, 0, 0, 0 ], # 0
  [ 1, 1, 0, 0 ], # H
  [ 2, 2, 0, 0 ], # He
  [ 3, 2, 1, 0 ], # Li
  [ 4, 2, 2, 0 ], # Be
  [ 5, 2, 2, 1 ], # B
  [ 6, 2, 2, 2 ], # C
  [ 7, 2, 2, 3 ], # N
  [ 8, 2, 2, 4 ], # O
  [ 9, 2, 2, 5 ], # F
]

def init_embeddings(e_conf_len: int, out_features: int):
  return torch.ones([e_conf_len, out_features], dtype=torch.float)

class EmbeddingLayer(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.e_conf = torch.tensor(e_conf, dtype=torch.float)
        self.element_embedding = nn.Parameter(init_embeddings(self.e_conf.size(0), out_features))
        self.linear = nn.Linear(self.e_conf.size(1), out_features, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        embedding = self.element_embedding + self.linear(self.e_conf)
        return embedding[z]