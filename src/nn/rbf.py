import torch
from torch import nn

class RBFLayer(nn.Module):
    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        self.offsets = nn.Parameter(
            torch.linspace(0, cutoff, n_rbf, dtype=torch.float)
        )
        coeff = nn.Parameter(
            torch.full_like(
                self.offsets, 
                fill_value=cutoff/n_rbf,
                dtype=torch.float
            )
        )
        self.coeff = -0.5 / torch.pow(coeff, 2)

    #d: n*n*3 
    def forward(self, d: torch.Tensor) -> torch.Tensor:
        #rbf: n*n*3*rbf_features
        rbf = torch.exp(self.coeff * torch.pow(d.unsqueeze(-1) - self.offsets, 2))
        return rbf