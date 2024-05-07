import math
import torch
from torch.nn import functional as F


def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x) - math.log(2.0)