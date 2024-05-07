import torch

property_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "μ": 3,
    "α": 4,
    "ϵHOMO": 5,
    "ϵLUMO": 6,
    "ϵgap": 7,
    "〈R2〉": 8,
    "zpve": 9,
    "U0": 10,
    "U": 11,
    "H": 12,
    "G": 13,
    "Cv": 14
}

def get_property(target: torch.Tensor, property: str) -> torch.Tensor:
    return target[:,property_map[property]]