from typing import Callable
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomRotation, RandomCrop, RandomAffine, Normalize

class CONFIG:
    batch_size = 1
    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=0.001)

    transforms = Compose(
        [
            ToTensor(),
            Normalize(0.5, 0.5),
            RandomRotation(10),
            RandomAffine(5),
        ]
    )