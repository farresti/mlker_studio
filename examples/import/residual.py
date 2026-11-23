"""Small trusted PyTorch source for the Model Builder import walkthrough."""
from torch import nn


class Residual(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.projection = nn.Linear(width, width)

    def forward(self, x):
        return self.projection(x).relu() + x
