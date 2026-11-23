from torch import nn


class Scale(nn.Module):
    """Multiply each tensor element by a configurable factor."""

    def __init__(self, factor=2.0):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor
