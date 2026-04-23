import torch
import torch.nn as nn


class MLP(nn.Module):
    """Single MLP block -- one FSDP unit.

    Architecture: Linear(H, 4H) -> GELU -> Linear(4H, H).

    Args:
        hidden_dim: Input / output feature dimension.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, hidden_dim).

        Returns:
            Output tensor of shape (batch, hidden_dim).
        """
        return self.fc2(self.act(self.fc1(x)))


class ToyModel(nn.Module):
    """Stack of MLP blocks with residual connections.

    Args:
        hidden_dim: Feature dimension for all layers.
        num_layers: Number of stacked MLP blocks.
    """

    def __init__(self, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([MLP(hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, hidden_dim).

        Returns:
            Output tensor of shape (batch, hidden_dim).
        """
        for layer in self.layers:
            x = x + layer(x)
        return x
