"""Small PyTorch model for audio router baseline training."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - exercised only when runtime lacks torch.
    raise RuntimeError("Audio router training requires torch to be installed.") from exc


class AudioRouterMLP(nn.Module):
    """Tiny MLP classifier for lightweight audio features."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return class logits for normalized feature rows."""
        return self.network(features)
