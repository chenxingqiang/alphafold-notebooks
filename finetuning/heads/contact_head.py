"""Contact prediction head.

Predicts residue-residue contacts from pair representations.
"""

from dataclasses import dataclass
from typing import Optional, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


@dataclass
class ContactHeadConfig:
    """Configuration for contact prediction head."""

    pair_channel: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    # Contact definition
    contact_threshold: float = 8.0  # Angstroms
    min_sequence_separation: int = 6

    # Output
    symmetric: bool = True


if TORCH_AVAILABLE:

    class ContactHead(nn.Module):
        """Contact prediction head."""

        def __init__(self, config: ContactHeadConfig):
            super().__init__()

            self.config = config

            # MLP for contact prediction
            layers = []
            in_dim = config.pair_channel

            for i in range(config.num_layers):
                out_dim = config.hidden_dim if i < config.num_layers - 1 else 1
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU() if i < config.num_layers - 1 else nn.Identity(),
                    nn.Dropout(config.dropout) if i < config.num_layers - 1 else nn.Identity(),
                ])
                in_dim = out_dim

            self.mlp = nn.Sequential(*layers)

        def forward(
            self,
            pair: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict contacts.

            Args:
                pair: Pair representation [batch, N, N, pair_channel]
                mask: Residue mask [batch, N]

            Returns:
                Contact predictions
            """
            # Predict logits
            logits = self.mlp(pair).squeeze(-1)  # [batch, N, N]

            # Symmetrize
            if self.config.symmetric:
                logits = (logits + logits.transpose(-1, -2)) / 2

            # Apply sequence separation mask
            N = logits.shape[-1]
            device = logits.device
            sep_mask = torch.abs(
                torch.arange(N, device=device).unsqueeze(0) -
                torch.arange(N, device=device).unsqueeze(1)
            ) >= self.config.min_sequence_separation

            logits = logits.masked_fill(~sep_mask, float("-inf"))

            # Apply residue mask
            if mask is not None:
                pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                logits = logits.masked_fill(~pair_mask, float("-inf"))

            probs = torch.sigmoid(logits)

            return {
                "logits": logits,
                "probs": probs,
            }

        def compute_loss(
            self,
            predictions: Dict[str, torch.Tensor],
            distances: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Compute contact prediction loss."""
            # Create contact labels
            contacts = (distances < self.config.contact_threshold).float()

            # BCE loss
            logits = predictions["logits"]

            # Mask for valid pairs
            N = logits.shape[-1]
            device = logits.device
            sep_mask = torch.abs(
                torch.arange(N, device=device).unsqueeze(0) -
                torch.arange(N, device=device).unsqueeze(1)
            ) >= self.config.min_sequence_separation

            if mask is not None:
                pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                valid_mask = sep_mask & pair_mask
            else:
                valid_mask = sep_mask

            loss = F.binary_cross_entropy_with_logits(
                logits[valid_mask],
                contacts[valid_mask],
            )

            return {"loss": loss}


# =============================================================================
# NumPy Reference
# =============================================================================

class ContactHeadNumPy:
    """NumPy reference implementation."""

    def __init__(self, pair_channel: int = 128):
        self.w1 = np.random.randn(pair_channel, 64) * 0.02
        self.w2 = np.random.randn(64, 1) * 0.02

    def forward(self, pair: np.ndarray) -> np.ndarray:
        """Predict contact probabilities."""
        h = np.maximum(0, pair @ self.w1)  # ReLU
        logits = (h @ self.w2).squeeze(-1)

        # Symmetrize
        logits = (logits + logits.T) / 2

        # Sigmoid
        probs = 1 / (1 + np.exp(-logits))

        return probs


def demonstrate_contact_head():
    """Demonstrate contact prediction."""
    print("Contact Prediction Head Demonstration")
    print("=" * 50)

    head = ContactHeadNumPy()

    num_res = 100
    pair = np.random.randn(num_res, num_res, 128)

    probs = head.forward(pair)

    print(f"Pair input shape: {pair.shape}")
    print(f"Contact probs shape: {probs.shape}")
    print(f"Contact probability range: [{probs.min():.4f}, {probs.max():.4f}]")

    # Estimate contacts at 8Å threshold
    predicted_contacts = (probs > 0.5).sum()
    print(f"Predicted contacts (p>0.5): {predicted_contacts}")


if __name__ == "__main__":
    demonstrate_contact_head()
