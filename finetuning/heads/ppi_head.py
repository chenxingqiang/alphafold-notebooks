"""Protein-Protein Interaction prediction heads.

Supports:
- PPI binding affinity prediction
- Interface residue prediction
- Hot spot prediction
- Complex structure scoring
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


@dataclass
class PPIHeadConfig:
    """Configuration for PPI prediction heads."""

    # Input dimensions
    single_channel: int = 384
    pair_channel: int = 128

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    num_attention_heads: int = 8

    # Interface
    interface_threshold: float = 10.0  # Å

    # Output
    predict_binding: bool = True
    predict_interface: bool = True
    predict_hotspots: bool = True


if TORCH_AVAILABLE:

    class CrossChainAttention(nn.Module):
        """Cross-attention between two protein chains."""

        def __init__(self, config: PPIHeadConfig):
            super().__init__()

            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )

            self.norm = nn.LayerNorm(config.hidden_dim)

        def forward(
            self,
            chain_a: torch.Tensor,
            chain_b: torch.Tensor,
            mask_a: Optional[torch.Tensor] = None,
            mask_b: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Cross-attend between chains."""
            # A attends to B
            attn_a, weights_ab = self.attention(
                chain_a, chain_b, chain_b,
                key_padding_mask=~mask_b if mask_b is not None else None,
            )
            out_a = self.norm(chain_a + attn_a)

            # B attends to A
            attn_b, weights_ba = self.attention(
                chain_b, chain_a, chain_a,
                key_padding_mask=~mask_a if mask_a is not None else None,
            )
            out_b = self.norm(chain_b + attn_b)

            return out_a, out_b, weights_ab


    class PPIBindingHead(nn.Module):
        """Head for PPI binding affinity prediction."""

        def __init__(self, config: PPIHeadConfig):
            super().__init__()

            self.config = config

            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            self.pair_proj = nn.Linear(config.pair_channel, config.hidden_dim)

            # Cross-chain attention layers
            self.cross_attention_layers = nn.ModuleList([
                CrossChainAttention(config) for _ in range(config.num_layers)
            ])

            # Interface feature extraction
            self.interface_mlp = nn.Sequential(
                nn.Linear(config.hidden_dim * 2 + config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )

            # Global pooling
            self.global_pool = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, 1),
            )

            # Output heads
            self.binding_head = nn.Linear(config.hidden_dim, 1)
            self.ddg_head = nn.Linear(config.hidden_dim, 1)  # For mutation effects

        def forward(
            self,
            single: torch.Tensor,
            pair: torch.Tensor,
            chain_a_mask: torch.Tensor,
            chain_b_mask: torch.Tensor,
            interface_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict PPI binding affinity."""
            batch_size = single.shape[0]

            # Project features
            single_proj = self.single_proj(single)
            pair_proj = self.pair_proj(pair)

            # Separate chains
            chain_a = single_proj * chain_a_mask.unsqueeze(-1)
            chain_b = single_proj * chain_b_mask.unsqueeze(-1)

            # Cross-attention
            for layer in self.cross_attention_layers:
                chain_a, chain_b, attn_weights = layer(
                    chain_a, chain_b, chain_a_mask, chain_b_mask
                )

            # Get interface pair features
            # Average pair representation at interface
            interface_pair_mask = chain_a_mask.unsqueeze(-1) * chain_b_mask.unsqueeze(-2)
            pair_interface = (pair_proj * interface_pair_mask.unsqueeze(-1)).sum(dim=(1, 2))
            pair_interface = pair_interface / (interface_pair_mask.sum(dim=(1, 2), keepdim=True) + 1e-8)

            # Pool chains
            chain_a_pooled = (chain_a * chain_a_mask.unsqueeze(-1)).sum(1) / (chain_a_mask.sum(1, keepdim=True) + 1e-8)
            chain_b_pooled = (chain_b * chain_b_mask.unsqueeze(-1)).sum(1) / (chain_b_mask.sum(1, keepdim=True) + 1e-8)

            # Combine
            combined = torch.cat([chain_a_pooled, chain_b_pooled, pair_interface.squeeze(-1)], dim=-1)
            features = self.interface_mlp(combined)

            # Predict
            binding_affinity = self.binding_head(features).squeeze(-1)
            ddg = self.ddg_head(features).squeeze(-1)

            return {
                "binding_affinity": binding_affinity,  # log10(Kd)
                "ddg": ddg,  # ΔΔG for mutations
                "attention_weights": attn_weights,
            }


    class PPIInterfaceHead(nn.Module):
        """Head for interface residue prediction."""

        def __init__(self, config: PPIHeadConfig):
            super().__init__()

            self.config = config

            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            self.pair_proj = nn.Linear(config.pair_channel, config.hidden_dim // 2)

            # Feature aggregation
            self.pair_to_single = nn.Sequential(
                nn.Linear(config.hidden_dim // 2, config.hidden_dim),
                nn.ReLU(),
            )

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(
            self,
            single: torch.Tensor,
            pair: torch.Tensor,
            partner_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict interface residues.

            Args:
                single: [batch, num_res, single_channel]
                pair: [batch, num_res, num_res, pair_channel]
                partner_mask: [batch, num_res] mask for partner chain

            Returns:
                Per-residue interface probability
            """
            # Project
            single_proj = self.single_proj(single)
            pair_proj = self.pair_proj(pair)

            # Aggregate pair features to single
            # For each residue, aggregate pair features from potential partners
            if partner_mask is not None:
                # Only aggregate from partner chain
                pair_masked = pair_proj * partner_mask.unsqueeze(1).unsqueeze(-1)
            else:
                pair_masked = pair_proj

            pair_agg = pair_masked.mean(dim=2)  # [batch, num_res, hidden//2]
            pair_single = self.pair_to_single(pair_agg)

            # Combine
            combined = torch.cat([single_proj, pair_single], dim=-1)

            # Classify
            logits = self.classifier(combined).squeeze(-1)
            probs = torch.sigmoid(logits)

            return {
                "interface_logits": logits,
                "interface_probs": probs,
            }


    class PPIHotspotHead(nn.Module):
        """Head for binding hot spot prediction."""

        def __init__(self, config: PPIHeadConfig):
            super().__init__()

            self.config = config

            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            self.pair_proj = nn.Linear(config.pair_channel, config.hidden_dim)

            # Energy-based features
            self.energy_encoder = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )

            # Hot spot classifier
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

            # ΔΔG predictor (for alanine scanning)
            self.ddg_predictor = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(
            self,
            single: torch.Tensor,
            pair: torch.Tensor,
            interface_mask: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """Predict binding hot spots.

            Hot spots: residues where mutation significantly reduces binding.
            """
            # Project
            single_proj = self.single_proj(single)
            pair_proj = self.pair_proj(pair)

            # Energy-like features from pair representation
            # Sum pair energies for each residue
            energy_features = (pair_proj * interface_mask.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
            energy_encoded = self.energy_encoder(energy_features)

            # Combine
            combined = torch.cat([single_proj, energy_encoded], dim=-1)

            # Classify hot spots
            hotspot_logits = self.classifier(combined).squeeze(-1)
            hotspot_probs = torch.sigmoid(hotspot_logits)

            # Predict ΔΔG for alanine mutation
            ddg = self.ddg_predictor(combined).squeeze(-1)

            # Only consider interface residues
            hotspot_probs = hotspot_probs * interface_mask

            return {
                "hotspot_logits": hotspot_logits,
                "hotspot_probs": hotspot_probs,
                "alanine_ddg": ddg,
            }


# =============================================================================
# NumPy Reference
# =============================================================================

class PPIBindingNumPy:
    """NumPy reference for PPI binding prediction."""

    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim
        self.proj = np.random.randn(384, hidden_dim) * 0.02
        self.output = np.random.randn(hidden_dim * 2, 1) * 0.02

    def forward(
        self,
        single: np.ndarray,
        chain_a_mask: np.ndarray,
        chain_b_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Forward pass."""
        features = single @ self.proj

        # Pool chains
        chain_a = (features * chain_a_mask[:, np.newaxis]).sum(0) / (chain_a_mask.sum() + 1e-8)
        chain_b = (features * chain_b_mask[:, np.newaxis]).sum(0) / (chain_b_mask.sum() + 1e-8)

        # Combine and predict
        combined = np.concatenate([chain_a, chain_b])
        binding = combined @ self.output

        return {"binding_affinity": binding.squeeze()}


def demonstrate_ppi_head():
    """Demonstrate PPI head functionality."""
    print("PPI Head Demonstration")
    print("=" * 50)

    head = PPIBindingNumPy()

    # Two-chain complex
    num_res = 300
    single = np.random.randn(num_res, 384)

    # Chain masks
    chain_a_mask = np.zeros(num_res)
    chain_b_mask = np.zeros(num_res)
    chain_a_mask[:150] = 1  # First chain
    chain_b_mask[150:] = 1  # Second chain

    output = head.forward(single, chain_a_mask, chain_b_mask)

    print(f"Complex: {num_res} residues")
    print(f"Chain A: {int(chain_a_mask.sum())} residues")
    print(f"Chain B: {int(chain_b_mask.sum())} residues")
    print(f"Predicted binding (log10 Kd): {output['binding_affinity']:.4f}")


if __name__ == "__main__":
    demonstrate_ppi_head()
