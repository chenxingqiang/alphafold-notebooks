"""Binding affinity prediction head.

This module implements the affinity prediction head similar to Boltz-2,
which predicts protein-ligand binding affinity (pKd, pIC50, ΔG).

Reference: Boltz-2 paper (bioRxiv 2025.06.14.659707)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math

# PyTorch implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


@dataclass
class AffinityHeadConfig:
    """Configuration for the affinity prediction head."""
    
    # Input dimensions
    single_channel: int = 384
    """Dimension of single representation."""
    
    pair_channel: int = 128
    """Dimension of pair representation."""
    
    # Architecture
    hidden_dim: int = 256
    """Hidden dimension for MLP layers."""
    
    num_layers: int = 3
    """Number of MLP layers."""
    
    dropout: float = 0.1
    """Dropout rate."""
    
    # Output
    num_outputs: int = 2
    """Number of outputs: [affinity_value, affinity_probability]."""
    
    # Gaussian smearing for distance features
    use_gaussian_smearing: bool = True
    """Whether to use Gaussian smearing for distance features."""
    
    num_gaussians: int = 50
    """Number of Gaussian basis functions."""
    
    gaussian_start: float = 0.0
    """Start of Gaussian basis range."""
    
    gaussian_end: float = 30.0
    """End of Gaussian basis range."""
    
    # Attention
    use_attention_pooling: bool = True
    """Whether to use attention-based pooling."""
    
    num_attention_heads: int = 4
    """Number of attention heads for pooling."""


if TORCH_AVAILABLE:
    
    class GaussianSmearing(nn.Module):
        """Gaussian smearing for distance features.
        
        Converts distances to Gaussian basis function representations.
        """
        
        def __init__(
            self,
            start: float = 0.0,
            stop: float = 30.0,
            num_gaussians: int = 50,
        ):
            super().__init__()
            
            offset = torch.linspace(start, stop, num_gaussians)
            self.register_buffer("offset", offset)
            self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        
        def forward(self, dist: torch.Tensor) -> torch.Tensor:
            """Apply Gaussian smearing to distances.
            
            Args:
                dist: Distances [..., 1] or [...]
            
            Returns:
                Gaussian features [..., num_gaussians]
            """
            if dist.dim() == 0 or dist.shape[-1] != 1:
                dist = dist.unsqueeze(-1)
            
            return torch.exp(self.coeff * (dist - self.offset) ** 2)
    
    
    class AttentionPooling(nn.Module):
        """Attention-based pooling for variable-length sequences."""
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_heads: int = 4,
        ):
            super().__init__()
            
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            
            self.query = nn.Parameter(torch.randn(1, num_heads, self.head_dim))
            self.key_proj = nn.Linear(input_dim, hidden_dim)
            self.value_proj = nn.Linear(input_dim, hidden_dim)
            
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)
            
            # Initialize
            nn.init.xavier_uniform_(self.query)
        
        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Pool sequence to single vector via attention.
            
            Args:
                x: Input [batch, seq_len, input_dim]
                mask: Mask [batch, seq_len], True for valid positions
            
            Returns:
                Pooled representation [batch, hidden_dim]
            """
            batch_size, seq_len, _ = x.shape
            
            # Project keys and values
            keys = self.key_proj(x)  # [batch, seq_len, hidden_dim]
            values = self.value_proj(x)
            
            # Reshape for multi-head attention
            keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
            values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Expand query for batch
            query = self.query.expand(batch_size, -1, -1)  # [batch, num_heads, head_dim]
            
            # Attention scores
            scores = torch.einsum("bhd,bshd->bhs", query, keys)
            scores = scores / math.sqrt(self.head_dim)
            
            # Apply mask
            if mask is not None:
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1)
                scores = scores.masked_fill(~mask, float("-inf"))
            
            # Softmax
            attn = F.softmax(scores, dim=-1)
            
            # Weighted sum
            out = torch.einsum("bhs,bshd->bhd", attn, values)
            out = out.reshape(batch_size, -1)  # [batch, hidden_dim]
            
            return self.output_proj(out)
    
    
    class AffinityHead(nn.Module):
        """Binding affinity prediction head.
        
        Predicts binding affinity from protein-ligand complex representations.
        Based on Boltz-2's affinity module architecture.
        """
        
        def __init__(self, config: AffinityHeadConfig):
            super().__init__()
            
            self.config = config
            
            # Gaussian smearing for distance features
            if config.use_gaussian_smearing:
                self.gaussian_smearing = GaussianSmearing(
                    start=config.gaussian_start,
                    stop=config.gaussian_end,
                    num_gaussians=config.num_gaussians,
                )
                distance_features = config.num_gaussians
            else:
                distance_features = 1
            
            # Feature projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            self.pair_proj = nn.Linear(config.pair_channel + distance_features, config.hidden_dim)
            
            # Attention pooling
            if config.use_attention_pooling:
                self.single_pooling = AttentionPooling(
                    input_dim=config.hidden_dim,
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_attention_heads,
                )
                self.pair_pooling = AttentionPooling(
                    input_dim=config.hidden_dim,
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_attention_heads,
                )
            
            # MLP layers
            layers = []
            input_dim = config.hidden_dim * 2  # Concatenated single and pair features
            
            for i in range(config.num_layers - 1):
                layers.extend([
                    nn.Linear(input_dim if i == 0 else config.hidden_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ])
            
            self.mlp = nn.Sequential(*layers)
            
            # Output heads
            self.affinity_value_head = nn.Linear(config.hidden_dim, 1)
            self.affinity_prob_head = nn.Linear(config.hidden_dim, 1)
            
            # Initialize output layers
            nn.init.zeros_(self.affinity_value_head.weight)
            nn.init.zeros_(self.affinity_value_head.bias)
            nn.init.zeros_(self.affinity_prob_head.weight)
            nn.init.zeros_(self.affinity_prob_head.bias)
        
        def forward(
            self,
            single: torch.Tensor,
            pair: torch.Tensor,
            distances: torch.Tensor,
            single_mask: Optional[torch.Tensor] = None,
            pair_mask: Optional[torch.Tensor] = None,
            interface_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict binding affinity.
            
            Args:
                single: Single representation [batch, num_res, single_channel]
                pair: Pair representation [batch, num_res, num_res, pair_channel]
                distances: Pairwise distances [batch, num_res, num_res]
                single_mask: Mask for single representation [batch, num_res]
                pair_mask: Mask for pair representation [batch, num_res, num_res]
                interface_mask: Mask for interface residues [batch, num_res]
            
            Returns:
                Dictionary with:
                    - affinity_pred_value: Predicted affinity (log10 scale)
                    - affinity_probability_binary: Probability of binding
            """
            batch_size = single.shape[0]
            
            # Process distances with Gaussian smearing
            if self.config.use_gaussian_smearing:
                dist_features = self.gaussian_smearing(distances)  # [batch, N, N, num_gaussians]
            else:
                dist_features = distances.unsqueeze(-1)
            
            # Concatenate pair features with distance features
            pair_with_dist = torch.cat([pair, dist_features], dim=-1)
            
            # Project features
            single_proj = self.single_proj(single)  # [batch, N, hidden]
            pair_proj = self.pair_proj(pair_with_dist)  # [batch, N, N, hidden]
            
            # Pool single representation
            if self.config.use_attention_pooling:
                # Focus on interface residues if mask provided
                if interface_mask is not None:
                    single_pooled = self.single_pooling(single_proj, interface_mask)
                else:
                    single_pooled = self.single_pooling(single_proj, single_mask)
            else:
                # Mean pooling
                if single_mask is not None:
                    single_pooled = (single_proj * single_mask.unsqueeze(-1)).sum(1) / single_mask.sum(1, keepdim=True)
                else:
                    single_pooled = single_proj.mean(dim=1)
            
            # Pool pair representation
            # Flatten spatial dimensions
            pair_flat = pair_proj.view(batch_size, -1, self.config.hidden_dim)
            if pair_mask is not None:
                pair_mask_flat = pair_mask.view(batch_size, -1)
            else:
                pair_mask_flat = None
            
            if self.config.use_attention_pooling:
                pair_pooled = self.pair_pooling(pair_flat, pair_mask_flat)
            else:
                if pair_mask_flat is not None:
                    pair_pooled = (pair_flat * pair_mask_flat.unsqueeze(-1)).sum(1) / pair_mask_flat.sum(1, keepdim=True)
                else:
                    pair_pooled = pair_flat.mean(dim=1)
            
            # Concatenate and process
            combined = torch.cat([single_pooled, pair_pooled], dim=-1)
            features = self.mlp(combined)
            
            # Predict outputs
            affinity_value = self.affinity_value_head(features).squeeze(-1)
            affinity_prob = torch.sigmoid(self.affinity_prob_head(features)).squeeze(-1)
            
            return {
                "affinity_pred_value": affinity_value,
                "affinity_probability_binary": affinity_prob,
            }
        
        def compute_loss(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            weights: Optional[Dict[str, float]] = None,
        ) -> Dict[str, torch.Tensor]:
            """Compute affinity prediction loss.
            
            Args:
                predictions: Model predictions
                targets: Ground truth targets
                    - affinity_value: True affinity (log10 scale)
                    - has_affinity: Binary indicator for affinity data
                weights: Optional loss weights
            
            Returns:
                Dictionary with loss values
            """
            weights = weights or {"value": 1.0, "prob": 0.1}
            
            # Regression loss for affinity value
            value_loss = F.mse_loss(
                predictions["affinity_pred_value"],
                targets["affinity_value"],
                reduction="none",
            )
            
            # Only compute loss for samples with affinity data
            if "has_affinity" in targets:
                value_loss = value_loss * targets["has_affinity"]
                value_loss = value_loss.sum() / (targets["has_affinity"].sum() + 1e-8)
            else:
                value_loss = value_loss.mean()
            
            # Binary classification loss
            if "is_binder" in targets:
                prob_loss = F.binary_cross_entropy(
                    predictions["affinity_probability_binary"],
                    targets["is_binder"].float(),
                )
            else:
                prob_loss = torch.tensor(0.0, device=value_loss.device)
            
            total_loss = weights["value"] * value_loss + weights["prob"] * prob_loss
            
            return {
                "total_loss": total_loss,
                "value_loss": value_loss,
                "prob_loss": prob_loss,
            }


# =============================================================================
# NumPy Reference Implementation
# =============================================================================

class AffinityHeadNumPy:
    """NumPy reference implementation of affinity head."""
    
    def __init__(
        self,
        single_channel: int = 384,
        pair_channel: int = 128,
        hidden_dim: int = 256,
        num_gaussians: int = 50,
    ):
        self.single_channel = single_channel
        self.pair_channel = pair_channel
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        
        # Initialize weights
        self.single_proj = np.random.randn(single_channel, hidden_dim) * 0.02
        self.pair_proj = np.random.randn(pair_channel + num_gaussians, hidden_dim) * 0.02
        
        # Output heads
        self.affinity_head = np.random.randn(hidden_dim * 2, 1) * 0.01
        self.prob_head = np.random.randn(hidden_dim * 2, 1) * 0.01
        
        # Gaussian smearing parameters
        self.gaussian_offsets = np.linspace(0, 30, num_gaussians)
        self.gaussian_width = (self.gaussian_offsets[1] - self.gaussian_offsets[0])
    
    def gaussian_smearing(self, distances: np.ndarray) -> np.ndarray:
        """Apply Gaussian smearing to distances."""
        dist = distances[..., np.newaxis]
        coeff = -0.5 / self.gaussian_width ** 2
        return np.exp(coeff * (dist - self.gaussian_offsets) ** 2)
    
    def forward(
        self,
        single: np.ndarray,
        pair: np.ndarray,
        distances: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Forward pass."""
        # Gaussian smearing
        dist_features = self.gaussian_smearing(distances)
        
        # Concatenate pair features with distance
        pair_with_dist = np.concatenate([pair, dist_features], axis=-1)
        
        # Project features
        single_proj = single @ self.single_proj
        pair_proj = pair_with_dist @ self.pair_proj
        
        # Pool (mean)
        single_pooled = single_proj.mean(axis=0)
        pair_pooled = pair_proj.mean(axis=(0, 1))
        
        # Concatenate and predict
        combined = np.concatenate([single_pooled, pair_pooled])
        
        affinity_value = combined @ self.affinity_head
        affinity_prob = 1 / (1 + np.exp(-combined @ self.prob_head))  # Sigmoid
        
        return {
            "affinity_pred_value": affinity_value.squeeze(),
            "affinity_probability_binary": affinity_prob.squeeze(),
        }


def demonstrate_affinity_head():
    """Demonstrate affinity head."""
    print("Affinity Head Demonstration")
    print("=" * 50)
    
    # Create head
    head = AffinityHeadNumPy()
    
    # Generate dummy inputs
    num_res = 100
    single = np.random.randn(num_res, 384)
    pair = np.random.randn(num_res, num_res, 128)
    distances = np.random.rand(num_res, num_res) * 20  # 0-20 Å
    
    # Forward pass
    output = head.forward(single, pair, distances)
    
    print(f"Input shapes:")
    print(f"  Single: {single.shape}")
    print(f"  Pair: {pair.shape}")
    print(f"  Distances: {distances.shape}")
    print(f"\nOutputs:")
    print(f"  Affinity value (log10 scale): {output['affinity_pred_value']:.4f}")
    print(f"  Binding probability: {output['affinity_probability_binary']:.4f}")


if __name__ == "__main__":
    demonstrate_affinity_head()
