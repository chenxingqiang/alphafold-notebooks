"""Protein property prediction head.

Predicts various protein properties including:
- Thermodynamic stability (ΔG)
- Solubility
- Aggregation propensity
- Expression level
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Literal
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
class PropertyHeadConfig:
    """Configuration for property prediction head."""
    
    # Input dimensions
    single_channel: int = 384
    pair_channel: int = 128
    
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # Pooling
    pooling_strategy: Literal["mean", "max", "attention", "cls"] = "attention"
    num_attention_heads: int = 4
    
    # Output
    num_properties: int = 1
    output_type: Literal["regression", "classification"] = "regression"
    num_classes: int = 2  # For classification
    
    # Multi-task learning
    property_names: List[str] = None
    
    # Uncertainty estimation
    predict_uncertainty: bool = False


if TORCH_AVAILABLE:
    
    class PropertyHead(nn.Module):
        """Protein property prediction head.
        
        Uses learned attention to aggregate residue features into
        protein-level properties.
        """
        
        def __init__(self, config: PropertyHeadConfig):
            super().__init__()
            
            self.config = config
            
            # Feature projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            self.pair_proj = nn.Linear(config.pair_channel, config.hidden_dim)
            
            # Pooling mechanism
            if config.pooling_strategy == "attention":
                self.attention_weights = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    nn.Tanh(),
                    nn.Linear(config.hidden_dim // 2, 1),
                )
            elif config.pooling_strategy == "cls":
                self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
                nn.init.normal_(self.cls_token, std=0.02)
            
            # Pair feature aggregation
            self.pair_to_single = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            
            # MLP layers
            layers = []
            for i in range(config.num_layers - 1):
                layers.extend([
                    nn.Linear(config.hidden_dim * 2 if i == 0 else config.hidden_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ])
            self.mlp = nn.Sequential(*layers)
            
            # Output head(s)
            if config.output_type == "regression":
                self.output_head = nn.Linear(config.hidden_dim, config.num_properties)
                if config.predict_uncertainty:
                    self.uncertainty_head = nn.Linear(config.hidden_dim, config.num_properties)
            else:
                self.output_head = nn.Linear(config.hidden_dim, config.num_classes * config.num_properties)
        
        def _pool_single(
            self,
            single: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Pool single representation to protein-level."""
            if self.config.pooling_strategy == "mean":
                if mask is not None:
                    return (single * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                return single.mean(dim=1)
            
            elif self.config.pooling_strategy == "max":
                if mask is not None:
                    single = single.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                return single.max(dim=1)[0]
            
            elif self.config.pooling_strategy == "attention":
                # Compute attention weights
                attn_logits = self.attention_weights(single).squeeze(-1)  # [batch, seq_len]
                
                if mask is not None:
                    attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
                
                attn_weights = F.softmax(attn_logits, dim=-1)
                return torch.einsum("bs,bsd->bd", attn_weights, single)
            
            elif self.config.pooling_strategy == "cls":
                batch_size = single.shape[0]
                cls_token = self.cls_token.expand(batch_size, -1, -1)
                # Concatenate CLS token
                combined = torch.cat([cls_token, single], dim=1)
                # Return CLS output (simplified - in practice would use transformer)
                return combined[:, 0]
        
        def _aggregate_pair(
            self,
            pair: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Aggregate pair representation to single-level."""
            # Sum over one dimension
            if mask is not None:
                pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
                pair = pair * pair_mask.unsqueeze(-1)
                pair_agg = pair.sum(dim=2) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                pair_agg = pair.mean(dim=2)
            
            return self.pair_to_single(pair_agg)
        
        def forward(
            self,
            single: torch.Tensor,
            pair: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict protein properties.
            
            Args:
                single: Single representation [batch, num_res, single_channel]
                pair: Pair representation [batch, num_res, num_res, pair_channel]
                mask: Residue mask [batch, num_res]
            
            Returns:
                Dictionary with predictions
            """
            # Project features
            single_proj = self.single_proj(single)
            pair_proj = self.pair_proj(pair)
            
            # Aggregate pair to single
            pair_agg = self._aggregate_pair(pair_proj, mask)
            
            # Combine single and aggregated pair
            combined = single_proj + pair_agg
            
            # Pool to protein level
            protein_features = self._pool_single(combined, mask)
            
            # Duplicate for concatenation with pair pooling
            pair_pooled = self._pool_single(pair_agg, mask)
            combined_features = torch.cat([protein_features, pair_pooled], dim=-1)
            
            # MLP
            features = self.mlp(combined_features)
            
            # Output
            outputs = {"features": features}
            
            if self.config.output_type == "regression":
                outputs["predictions"] = self.output_head(features)
                
                if self.config.predict_uncertainty:
                    # Predict log variance
                    log_var = self.uncertainty_head(features)
                    outputs["uncertainty"] = torch.exp(0.5 * log_var)
            else:
                logits = self.output_head(features)
                logits = logits.view(-1, self.config.num_properties, self.config.num_classes)
                outputs["logits"] = logits
                outputs["predictions"] = F.softmax(logits, dim=-1)
            
            return outputs
        
        def compute_loss(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Compute property prediction loss."""
            if self.config.output_type == "regression":
                if self.config.predict_uncertainty:
                    # Negative log likelihood with learned uncertainty
                    log_var = torch.log(predictions["uncertainty"] ** 2 + 1e-8)
                    loss = 0.5 * (
                        torch.exp(-log_var) * (predictions["predictions"] - targets) ** 2
                        + log_var
                    )
                else:
                    loss = F.mse_loss(predictions["predictions"], targets, reduction="none")
                
                if mask is not None:
                    loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
                else:
                    loss = loss.mean()
                
                return {"loss": loss, "mse": loss}
            else:
                loss = F.cross_entropy(
                    predictions["logits"].view(-1, self.config.num_classes),
                    targets.view(-1).long(),
                )
                return {"loss": loss, "ce": loss}


# =============================================================================
# NumPy Reference Implementation
# =============================================================================

class PropertyHeadNumPy:
    """NumPy reference implementation."""
    
    def __init__(
        self,
        single_channel: int = 384,
        hidden_dim: int = 256,
        num_properties: int = 1,
    ):
        self.single_channel = single_channel
        self.hidden_dim = hidden_dim
        self.num_properties = num_properties
        
        # Initialize weights
        self.single_proj = np.random.randn(single_channel, hidden_dim) * 0.02
        self.attention_w1 = np.random.randn(hidden_dim, hidden_dim // 2) * 0.02
        self.attention_w2 = np.random.randn(hidden_dim // 2, 1) * 0.02
        self.output_w = np.random.randn(hidden_dim, num_properties) * 0.02
    
    def forward(self, single: np.ndarray) -> np.ndarray:
        """Forward pass with attention pooling."""
        # Project
        h = single @ self.single_proj  # [seq_len, hidden]
        
        # Attention pooling
        attn_h = np.tanh(h @ self.attention_w1)
        attn_logits = attn_h @ self.attention_w2  # [seq_len, 1]
        attn_weights = np.exp(attn_logits - attn_logits.max())
        attn_weights = attn_weights / attn_weights.sum()
        
        # Weighted sum
        pooled = (h * attn_weights).sum(axis=0)  # [hidden]
        
        # Output
        return pooled @ self.output_w


def demonstrate_property_head():
    """Demonstrate property prediction."""
    print("Property Prediction Head Demonstration")
    print("=" * 50)
    
    head = PropertyHeadNumPy(num_properties=3)
    
    # Generate dummy input
    num_res = 150
    single = np.random.randn(num_res, 384)
    
    # Predict
    properties = head.forward(single)
    
    print(f"Input shape: {single.shape}")
    print(f"Output shape: {properties.shape}")
    print(f"\nPredicted properties:")
    print(f"  Stability (ΔG): {properties[0]:.4f}")
    print(f"  Solubility: {properties[1]:.4f}")
    print(f"  Expression: {properties[2]:.4f}")


if __name__ == "__main__":
    demonstrate_property_head()
