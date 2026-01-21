"""Enzyme engineering prediction heads.

Supports:
- Enzyme activity prediction (kcat, Km, kcat/Km)
- Substrate specificity prediction
- Thermostability prediction
- Directed evolution guidance
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
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
class EnzymeHeadConfig:
    """Configuration for enzyme prediction heads."""
    
    # Input dimensions
    single_channel: int = 384
    pair_channel: int = 128
    substrate_dim: int = 256  # Molecular fingerprint dimension
    
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    num_attention_heads: int = 4
    
    # Active site
    active_site_radius: float = 8.0
    max_active_site_residues: int = 50
    
    # Output
    predict_kcat: bool = True
    predict_km: bool = True
    predict_stability: bool = True


if TORCH_AVAILABLE:
    
    class ActiveSiteEncoder(nn.Module):
        """Encode active site residues."""
        
        def __init__(self, config: EnzymeHeadConfig):
            super().__init__()
            
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            )
            
            self.norm1 = nn.LayerNorm(config.hidden_dim)
            self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        def forward(
            self,
            residue_features: torch.Tensor,
            active_site_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Encode active site with self-attention."""
            # Self-attention on active site residues
            attn_out, _ = self.attention(
                residue_features, residue_features, residue_features,
                key_padding_mask=~active_site_mask,
            )
            x = self.norm1(residue_features + attn_out)
            x = self.norm2(x + self.mlp(x))
            
            # Pool active site
            x = x * active_site_mask.unsqueeze(-1)
            pooled = x.sum(dim=1) / (active_site_mask.sum(dim=1, keepdim=True) + 1e-8)
            
            return pooled
    
    
    class SubstrateEncoder(nn.Module):
        """Encode substrate molecules."""
        
        def __init__(self, config: EnzymeHeadConfig):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(config.substrate_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        
        def forward(self, substrate_features: torch.Tensor) -> torch.Tensor:
            """Encode substrate fingerprint or graph features."""
            return self.encoder(substrate_features)
    
    
    class EnzymeActivityHead(nn.Module):
        """Head for enzyme kinetic parameter prediction."""
        
        def __init__(self, config: EnzymeHeadConfig):
            super().__init__()
            
            self.config = config
            
            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            
            # Active site encoder
            self.active_site_encoder = ActiveSiteEncoder(config)
            
            # Substrate encoder
            self.substrate_encoder = SubstrateEncoder(config)
            
            # Enzyme-substrate interaction
            self.interaction_layer = nn.Sequential(
                nn.Linear(config.hidden_dim * 3, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            
            # Kinetic parameter heads
            self.kcat_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            
            self.km_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
        
        def forward(
            self,
            single: torch.Tensor,
            active_site_mask: torch.Tensor,
            substrate_features: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict enzyme kinetic parameters."""
            # Project enzyme features
            single_proj = self.single_proj(single)
            
            # Encode active site
            active_site_encoding = self.active_site_encoder(single_proj, active_site_mask)
            
            # Encode substrate
            substrate_encoding = self.substrate_encoder(substrate_features)
            
            # Global enzyme features
            if mask is not None:
                global_encoding = (single_proj * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-8)
            else:
                global_encoding = single_proj.mean(dim=1)
            
            # Combine
            combined = torch.cat([
                active_site_encoding,
                substrate_encoding,
                global_encoding,
            ], dim=-1)
            
            features = self.interaction_layer(combined)
            
            # Predict kinetic parameters (log scale)
            log_kcat = self.kcat_head(features).squeeze(-1)
            log_km = self.km_head(features).squeeze(-1)
            
            # Catalytic efficiency
            log_kcat_km = log_kcat - log_km
            
            return {
                "log_kcat": log_kcat,
                "log_km": log_km,
                "log_kcat_km": log_kcat_km,
                "kcat": torch.exp(log_kcat),
                "km": torch.exp(log_km),
                "kcat_km": torch.exp(log_kcat_km),
            }
    
    
    class EnzymeSpecificityHead(nn.Module):
        """Head for substrate specificity prediction."""
        
        def __init__(self, config: EnzymeHeadConfig, num_substrates: int = 100):
            super().__init__()
            
            self.config = config
            self.num_substrates = num_substrates
            
            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            
            # Substrate embedding
            self.substrate_embeddings = nn.Embedding(num_substrates, config.hidden_dim)
            
            # Matching network
            self.matcher = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            
            # Activity predictor
            self.activity_head = nn.Linear(config.hidden_dim, 1)
        
        def forward(
            self,
            single: torch.Tensor,
            active_site_mask: torch.Tensor,
            substrate_ids: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict activity on different substrates."""
            batch_size = single.shape[0]
            device = single.device
            
            # Project and pool enzyme
            single_proj = self.single_proj(single)
            enzyme_encoding = (single_proj * active_site_mask.unsqueeze(-1)).sum(1)
            enzyme_encoding = enzyme_encoding / (active_site_mask.sum(1, keepdim=True) + 1e-8)
            
            if substrate_ids is not None:
                # Predict for specific substrates
                substrate_encoding = self.substrate_embeddings(substrate_ids)
                combined = torch.cat([
                    enzyme_encoding.unsqueeze(1).expand(-1, substrate_ids.shape[1], -1),
                    substrate_encoding,
                ], dim=-1)
                features = self.matcher(combined)
                activity = torch.sigmoid(self.activity_head(features).squeeze(-1))
                
                return {"activity": activity}
            else:
                # Predict for all substrates
                all_substrates = self.substrate_embeddings.weight  # [num_substrates, hidden]
                
                # Expand for batch
                enzyme_exp = enzyme_encoding.unsqueeze(1).expand(-1, self.num_substrates, -1)
                substrate_exp = all_substrates.unsqueeze(0).expand(batch_size, -1, -1)
                
                combined = torch.cat([enzyme_exp, substrate_exp], dim=-1)
                features = self.matcher(combined)
                activity = torch.sigmoid(self.activity_head(features).squeeze(-1))
                
                return {"activity_profile": activity}
    
    
    class EnzymeEvolutionHead(nn.Module):
        """Head for guiding directed evolution."""
        
        def __init__(self, config: EnzymeHeadConfig, num_amino_acids: int = 20):
            super().__init__()
            
            self.config = config
            self.num_amino_acids = num_amino_acids
            
            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            
            # Position-wise mutation scoring
            self.mutation_scorer = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, num_amino_acids),
            )
            
            # Fitness predictor
            self.fitness_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
        
        def forward(
            self,
            single: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Score mutations and predict fitness."""
            # Project
            single_proj = self.single_proj(single)
            
            # Per-position mutation scores
            mutation_logits = self.mutation_scorer(single_proj)  # [batch, num_res, 20]
            mutation_probs = F.softmax(mutation_logits, dim=-1)
            
            # Global fitness
            if mask is not None:
                pooled = (single_proj * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-8)
            else:
                pooled = single_proj.mean(dim=1)
            
            fitness = self.fitness_head(pooled).squeeze(-1)
            
            return {
                "mutation_logits": mutation_logits,
                "mutation_probs": mutation_probs,
                "predicted_fitness": fitness,
            }


# =============================================================================
# NumPy Reference
# =============================================================================

class EnzymeActivityNumPy:
    """NumPy reference for enzyme activity prediction."""
    
    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim
        self.proj = np.random.randn(384, hidden_dim) * 0.02
        self.kcat_head = np.random.randn(hidden_dim, 1) * 0.02
        self.km_head = np.random.randn(hidden_dim, 1) * 0.02
    
    def forward(
        self,
        single: np.ndarray,
        active_site_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Forward pass."""
        features = single @ self.proj
        
        # Active site pooling
        active_features = features * active_site_mask[:, np.newaxis]
        pooled = active_features.sum(axis=0) / (active_site_mask.sum() + 1e-8)
        
        # Predict
        log_kcat = pooled @ self.kcat_head
        log_km = pooled @ self.km_head
        
        return {
            "log_kcat": log_kcat.squeeze(),
            "log_km": log_km.squeeze(),
            "kcat": np.exp(log_kcat.squeeze()),
            "km": np.exp(log_km.squeeze()),
        }


def demonstrate_enzyme_head():
    """Demonstrate enzyme head functionality."""
    print("Enzyme Head Demonstration")
    print("=" * 50)
    
    head = EnzymeActivityNumPy()
    
    # Enzyme structure
    num_res = 300
    single = np.random.randn(num_res, 384)
    
    # Active site (e.g., residues 100-120)
    active_site_mask = np.zeros(num_res)
    active_site_mask[100:120] = 1
    
    output = head.forward(single, active_site_mask)
    
    print(f"Enzyme: {num_res} residues")
    print(f"Active site: {int(active_site_mask.sum())} residues")
    print(f"Predicted log(kcat): {output['log_kcat']:.4f}")
    print(f"Predicted log(Km): {output['log_km']:.4f}")
    print(f"Predicted kcat: {output['kcat']:.4f} s^-1")
    print(f"Predicted Km: {output['km']:.4f} M")


if __name__ == "__main__":
    demonstrate_enzyme_head()
