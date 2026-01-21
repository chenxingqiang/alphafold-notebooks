"""Protein function prediction heads.

Supports:
- GO term prediction (MF, BP, CC)
- EC number prediction
- Subcellular localization
- Domain annotation
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
class FunctionHeadConfig:
    """Configuration for function prediction heads."""
    
    # Input dimensions
    single_channel: int = 384
    pair_channel: int = 128
    
    # Architecture
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    num_attention_heads: int = 8
    
    # GO terms
    num_go_mf: int = 1000   # Molecular Function terms
    num_go_bp: int = 2000   # Biological Process terms
    num_go_cc: int = 500    # Cellular Component terms
    
    # EC numbers
    num_ec_classes: int = 7  # 7 main classes
    num_ec_full: int = 5000  # Full EC numbers
    
    # Localization
    num_localizations: int = 10
    
    # Hierarchical prediction
    use_hierarchy: bool = True


if TORCH_AVAILABLE:
    
    class HierarchicalClassifier(nn.Module):
        """Hierarchical multi-label classifier for GO terms."""
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_classes: int,
            hierarchy_matrix: Optional[torch.Tensor] = None,
        ):
            super().__init__()
            
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            )
            
            # Hierarchy constraint matrix (child -> parents)
            if hierarchy_matrix is not None:
                self.register_buffer("hierarchy", hierarchy_matrix)
            else:
                self.hierarchy = None
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict with hierarchy constraints."""
            logits = self.classifier(x)
            probs = torch.sigmoid(logits)
            
            # Apply hierarchy: if child is predicted, parents must be predicted
            if self.hierarchy is not None:
                # Propagate probabilities up the hierarchy
                probs = torch.max(probs.unsqueeze(-1), 
                                  (probs.unsqueeze(-2) * self.hierarchy).max(dim=-1)[0])
            
            return logits, probs
    
    
    class GOPredictionHead(nn.Module):
        """Head for Gene Ontology term prediction."""
        
        def __init__(self, config: FunctionHeadConfig):
            super().__init__()
            
            self.config = config
            
            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            
            # Sequence encoder
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.hidden_dim * 4,
                    dropout=config.dropout,
                    batch_first=True,
                ),
                num_layers=config.num_layers,
            )
            
            # Attention pooling
            self.pool_attention = nn.Linear(config.hidden_dim, 1)
            
            # GO classifiers
            self.mf_classifier = HierarchicalClassifier(
                config.hidden_dim, config.hidden_dim, config.num_go_mf
            )
            self.bp_classifier = HierarchicalClassifier(
                config.hidden_dim, config.hidden_dim, config.num_go_bp
            )
            self.cc_classifier = HierarchicalClassifier(
                config.hidden_dim, config.hidden_dim, config.num_go_cc
            )
        
        def forward(
            self,
            single: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict GO terms."""
            # Project and encode
            features = self.single_proj(single)
            
            if mask is not None:
                src_key_padding_mask = ~mask
            else:
                src_key_padding_mask = None
            
            encoded = self.encoder(features, src_key_padding_mask=src_key_padding_mask)
            
            # Attention pooling
            attn_weights = self.pool_attention(encoded)
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = (encoded * attn_weights).sum(dim=1)
            
            # Predict GO terms
            mf_logits, mf_probs = self.mf_classifier(pooled)
            bp_logits, bp_probs = self.bp_classifier(pooled)
            cc_logits, cc_probs = self.cc_classifier(pooled)
            
            return {
                "mf_logits": mf_logits,
                "mf_probs": mf_probs,
                "bp_logits": bp_logits,
                "bp_probs": bp_probs,
                "cc_logits": cc_logits,
                "cc_probs": cc_probs,
            }
    
    
    class ECNumberHead(nn.Module):
        """Head for EC number prediction."""
        
        def __init__(self, config: FunctionHeadConfig):
            super().__init__()
            
            self.config = config
            
            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            
            # Encoder
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.hidden_dim * 4,
                    dropout=config.dropout,
                    batch_first=True,
                ),
                num_layers=2,
            )
            
            # Hierarchical EC classifiers
            # EC: X.X.X.X (4 levels)
            self.ec1_classifier = nn.Linear(config.hidden_dim, 7)   # 7 main classes
            self.ec2_classifier = nn.Linear(config.hidden_dim, 100)  # Subclass
            self.ec3_classifier = nn.Linear(config.hidden_dim, 300)  # Sub-subclass
            self.ec4_classifier = nn.Linear(config.hidden_dim, config.num_ec_full)  # Full EC
        
        def forward(
            self,
            single: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict EC numbers."""
            # Project and encode
            features = self.single_proj(single)
            
            if mask is not None:
                src_key_padding_mask = ~mask
            else:
                src_key_padding_mask = None
            
            encoded = self.encoder(features, src_key_padding_mask=src_key_padding_mask)
            
            # Pool
            if mask is not None:
                pooled = (encoded * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-8)
            else:
                pooled = encoded.mean(dim=1)
            
            # Hierarchical prediction
            ec1_logits = self.ec1_classifier(pooled)
            ec2_logits = self.ec2_classifier(pooled)
            ec3_logits = self.ec3_classifier(pooled)
            ec4_logits = self.ec4_classifier(pooled)
            
            return {
                "ec1_logits": ec1_logits,
                "ec2_logits": ec2_logits,
                "ec3_logits": ec3_logits,
                "ec4_logits": ec4_logits,
                "ec1_probs": F.softmax(ec1_logits, dim=-1),
                "is_enzyme_prob": F.softmax(ec1_logits, dim=-1)[:, 1:].sum(dim=-1),  # Non-class-0
            }
    
    
    class LocalizationHead(nn.Module):
        """Head for subcellular localization prediction."""
        
        def __init__(self, config: FunctionHeadConfig):
            super().__init__()
            
            self.config = config
            
            # Localization labels
            self.localizations = [
                "nucleus", "cytoplasm", "membrane", "mitochondria",
                "er", "golgi", "lysosome", "extracellular", 
                "peroxisome", "plastid"
            ]
            
            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            
            # Signal peptide detector (first 30 residues)
            self.signal_detector = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            
            # Transmembrane detector
            self.tm_detector = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            
            # Global classifier
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_dim + 2, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.num_localizations),
            )
        
        def forward(
            self,
            single: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict subcellular localization."""
            batch_size, num_res, _ = single.shape
            
            # Project
            features = self.single_proj(single)
            
            # Signal peptide (N-terminal)
            n_term = features[:, :30].mean(dim=1)
            has_signal = torch.sigmoid(self.signal_detector(n_term))
            
            # Transmembrane regions
            tm_scores = torch.sigmoid(self.tm_detector(features).squeeze(-1))
            has_tm = tm_scores.max(dim=1)[0].unsqueeze(-1)
            
            # Global pooling
            if mask is not None:
                pooled = (features * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-8)
            else:
                pooled = features.mean(dim=1)
            
            # Combine with signal features
            combined = torch.cat([pooled, has_signal, has_tm], dim=-1)
            
            # Classify
            logits = self.classifier(combined)
            probs = torch.sigmoid(logits)  # Multi-label
            
            return {
                "localization_logits": logits,
                "localization_probs": probs,
                "has_signal_peptide": has_signal.squeeze(-1),
                "has_transmembrane": has_tm.squeeze(-1),
                "tm_regions": tm_scores,
            }


# =============================================================================
# NumPy Reference
# =============================================================================

class GOPredictionNumPy:
    """NumPy reference for GO prediction."""
    
    def __init__(self, hidden_dim: int = 512, num_go_terms: int = 1000):
        self.hidden_dim = hidden_dim
        self.num_go_terms = num_go_terms
        
        self.proj = np.random.randn(384, hidden_dim) * 0.02
        self.classifier = np.random.randn(hidden_dim, num_go_terms) * 0.02
    
    def forward(self, single: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward pass."""
        features = single @ self.proj
        pooled = features.mean(axis=0)
        
        logits = pooled @ self.classifier
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        
        return {
            "logits": logits,
            "probs": probs,
            "top_terms": np.argsort(probs)[-10:][::-1],
        }


def demonstrate_function_head():
    """Demonstrate function prediction head."""
    print("Function Prediction Head Demonstration")
    print("=" * 50)
    
    head = GOPredictionNumPy()
    
    num_res = 200
    single = np.random.randn(num_res, 384)
    
    output = head.forward(single)
    
    print(f"Protein: {num_res} residues")
    print(f"Number of GO terms: {head.num_go_terms}")
    print(f"Top predicted terms: {output['top_terms']}")
    print(f"Top term probability: {output['probs'][output['top_terms'][0]]:.4f}")


if __name__ == "__main__":
    demonstrate_function_head()
