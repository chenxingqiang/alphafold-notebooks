"""Epitope and immunogenicity prediction heads.

Supports:
- B-cell epitope prediction (linear and conformational)
- T-cell epitope prediction (MHC binding)
- Immunogenicity assessment
- Antigenicity scoring
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
class EpitopeHeadConfig:
    """Configuration for epitope prediction heads."""

    # Input dimensions
    single_channel: int = 384
    pair_channel: int = 128

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    num_attention_heads: int = 4

    # B-cell epitopes
    min_epitope_length: int = 5
    max_epitope_length: int = 25

    # T-cell epitopes
    peptide_length_mhc1: int = 9
    peptide_length_mhc2: int = 15
    num_mhc_alleles: int = 100


if TORCH_AVAILABLE:

    class BcellEpitopeHead(nn.Module):
        """Head for B-cell epitope prediction."""

        def __init__(self, config: EpitopeHeadConfig):
            super().__init__()

            self.config = config

            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)
            self.pair_proj = nn.Linear(config.pair_channel, config.hidden_dim // 2)

            # Surface accessibility predictor
            self.accessibility_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

            # Flexibility predictor
            self.flexibility_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

            # Protrusion predictor
            self.protrusion_head = nn.Sequential(
                nn.Linear(config.hidden_dim + config.hidden_dim // 2, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

            # Combined epitope scorer
            self.epitope_scorer = nn.Sequential(
                nn.Linear(4, config.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 4, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            single: torch.Tensor,
            pair: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Predict B-cell epitopes."""
            # Project features
            single_proj = self.single_proj(single)
            pair_proj = self.pair_proj(pair)

            # Accessibility (from single representation)
            accessibility = self.accessibility_head(single_proj).squeeze(-1)

            # Flexibility
            flexibility = self.flexibility_head(single_proj).squeeze(-1)

            # Protrusion (needs pair context)
            pair_context = pair_proj.mean(dim=2)  # Aggregate pair info
            combined = torch.cat([single_proj, pair_context], dim=-1)
            protrusion = self.protrusion_head(combined).squeeze(-1)

            # Antigenicity (placeholder - would use sequence features)
            antigenicity = torch.sigmoid(single_proj.mean(dim=-1))

            # Combined score
            features = torch.stack([
                accessibility, flexibility, protrusion, antigenicity
            ], dim=-1)
            epitope_score = self.epitope_scorer(features).squeeze(-1)

            if mask is not None:
                epitope_score = epitope_score * mask

            return {
                "epitope_score": epitope_score,
                "accessibility": accessibility,
                "flexibility": flexibility,
                "protrusion": protrusion,
                "antigenicity": antigenicity,
            }


    class TcellEpitopeHead(nn.Module):
        """Head for T-cell epitope / MHC binding prediction."""

        def __init__(self, config: EpitopeHeadConfig):
            super().__init__()

            self.config = config

            # Input projection
            self.single_proj = nn.Linear(config.single_channel, config.hidden_dim)

            # Peptide encoder (for sliding window)
            self.peptide_encoder = nn.LSTM(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

            # MHC allele embeddings
            self.allele_embeddings = nn.Embedding(
                config.num_mhc_alleles, config.hidden_dim
            )

            # Binding predictor
            self.binding_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 3, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

            # Immunogenicity predictor
            self.immunogenicity_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 3, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(
            self,
            single: torch.Tensor,
            allele_ids: torch.Tensor,
            peptide_starts: Optional[torch.Tensor] = None,
            peptide_length: int = 9,
        ) -> Dict[str, torch.Tensor]:
            """Predict MHC binding and immunogenicity."""
            batch_size, num_res, _ = single.shape
            device = single.device

            # Project
            single_proj = self.single_proj(single)

            # Get allele embeddings
            allele_emb = self.allele_embeddings(allele_ids)  # [batch, hidden]

            if peptide_starts is not None:
                # Predict for specific peptides
                num_peptides = peptide_starts.shape[1]
                binding_scores = []
                immunogenicity_scores = []

                for i in range(num_peptides):
                    start = peptide_starts[:, i]
                    # Extract peptide features (simplified)
                    peptide_features = single_proj.mean(dim=1)  # Would need proper extraction

                    # LSTM encoding
                    lstm_out, _ = self.peptide_encoder(peptide_features.unsqueeze(1))
                    peptide_encoding = lstm_out.squeeze(1)

                    # Combine with allele
                    combined = torch.cat([peptide_encoding, allele_emb], dim=-1)
                    combined = torch.cat([combined, peptide_encoding * allele_emb], dim=-1)

                    binding = torch.sigmoid(self.binding_head(combined))
                    immunogenicity = torch.sigmoid(self.immunogenicity_head(combined))

                    binding_scores.append(binding)
                    immunogenicity_scores.append(immunogenicity)

                binding_scores = torch.cat(binding_scores, dim=-1)
                immunogenicity_scores = torch.cat(immunogenicity_scores, dim=-1)
            else:
                # Sliding window over all positions
                num_peptides = num_res - peptide_length + 1

                # Pool sequence representation
                seq_encoding = single_proj.mean(dim=1)

                # Combine with allele
                combined = torch.cat([seq_encoding, allele_emb], dim=-1)
                combined = torch.cat([combined, seq_encoding * allele_emb], dim=-1)

                binding_scores = torch.sigmoid(self.binding_head(combined))
                immunogenicity_scores = torch.sigmoid(self.immunogenicity_head(combined))

            return {
                "binding_score": binding_scores.squeeze(-1),
                "immunogenicity_score": immunogenicity_scores.squeeze(-1),
            }


    class ImmunogenicityHead(nn.Module):
        """Head for therapeutic protein immunogenicity assessment."""

        def __init__(self, config: EpitopeHeadConfig):
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

            # Risk predictors
            self.tcell_risk_head = nn.Linear(config.hidden_dim, 1)
            self.bcell_risk_head = nn.Linear(config.hidden_dim, 1)
            self.aggregation_risk_head = nn.Linear(config.hidden_dim, 1)
            self.humanness_head = nn.Linear(config.hidden_dim, 1)

            # Overall immunogenicity
            self.overall_head = nn.Sequential(
                nn.Linear(4, config.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 4, 1),
            )

        def forward(
            self,
            single: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Assess immunogenicity risk."""
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

            # Predict risk factors
            tcell_risk = torch.sigmoid(self.tcell_risk_head(pooled)).squeeze(-1)
            bcell_risk = torch.sigmoid(self.bcell_risk_head(pooled)).squeeze(-1)
            aggregation_risk = torch.sigmoid(self.aggregation_risk_head(pooled)).squeeze(-1)
            humanness = torch.sigmoid(self.humanness_head(pooled)).squeeze(-1)

            # Overall risk (low humanness = high risk)
            risk_features = torch.stack([
                tcell_risk, bcell_risk, aggregation_risk, 1 - humanness
            ], dim=-1)
            overall_risk = torch.sigmoid(self.overall_head(risk_features)).squeeze(-1)

            return {
                "overall_risk": overall_risk,
                "tcell_risk": tcell_risk,
                "bcell_risk": bcell_risk,
                "aggregation_risk": aggregation_risk,
                "humanness": humanness,
            }


# =============================================================================
# NumPy Reference
# =============================================================================

class BcellEpitopeNumPy:
    """NumPy reference for B-cell epitope prediction."""

    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim
        self.proj = np.random.randn(384, hidden_dim) * 0.02
        self.scorer = np.random.randn(hidden_dim, 1) * 0.02

    def forward(self, single: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward pass."""
        features = single @ self.proj
        scores = 1 / (1 + np.exp(-(features @ self.scorer).squeeze()))

        # Find epitope regions (consecutive high-scoring residues)
        threshold = 0.5
        epitope_mask = scores > threshold

        return {
            "epitope_scores": scores,
            "epitope_mask": epitope_mask,
            "num_epitope_residues": epitope_mask.sum(),
        }


def demonstrate_epitope_head():
    """Demonstrate epitope prediction head."""
    print("Epitope Prediction Head Demonstration")
    print("=" * 50)

    head = BcellEpitopeNumPy()

    num_res = 150
    single = np.random.randn(num_res, 384)

    output = head.forward(single)

    print(f"Protein: {num_res} residues")
    print(f"Predicted epitope residues: {output['num_epitope_residues']}")
    print(f"Score range: [{output['epitope_scores'].min():.4f}, {output['epitope_scores'].max():.4f}]")

    # Find epitope regions
    high_score_positions = np.where(output['epitope_mask'])[0]
    if len(high_score_positions) > 0:
        print(f"High-scoring positions: {high_score_positions[:10]}...")


if __name__ == "__main__":
    demonstrate_epitope_head()
