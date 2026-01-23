# Boltz Algorithm Index

This index provides a comprehensive mapping of all key algorithms from Boltz-1 and Boltz-2 to their source code implementations and explanation notebooks.

## Key Features of Boltz

| Model | Key Innovation |
|-------|---------------|
| **Boltz-1** | First fully open source model to approach AlphaFold3 accuracy |
| **Boltz-2** | Adds binding affinity prediction, approaching FEP accuracy 1000x faster |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Boltz Pipeline                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ Input        │    │    MSA       │    │   Atom Attention     │   │
│  │ Embedder     │    │   Module     │    │     Encoder          │   │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘   │
│         │                   │                       │               │
│         └─────────┬─────────┴───────────────────────┘               │
│                   ▼                                                  │
│         ┌─────────────────────┐                                     │
│         │  Pairformer Stack   │ (48 blocks)                         │
│         │  - Triangle Mult    │                                     │
│         │  - Triangle Attn    │                                     │
│         │  - Pair Averaging   │                                     │
│         └─────────┬───────────┘                                     │
│                   ▼                                                  │
│         ┌─────────────────────┐                                     │
│         │  Diffusion Module   │                                     │
│         │  - Diffusion Trans. │                                     │
│         │  - Atom Decoder     │                                     │
│         └─────────┬───────────┘                                     │
│                   ▼                                                  │
│         ┌─────────────────────────────────────┐                     │
│         │   Confidence Head + Affinity Head   │ (Boltz-2)           │
│         │   - pLDDT, pAE, pTM                 │                     │
│         │   - Binding Affinity                │                     │
│         └─────────────────────────────────────┘                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Legend
- ✅ Complete (includes overview, source code reference, NumPy implementation, and test examples)
- 🔄 In Progress
- 📋 Planned

## Algorithms by Category

### 1. Input Processing

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 1 | Input Embedder | [algorithm-01-InputEmbedder.ipynb](algorithm-01-InputEmbedder.ipynb) | `model/modules/trunk.py:InputEmbedder` | ✅ |
| 2 | Atom Attention Encoder | [algorithm-02-AtomAttentionEncoder.ipynb](algorithm-02-AtomAttentionEncoder.ipynb) | `model/modules/encoders.py` | ✅ |
| 3 | Relative Position Encoding | [algorithm-03-RelativePositionEncoding.ipynb](algorithm-03-RelativePositionEncoding.ipynb) | `model/layers/relative.py` | ✅ |

### 2. MSA Processing

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 4 | MSA Module | [algorithm-04-MSAModule.ipynb](algorithm-04-MSAModule.ipynb) | `model/modules/trunk.py:MSAModule` | ✅ |
| 5 | Outer Product Mean | [algorithm-05-OuterProductMean.ipynb](algorithm-05-OuterProductMean.ipynb) | `model/layers/outer_product_mean.py` | ✅ |
| 6 | Pair Weighted Averaging | [algorithm-06-PairWeightedAveraging.ipynb](algorithm-06-PairWeightedAveraging.ipynb) | `model/layers/pair_averaging.py` | ✅ |

### 3. Pairformer Stack

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 7 | Pairformer Module | [algorithm-07-PairformerModule.ipynb](algorithm-07-PairformerModule.ipynb) | `model/modules/trunk.py:PairformerModule` | ✅ |
| 8 | Triangle Multiplication | [algorithm-08-TriangleMultiplication.ipynb](algorithm-08-TriangleMultiplication.ipynb) | `model/layers/triangular_mult.py` | ✅ |
| 9 | Triangle Attention | [algorithm-09-TriangleAttention.ipynb](algorithm-09-TriangleAttention.ipynb) | `model/layers/triangular_attention/` | ✅ |
| 10 | Attention Pair Bias | [algorithm-10-AttentionPairBias.ipynb](algorithm-10-AttentionPairBias.ipynb) | `model/layers/attention.py` | ✅ |
| 11 | Transition Block | [algorithm-11-TransitionBlock.ipynb](algorithm-11-TransitionBlock.ipynb) | `model/layers/transition.py` | ✅ |

### 4. Diffusion Module

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 12 | Diffusion Module | [algorithm-12-DiffusionModule.ipynb](algorithm-12-DiffusionModule.ipynb) | `model/modules/diffusion.py` | ✅ |
| 13 | Diffusion Transformer | [algorithm-13-DiffusionTransformer.ipynb](algorithm-13-DiffusionTransformer.ipynb) | `model/modules/transformers.py` | ✅ |
| 14 | Fourier Embedding | [algorithm-14-FourierEmbedding.ipynb](algorithm-14-FourierEmbedding.ipynb) | `model/modules/encoders.py` | ✅ |
| 15 | Atom Attention Decoder | [algorithm-15-AtomAttentionDecoder.ipynb](algorithm-15-AtomAttentionDecoder.ipynb) | `model/modules/encoders.py` | ✅ |

### 5. Confidence & Affinity

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 16 | Confidence Module | [algorithm-16-ConfidenceModule.ipynb](algorithm-16-ConfidenceModule.ipynb) | `model/modules/confidence.py` | ✅ |
| 17 | Distogram Head | [algorithm-17-DistogramHead.ipynb](algorithm-17-DistogramHead.ipynb) | `model/modules/trunk.py:DistogramModule` | ✅ |
| 18 | Affinity Module | [algorithm-18-AffinityModule.ipynb](algorithm-18-AffinityModule.ipynb) | `model/modules/affinity.py` | ✅ |

### 6. Loss Functions

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 19 | Diffusion Loss | [algorithm-19-DiffusionLoss.ipynb](algorithm-19-DiffusionLoss.ipynb) | `model/loss/diffusion.py` | ✅ |
| 20 | Confidence Loss | [algorithm-20-ConfidenceLoss.ipynb](algorithm-20-ConfidenceLoss.ipynb) | `model/loss/confidence.py` | ✅ |

## Source Code References

- **Official Repository**: [jwohlwend/boltz](https://github.com/jwohlwend/boltz)
- **Boltz-1 Paper**: [bioRxiv 2024.11.19.624167](https://doi.org/10.1101/2024.11.19.624167)
- **Boltz-2 Paper**: [bioRxiv 2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707)
