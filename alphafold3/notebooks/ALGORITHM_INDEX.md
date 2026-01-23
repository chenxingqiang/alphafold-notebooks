# AlphaFold3 Algorithm Index

This index provides a comprehensive mapping of all key algorithms from AlphaFold3 to their source code implementations and explanation notebooks.

## Key Differences from AlphaFold2

| Aspect | AlphaFold2 | AlphaFold3 |
|--------|------------|------------|
| Structure Prediction | IPA + Backbone Update | Diffusion Transformer |
| MSA Processing | Full Evoformer (48 blocks) | Simplified MSA Module (4 blocks) |
| Pair Processing | Evoformer | Pairformer (48 blocks) |
| Atom Representation | Implicit (torsion angles) | Explicit (per-atom features) |
| Output | Single structure | Ensemble via diffusion |

## Legend
- ✅ Complete (includes overview, source code reference, NumPy implementation, and test examples)
- 🔄 In Progress
- 📋 Planned

## Algorithms by Category

### 1. Input Preparation

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 1 | MSA Features | [algorithm-01-MSAFeatures.ipynb](algorithm-01-MSAFeatures.ipynb) | `model/features.py` | ✅ |
| 2 | Template Embedder | [algorithm-02-TemplateEmbedder.ipynb](algorithm-02-TemplateEmbedder.ipynb) | `model/network/template_modules.py` | ✅ |
| 3 | Atom Transformer | [algorithm-03-AtomTransformer.ipynb](algorithm-03-AtomTransformer.ipynb) | `model/network/atom_cross_attention.py` | ✅ |
| 4 | Relative Position Encoding | [algorithm-04-RelativePositionEncoding.ipynb](algorithm-04-RelativePositionEncoding.ipynb) | `model/network/featurization.py` | ✅ |

### 2. Representation Learning - MSA Module

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 5 | Outer Product Mean | [algorithm-05-OuterProductMean.ipynb](algorithm-05-OuterProductMean.ipynb) | `model/network/modules.py` | ✅ |
| 6 | MSA Row Attention | [algorithm-06-MSARowAttention.ipynb](algorithm-06-MSARowAttention.ipynb) | `model/network/modules.py:MSAAttention` | ✅ |
| 7 | MSA Transition | [algorithm-07-MSATransition.ipynb](algorithm-07-MSATransition.ipynb) | `model/network/modules.py:TransitionBlock` | ✅ |

### 3. Representation Learning - Pairformer

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 8 | Pairformer Stack | [algorithm-08-PairformerStack.ipynb](algorithm-08-PairformerStack.ipynb) | `model/network/modules.py:PairFormerIteration` | ✅ |
| 9 | Triangle Multiplication (Outgoing) | [algorithm-09-TriangleMultiplicationOutgoing.ipynb](algorithm-09-TriangleMultiplicationOutgoing.ipynb) | `model/network/modules.py` | ✅ |
| 10 | Triangle Multiplication (Incoming) | [algorithm-10-TriangleMultiplicationIncoming.ipynb](algorithm-10-TriangleMultiplicationIncoming.ipynb) | `model/network/modules.py` | ✅ |
| 11 | Triangle Attention (Starting Node) | [algorithm-11-TriangleAttentionStartingNode.ipynb](algorithm-11-TriangleAttentionStartingNode.ipynb) | `model/network/modules.py` | ✅ |
| 12 | Triangle Attention (Ending Node) | [algorithm-12-TriangleAttentionEndingNode.ipynb](algorithm-12-TriangleAttentionEndingNode.ipynb) | `model/network/modules.py` | ✅ |
| 13 | Single Attention with Pair Bias | [algorithm-13-SingleAttentionWithPairBias.ipynb](algorithm-13-SingleAttentionWithPairBias.ipynb) | `model/network/modules.py` | ✅ |
| 14 | Pair Transition | [algorithm-14-PairTransition.ipynb](algorithm-14-PairTransition.ipynb) | `model/network/modules.py:TransitionBlock` | ✅ |

### 4. Structure Prediction - Diffusion

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 15 | Diffusion Module Overview | [algorithm-15-DiffusionModule.ipynb](algorithm-15-DiffusionModule.ipynb) | `model/network/diffusion_head.py` | ✅ |
| 16 | Adaptive LayerNorm | [algorithm-16-AdaptiveLayerNorm.ipynb](algorithm-16-AdaptiveLayerNorm.ipynb) | `model/network/diffusion_transformer.py` | ✅ |
| 17 | Diffusion Transformer | [algorithm-17-DiffusionTransformer.ipynb](algorithm-17-DiffusionTransformer.ipynb) | `model/network/diffusion_transformer.py` | ✅ |
| 18 | Atom Cross Attention | [algorithm-18-AtomCrossAttention.ipynb](algorithm-18-AtomCrossAttention.ipynb) | `model/network/atom_cross_attention.py` | ✅ |
| 19 | Noise Level Embedding | [algorithm-19-NoiseLevelEmbedding.ipynb](algorithm-19-NoiseLevelEmbedding.ipynb) | `model/network/noise_level_embeddings.py` | ✅ |

### 5. Confidence & Loss

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 20 | Distogram Head | [algorithm-20-DistogramHead.ipynb](algorithm-20-DistogramHead.ipynb) | `model/network/distogram_head.py` | ✅ |
| 21 | Confidence Head | [algorithm-21-ConfidenceHead.ipynb](algorithm-21-ConfidenceHead.ipynb) | `model/network/confidence_head.py` | ✅ |
| 22 | Diffusion Loss | [algorithm-22-DiffusionLoss.ipynb](algorithm-22-DiffusionLoss.ipynb) | `model/scoring/scoring.py` | ✅ |
| 23 | Smooth LDDT | [algorithm-23-SmoothLDDT.ipynb](algorithm-23-SmoothLDDT.ipynb) | `model/confidences.py` | ✅ |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AlphaFold3 Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ MSA Features │    │   Template   │    │  Atom-level Features │   │
│  │              │    │   Embedder   │    │   (Atom Transformer) │   │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘   │
│         │                   │                       │               │
│         └─────────┬─────────┴───────────────────────┘               │
│                   ▼                                                  │
│         ┌─────────────────────┐                                     │
│         │     MSA Module      │ (4 blocks)                          │
│         │  - Outer Product    │                                     │
│         │  - MSA Attention    │                                     │
│         └─────────┬───────────┘                                     │
│                   ▼                                                  │
│         ┌─────────────────────┐                                     │
│         │   Pairformer Stack  │ (48 blocks)                         │
│         │  - Triangle Mult    │                                     │
│         │  - Triangle Attn    │                                     │
│         │  - Single Attn      │                                     │
│         └─────────┬───────────┘                                     │
│                   ▼                                                  │
│         ┌─────────────────────┐                                     │
│         │  Diffusion Module   │ (200 steps inference)               │
│         │  - Atom Cross Attn  │                                     │
│         │  - Diffusion Trans. │                                     │
│         │  - Adaptive LN      │                                     │
│         └─────────┬───────────┘                                     │
│                   ▼                                                  │
│         ┌─────────────────────┐                                     │
│         │   Confidence Head   │                                     │
│         │  - pLDDT, pAE, pTM  │                                     │
│         └─────────────────────┘                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Source Code References

- **Official Repository**: [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)
- **PyTorch Implementation**: [lucidrains/alphafold3-pytorch](https://github.com/lucidrains/alphafold3-pytorch)
- **Architecture Walkthrough**: [shenyichong/alphafold3-architecture-walkthrough](https://github.com/shenyichong/alphafold3-architecture-walkthrough)
