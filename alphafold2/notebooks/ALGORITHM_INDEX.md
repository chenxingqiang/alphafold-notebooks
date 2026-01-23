# AlphaFold2 Algorithm Index

This index provides a comprehensive mapping of all 32 algorithms from the AlphaFold2 supplementary materials to their source code implementations and explanation notebooks.

## Legend
- ✅ Complete (includes pseudocode image, source code reference, NumPy implementation, and test examples)

## Algorithms

| # | Algorithm | Notebook | Source File | Status |
|---|-----------|----------|-------------|--------|
| 1 | MSA Block Deletion | [algorithm-1-MSABlockDeletion.ipynb](algorithm-1-MSABlockDeletion.ipynb) | `model/tf/data_transforms.py` | ✅ |
| 2 | Inference | [algorithm-2-Inference.ipynb](algorithm-2-Inference.ipynb) | `model/modules.py:AlphaFold` | ✅ |
| 3 | Input Embedder | [algorithm-3-InputEmbedder.ipynb](algorithm-3-InputEmbedder.ipynb) | `model/modules.py:EmbeddingsAndEvoformer` | ✅ |
| 4 | relpos | [algorithm-4-relpos.ipynb](algorithm-4-relpos.ipynb) | `model/modules.py:EmbeddingsAndEvoformer` | ✅ |
| 5 | one_hot | [algorithm-5-one_hot.ipynb](algorithm-5-one_hot.ipynb) | `jax.nn.one_hot` | ✅ |
| 6 | Evoformer Stack | [algorithm-6-EvoformerStack.ipynb](algorithm-6-EvoformerStack.ipynb) | `model/modules.py:EvoformerIteration` | ✅ |
| 7 | MSA Row Attention with Pair Bias | [algorithm-7-MSARowAttentionWithPairBias.ipynb](algorithm-7-MSARowAttentionWithPairBias.ipynb) | `model/modules.py:MSARowAttentionWithPairBias` | ✅ |
| 8 | MSA Column Attention | [algorithm-8-MSAColumnAttention.ipynb](algorithm-8-MSAColumnAttention.ipynb) | `model/modules.py:MSAColumnAttention` | ✅ |
| 9 | MSA Transition | [algorithm-9-MSATransition.ipynb](algorithm-9-MSATransition.ipynb) | `model/modules.py:Transition` | ✅ |
| 10 | Outer Product Mean | [algorithm-10-OuterProductMean.ipynb](algorithm-10-OuterProductMean.ipynb) | `model/modules.py:OuterProductMean` | ✅ |
| 11 | Triangle Multiplication (Outgoing) | [algorithm-11-TriangleMultiplicationOutgoing.ipynb](algorithm-11-TriangleMultiplicationOutgoing.ipynb) | `model/modules.py:TriangleMultiplication` | ✅ |
| 12 | Triangle Multiplication (Incoming) | [algorithm-12-TriangleMultiplicationIncoming.ipynb](algorithm-12-TriangleMultiplicationIncoming.ipynb) | `model/modules.py:TriangleMultiplication` | ✅ |
| 13 | Triangle Attention (Starting Node) | [algorithm-13-TriangleAttentionStartingNode.ipynb](algorithm-13-TriangleAttentionStartingNode.ipynb) | `model/modules.py:TriangleAttention` | ✅ |
| 14 | Triangle Attention (Ending Node) | [algorithm-14-TriangleAttentionEndingNode.ipynb](algorithm-14-TriangleAttentionEndingNode.ipynb) | `model/modules.py:TriangleAttention` | ✅ |
| 15 | Pair Transition | [algorithm-15-PairTransition.ipynb](algorithm-15-PairTransition.ipynb) | `model/modules.py:Transition` | ✅ |
| 16 | Template Pair Stack | [algorithm-16-TemplatePairStack.ipynb](algorithm-16-TemplatePairStack.ipynb) | `model/modules.py:TemplatePairStack` | ✅ |
| 17 | Template Pointwise Attention | [algorithm-17-TemplatePointwiseAttention.ipynb](algorithm-17-TemplatePointwiseAttention.ipynb) | `model/modules.py:Attention` | ✅ |
| 18 | Extra MSA Stack | [algorithm-18-ExtraMsaStack.ipynb](algorithm-18-ExtraMsaStack.ipynb) | `model/modules.py:ExtraMsaStack` | ✅ |
| 19 | MSA Column Global Attention | [algorithm-19-MSAColumnGlobalAttention.ipynb](algorithm-19-MSAColumnGlobalAttention.ipynb) | `model/modules.py:GlobalAttention` | ✅ |
| 20 | Structure Module | [algorithm-20-StructureModule.ipynb](algorithm-20-StructureModule.ipynb) | `model/folding.py:StructureModule` | ✅ |
| 21 | Rigid from 3 Points | [algorithm-21-rigidFrom3Points.ipynb](algorithm-21-rigidFrom3Points.ipynb) | `model/r3.py:rigids_from_3_points` | ✅ |
| 22 | Invariant Point Attention | [algorithm-22-InvariantPointAttention.ipynb](algorithm-22-InvariantPointAttention.ipynb) | `model/folding.py:InvariantPointAttention` | ✅ |
| 23 | Backbone Update | [algorithm-23-BackboneUpdate.ipynb](algorithm-23-BackboneUpdate.ipynb) | `model/folding.py:FoldIteration` | ✅ |
| 24 | Compute All Atom Coordinates | [algorithm-24-computeAllAtomCoordinates.ipynb](algorithm-24-computeAllAtomCoordinates.ipynb) | `model/all_atom.py` | ✅ |
| 25 | makeRotX | [algorithm-25-makeRotX.ipynb](algorithm-25-makeRotX.ipynb) | `model/all_atom.py` | ✅ |
| 26 | Rename Symmetric Ground Truth Atoms | [algorithm-26-renameSymmetricGroundTruthAtoms.ipynb](algorithm-26-renameSymmetricGroundTruthAtoms.ipynb) | `model/folding.py:compute_renamed_ground_truth` | ✅ |
| 27 | Torsion Angle Loss | [algorithm-27-torsionAngleLoss.ipynb](algorithm-27-torsionAngleLoss.ipynb) | `model/folding.py:supervised_chi_loss` | ✅ |
| 28 | Compute FAPE | [algorithm-28-computeFAPE.ipynb](algorithm-28-computeFAPE.ipynb) | `model/all_atom.py:frame_aligned_point_error` | ✅ |
| 29 | Predict Per-Residue LDDT | [algorithm-29-predictPerResidueLDDT.ipynb](algorithm-29-predictPerResidueLDDT.ipynb) | `model/modules.py:PredictedLDDTHead` | ✅ |
| 30 | Recycling (Inference) | [algorithm-30-RecyclingInference.ipynb](algorithm-30-RecyclingInference.ipynb) | `model/modules.py:AlphaFold` | ✅ |
| 31 | Recycling (Training) | [algorithm-31-RecyclingTraining.ipynb](algorithm-31-RecyclingTraining.ipynb) | `model/modules.py:AlphaFold` | ✅ |
| 32 | Recycling Embedder | [algorithm-32-RecyclingEmbedder.ipynb](algorithm-32-RecyclingEmbedder.ipynb) | `model/modules.py:EmbeddingsAndEvoformer` | ✅ |

## Categories

### Data Preprocessing
- Algorithm 1: MSA Block Deletion

### Embedding
- Algorithm 3: Input Embedder
- Algorithm 4: relpos (Relative Position Encoding)
- Algorithm 5: one_hot

### Evoformer
- Algorithm 6: Evoformer Stack
- Algorithm 7: MSA Row Attention with Pair Bias
- Algorithm 8: MSA Column Attention
- Algorithm 9: MSA Transition
- Algorithm 10: Outer Product Mean
- Algorithm 11-12: Triangle Multiplication (Outgoing/Incoming)
- Algorithm 13-14: Triangle Attention (Starting/Ending Node)
- Algorithm 15: Pair Transition

### Templates
- Algorithm 16: Template Pair Stack
- Algorithm 17: Template Pointwise Attention

### Extra MSA
- Algorithm 18: Extra MSA Stack
- Algorithm 19: MSA Column Global Attention

### Structure Module
- Algorithm 20: Structure Module
- Algorithm 21: Rigid from 3 Points
- Algorithm 22: Invariant Point Attention (IPA)
- Algorithm 23: Backbone Update
- Algorithm 24: Compute All Atom Coordinates
- Algorithm 25: makeRotX

### Losses
- Algorithm 26: Rename Symmetric Ground Truth Atoms
- Algorithm 27: Torsion Angle Loss
- Algorithm 28: Compute FAPE
- Algorithm 29: Predict Per-Residue LDDT (pLDDT)

### Recycling
- Algorithm 30: Recycling (Inference)
- Algorithm 31: Recycling (Training)
- Algorithm 32: Recycling Embedder

### Main Pipeline
- Algorithm 2: Inference
