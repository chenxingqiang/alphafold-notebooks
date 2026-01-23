# Protein Structure Prediction Algorithm Notebooks

A comprehensive educational resource for understanding the core algorithms of modern protein structure prediction models, including **AlphaFold2**, **AlphaFold3**, **Boltz-1**, and **Boltz-2**.

## Overview

This repository provides detailed Jupyter notebooks that explain the key algorithms from each model's architecture, with:

- **Pseudocode explanations** from original papers
- **NumPy implementations** for educational purposes
- **Source code references** to official implementations
- **Test examples** to verify understanding

## Repository Structure

```
alphafold-notebooks/
├── alphafold2/           # AlphaFold2 (32 algorithms)
│   ├── notebooks/        # Algorithm explanation notebooks
│   ├── source/           # Source code reference (local copy)
│   ├── ref-src/          # External reference repositories (submodules)
│   ├── references/       # Related papers
│   ├── presentations/    # Slides and presentations
│   └── applications/     # Application examples
│
├── alphafold3/           # AlphaFold3 (23 algorithms)
│   ├── notebooks/
│   └── ref-src/          # External reference repositories (submodules)
│
├── boltz/                # Boltz-1 (20 algorithms)
│   ├── notebooks/
│   └── ref-src/          # External reference repositories (submodules)
│
├── boltz2/               # Boltz-2 (10 new algorithms)
│   └── notebooks/
│
└── assets/
    └── images/           # Shared image resources
```

## Model Comparison

| Model | Key Architecture | Algorithms Covered |
|-------|-----------------|-------------------|
| **AlphaFold2** | Evoformer + IPA Structure Module | 32 |
| **AlphaFold3** | MSA Module + Pairformer + Diffusion | 23 |
| **Boltz-1** | Pairformer + Diffusion (open source) | 20 |
| **Boltz-2** | + Affinity Prediction (binding affinity) | 10 (new) |

## Reference Source Code (Git Submodules)

### AlphaFold2 References

| Repository | Description | URL |
|------------|-------------|-----|
| **alphafold-official** | DeepMind's official AlphaFold2 | [deepmind/alphafold](https://github.com/deepmind/alphafold) |
| **openfold** | Trainable PyTorch reproduction | [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold) |
| **colabfold** | Fast AlphaFold on Google Colab | [sokrypton/ColabFold](https://github.com/sokrypton/ColabFold) |
| **mmseqs2** | Fast sequence search tool | [soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2) |
| **hh-suite** | HMM-based sequence search | [soedinglab/hh-suite](https://github.com/soedinglab/hh-suite) |
| **trRosetta2** | Alternative structure prediction | [RosettaCommons/trRosetta2](https://github.com/RosettaCommons/trRosetta2) |
| **esm** | Meta's protein language models | [facebookresearch/esm](https://github.com/facebookresearch/esm) |
| **unirep** | UniRep protein representation | [churchlab/UniRep](https://github.com/churchlab/UniRep) |
| **seqvec** | ELMo for proteins | [rostlab/SeqVec](https://github.com/rostlab/SeqVec) |

### AlphaFold3 References

| Repository | Description | URL |
|------------|-------------|-----|
| **alphafold3-official** | DeepMind's official AlphaFold3 | [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) |
| **alphafold3-pytorch** | PyTorch reproduction by lucidrains | [lucidrains/alphafold3-pytorch](https://github.com/lucidrains/alphafold3-pytorch) |
| **alphafold3-walkthrough** | Architecture walkthrough | [shenyichong/alphafold3-architecture-walkthrough](https://github.com/shenyichong/alphafold3-architecture-walkthrough) |

### Boltz References

| Repository | Description | URL |
|------------|-------------|-----|
| **boltz-official** | Official Boltz-1 & Boltz-2 | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |
| **boltzina** | Boltz for virtual screening | [ohuelab/boltzina](https://github.com/ohuelab/boltzina) |

### Clone with Submodules

```bash
# Clone with all submodules
git clone --recursive https://github.com/your-repo/alphafold-notebooks.git

# Or initialize submodules after clone
git submodule update --init --recursive
```

## Quick Start

Each model directory contains an `ALGORITHM_INDEX.md` that provides:
- Complete algorithm listing with links
- Category-based organization
- Source code file references
- Completion status

Start here:
- [AlphaFold2 Algorithm Index](alphafold2/notebooks/ALGORITHM_INDEX.md)
- [AlphaFold3 Algorithm Index](alphafold3/notebooks/ALGORITHM_INDEX.md)
- [Boltz-1 Algorithm Index](boltz/notebooks/ALGORITHM_INDEX.md)
- [Boltz-2 Algorithm Index](boltz2/notebooks/ALGORITHM_INDEX.md)

## Key Topics Covered

### Representation Learning
- MSA Processing (Row/Column Attention)
- Outer Product Mean
- Triangle Multiplication & Attention
- Pairformer Stack

### Structure Prediction
- Invariant Point Attention (IPA) - AlphaFold2
- Diffusion Transformer - AlphaFold3/Boltz
- Atom Cross Attention

### Confidence & Loss
- pLDDT, pAE, pTM metrics
- FAPE Loss
- Diffusion Loss
- Binding Affinity (Boltz-2)

## References

- [AlphaFold2 Paper](https://www.nature.com/articles/s41586-021-03819-2)
- [AlphaFold3 Paper](https://www.nature.com/articles/s41586-024-07487-w)
- [Boltz-1 Paper](https://doi.org/10.1101/2024.11.19.624167)
- [Boltz-2 Paper](https://doi.org/10.1101/2025.06.14.659707)

## License

Educational use only. Please refer to the original papers and repositories for licensing information.
