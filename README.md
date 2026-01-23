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

## 🔧 Fine-tuning Framework (NEW!)

We provide a comprehensive **fine-tuning framework** for adapting protein structure prediction models to downstream tasks.

👉 **[Full Fine-tuning Guide](finetuning/FINETUNING_GUIDE.md)**

### Supported Models

| Model | Framework | Fine-tuning Support |
|-------|-----------|-------------------|
| AlphaFold2 | JAX/Haiku | ✅ Full, Head-only, LoRA |
| AlphaFold3 | JAX/Haiku | ✅ Full, Head-only, LoRA |
| Boltz-1 | PyTorch | ✅ Full, LoRA, Adapter |
| Boltz-2 | PyTorch | ✅ Full, LoRA, Adapter |

### Supported Tasks (50+ Task Types)

<details>
<summary><b>💊 Drug Discovery</b></summary>

| Task | Outputs | Applications |
|------|---------|--------------|
| Binding Affinity | pKd, pIC50, ΔG, Ki | Lead optimization, SAR |
| Virtual Screening | Hit probability, ranking | HTS prioritization |
| ADMET | Absorption, metabolism, toxicity | Compound triage |

</details>

<details>
<summary><b>🔬 Protein Engineering</b></summary>

| Task | Outputs | Applications |
|------|---------|--------------|
| Stability | ΔΔG, Tm shift | Thermostabilization |
| Solubility | Expression score | Biomanufacturing |
| Mutation Effects | Fitness, pathogenicity | Variant analysis |

</details>

<details>
<summary><b>🧫 Antibody Design</b></summary>

| Task | Outputs | Applications |
|------|---------|--------------|
| Affinity Maturation | CDR binding, ΔΔG | Therapeutic optimization |
| Humanization | Humanness score | Drug development |
| Developability | Aggregation, viscosity | Manufacturing |

</details>

<details>
<summary><b>⚗️ Enzyme Engineering</b></summary>

| Task | Outputs | Applications |
|------|---------|--------------|
| Activity | kcat, Km, kcat/Km | Catalyst design |
| Specificity | Substrate profiles | Industrial enzymes |
| Directed Evolution | Fitness landscapes | Protein engineering |

</details>

<details>
<summary><b>🔗 Protein-Protein Interactions</b></summary>

| Task | Outputs | Applications |
|------|---------|--------------|
| PPI Binding | Kd, interface stability | Complex analysis |
| Interface Prediction | Contact residues | Structure analysis |
| Hot Spot Detection | ΔΔG per residue | PPI drug targets |

</details>

<details>
<summary><b>🧬 Function & Immunology</b></summary>

| Task | Outputs | Applications |
|------|---------|--------------|
| GO Terms | MF, BP, CC | Annotation |
| B-cell Epitopes | Epitope probability | Vaccine design |
| T-cell Epitopes | MHC binding | Immunotherapy |

</details>

### Quick Start

```python
from finetuning import TaskRegistry, create_finetuning_pipeline
from finetuning.modules import LoRAModule

# List all 50+ tasks
print(TaskRegistry.list_all_tasks())

# Get task recommendations
info = TaskRegistry.get_task_info("binding_affinity")
print(f"Recommended LoRA rank: {info.recommended_rank}")

# Create pipeline
pipeline = create_finetuning_pipeline(
    task="binding_affinity",
    base_model=model,
    strategy="lora",
)
```

### Module Structure

```
finetuning/
├── configs/           # Task configs (25+ types)
├── modules/           # LoRA, Adapter, Prompt Tuning
├── heads/             # 15+ specialized prediction heads
├── trainers/          # Training with DDP, AMP
├── data/              # 10+ dataset classes
└── registry.py        # Task registry & factory
```

---

## References

- [AlphaFold2 Paper](https://www.nature.com/articles/s41586-021-03819-2)
- [AlphaFold3 Paper](https://www.nature.com/articles/s41586-024-07487-w)
- [Boltz-1 Paper](https://doi.org/10.1101/2024.11.19.624167)
- [Boltz-2 Paper](https://doi.org/10.1101/2025.06.14.659707)

## License

Educational use only. Please refer to the original papers and repositories for licensing information.
