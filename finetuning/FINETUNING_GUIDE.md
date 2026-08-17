# Fine-tuning Guide for Protein Structure Prediction Models

Comprehensive fine-tuning support for AlphaFold2, AlphaFold3, Boltz-1, and Boltz-2.

**Inspired by ProteinBase.com business logic** - Supporting all major protein analysis tasks.

## Table of Contents

1. [Supported Tasks Overview](#supported-tasks-overview)
2. [Task Categories](#task-categories)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Examples by Task](#examples-by-task)
6. [Best Practices](#best-practices)

---

## Supported Tasks Overview

| Category | Tasks | Applications |
|----------|-------|--------------|
| **Drug Discovery** | Binding Affinity, Virtual Screening, ADMET | Lead optimization, Hit identification |
| **Protein Engineering** | Stability, Solubility, Mutation Effects | Enzyme optimization, Therapeutic proteins |
| **Antibody Design** | Affinity Maturation, Humanization, Developability | Biologics development |
| **Enzyme Engineering** | Activity, Specificity, Directed Evolution | Industrial enzymes, Biocatalysis |
| **Protein-Protein Interaction** | Binding, Interface, Hot Spots | Drug targets, Signaling pathways |
| **Function Prediction** | GO Terms, EC Numbers, Localization | Annotation, Discovery |
| **Immunology** | B-cell Epitopes, T-cell Epitopes, Immunogenicity | Vaccine design, Therapeutic safety |

---

## Task Categories

### 1. Drug Discovery

#### Binding Affinity Prediction
Predict protein-ligand binding strength (pKd, pIC50, ΔG).

```python
from finetuning.configs import BindingAffinityConfig
from finetuning.heads import AffinityHead

config = BindingAffinityConfig(
    affinity_type="pic50",
    use_pocket_features=True,
    ligand_representation="graph",
)
```

**Datasets**: PDBbind, BindingDB, ChEMBL

#### Virtual Screening
Rank compounds by predicted binding.

```python
from finetuning.configs import VirtualScreeningConfig

config = VirtualScreeningConfig(
    activity_threshold=6.0,  # pIC50 threshold
    use_decoys=True,
)
```

**Metrics**: AUROC, AUPRC, Enrichment Factors (EF1%, EF5%)

---

### 2. Protein Engineering

#### Stability Prediction
Predict thermodynamic stability and thermal stability.

```python
from finetuning.configs import StabilityConfig

config = StabilityConfig(
    stability_type="ddg",  # or "tm", "t50"
    use_evolutionary_features=True,
)
```

**Datasets**: ProTherm, FireProtDB, Megascale

#### Mutation Effect Prediction
Predict ΔΔG, fitness, or pathogenicity.

```python
from finetuning.configs import MutationEffectConfig

config = MutationEffectConfig(
    effect_type="ddg",
    max_mutations=10,
    predict_epistasis=False,
)
```

**Datasets**: ProteinGym, DMS datasets, ClinVar

---

### 3. Antibody Design

#### Affinity Maturation
Optimize antibody binding affinity.

```python
from finetuning.configs import AntibodyOptimizationConfig
from finetuning.heads import AntibodyAffinityHead

config = AntibodyOptimizationConfig(
    optimization_target="affinity",
    optimize_regions=["cdr_h3", "cdr_l3"],
    preserve_framework=True,
)

head = AntibodyAffinityHead(AntibodyHeadConfig())
```

#### Humanization
Humanize non-human antibodies while preserving binding.

```python
from finetuning.configs import HumanizationConfig
from finetuning.heads import HumannessHead

config = HumanizationConfig(
    method="cdr_grafting",
    humanness_threshold=0.8,
)
```

#### Developability Assessment
Predict manufacturability risks.

```python
from finetuning.configs import AntibodyDevelopabilityConfig
from finetuning.heads import DevelopabilityHead

config = AntibodyDevelopabilityConfig(
    properties=[
        "aggregation_propensity",
        "viscosity",
        "immunogenicity",
        "expression",
    ],
)
```

**Output Properties**:
- Aggregation propensity
- Viscosity at high concentration
- Self-interaction
- Polyreactivity
- Clearance rate
- Immunogenicity risk
- Expression level
- Stability

---

### 4. Enzyme Engineering

#### Activity Prediction
Predict kinetic parameters (kcat, Km, kcat/Km).

```python
from finetuning.configs import EnzymeActivityConfig
from finetuning.heads import EnzymeActivityHead

config = EnzymeActivityConfig(
    activity_type="kcat_km",
    use_substrate_features=True,
    include_conditions=True,
)

head = EnzymeActivityHead(EnzymeHeadConfig())
```

#### Substrate Specificity
Predict activity across substrate libraries.

```python
from finetuning.configs import EnzymeSpecificityConfig
from finetuning.heads import EnzymeSpecificityHead

config = EnzymeSpecificityConfig(
    specificity_type="substrate",
    use_docking_features=True,
)
```

#### Directed Evolution Guidance
Score mutations for activity improvement.

```python
from finetuning.configs import EnzymeEngineeringConfig
from finetuning.heads import EnzymeEvolutionHead

config = EnzymeEngineeringConfig(
    optimization_target="activity",
    focus_active_site=True,
)
```

**Datasets**: BRENDA, SABIO-RK, UniProt enzyme data

---

### 5. Protein-Protein Interaction

#### PPI Binding Prediction
Predict binding affinity for protein complexes.

```python
from finetuning.configs import PPIBindingConfig
from finetuning.heads import PPIBindingHead

config = PPIBindingConfig(
    binding_type="kd",
    use_interface_features=True,
)
```

#### Interface Prediction
Identify interface residues.

```python
from finetuning.configs import PPIInterfaceConfig
from finetuning.heads import PPIInterfaceHead

config = PPIInterfaceConfig(
    interface_threshold=5.0,  # Å
    use_coevolution=True,
)
```

#### Hot Spot Prediction
Identify binding energy hot spots.

```python
from finetuning.configs import PPIHotspotConfig
from finetuning.heads import PPIHotspotHead

config = PPIHotspotConfig(
    ddg_threshold=2.0,  # kcal/mol
)
```

**Datasets**: PDBbind (protein-protein), SKEMPI, ASEdb

---

### 6. Function Prediction

#### GO Term Prediction
Predict Gene Ontology annotations.

```python
from finetuning.configs import FunctionPredictionConfig
from finetuning.heads import GOPredictionHead

config = FunctionPredictionConfig(
    ontology="go_mf",  # go_mf, go_bp, go_cc
    use_hierarchy=True,
)
```

**Output**: Multi-label predictions for MF, BP, CC terms

#### EC Number Prediction
Predict enzyme classification.

```python
from finetuning.heads import ECNumberHead

head = ECNumberHead(FunctionHeadConfig())
```

**Output**: Hierarchical EC predictions (X.X.X.X)

#### Subcellular Localization
Predict where proteins are located.

```python
from finetuning.configs import LocalizationConfig
from finetuning.heads import LocalizationHead

config = LocalizationConfig(
    localizations=[
        "nucleus", "cytoplasm", "membrane",
        "mitochondria", "extracellular",
    ],
)
```

**Datasets**: UniProt, DeepLoc, TargetP training data

---

### 7. Immunology / Epitopes

#### B-cell Epitope Prediction
Predict antibody binding sites (conformational epitopes).

```python
from finetuning.configs import BcellEpitopeConfig
from finetuning.heads import BcellEpitopeHead

config = BcellEpitopeConfig(
    epitope_type="conformational",
    use_accessibility=True,
)
```

**Features**: Surface accessibility, Flexibility, Protrusion, Antigenicity

#### T-cell Epitope / MHC Binding
Predict peptide-MHC binding.

```python
from finetuning.configs import TcellEpitopeConfig
from finetuning.heads import TcellEpitopeHead

config = TcellEpitopeConfig(
    mhc_class="I",
    alleles=["HLA-A*02:01", "HLA-A*01:01"],
    predict_immunogenicity=True,
)
```

#### Therapeutic Immunogenicity
Assess immunogenicity risk for biologics.

```python
from finetuning.configs import ImmunogenicityConfig
from finetuning.heads import ImmunogenicityHead

config = ImmunogenicityConfig(
    predict_tcell_response=True,
    predict_ada=True,  # Anti-drug antibodies
)
```

**Output**: Overall risk score + component risks (T-cell, B-cell, aggregation)

**Datasets**: IEDB, NetMHC training data

---

## Quick Start

### Installation

```bash
git clone https://github.com/chenxingqiang/alphafold-notebooks.git
cd alphafold-notebooks

pip install torch>=2.0 numpy scipy
```

### Basic Usage

```python
from finetuning import FineTuningConfig, Trainer
from finetuning.configs import get_task_config
from finetuning.data import get_dataset

# 1. Choose task
task_config = get_task_config("binding_affinity")

# 2. Configure fine-tuning
config = FineTuningConfig(
    strategy="lora",
    task="binding_affinity",
    lora_rank=8,
)

# 3. Load data
train_dataset = get_dataset("affinity", data_path="./data/train")
train_loader = DataLoader(train_dataset, batch_size=1)

# 4. Train
trainer = Trainer(model, config, train_loader)
trainer.train()
```

### AlphaFold 3 Weight Utilities & LoRA Fine-tuning

Since **v3.0.4**, official weights are available at
`https://storage.googleapis.com/alphafold3/af3.bin.zst` (non-commercial use;
see [WEIGHTS_TERMS_OF_USE.md](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)).

```bash
pip install numpy zstandard  # zstandard required for .bin.zst files

python -m finetuning.af3.weights info
python -m finetuning.af3.weights check /path/to/model_parameters/
```

```python
from finetuning.af3 import AlphaFold3FineTuner, AF3FineTuneConfig, LoRAConfig

config = AF3FineTuneConfig(
    strategy="lora",
    lora=LoRAConfig(rank=8, alpha=16.0),
)
tuner = AlphaFold3FineTuner.from_pretrained("/path/to/model_parameters/", config)

# Trainable = LoRA adapter tensors only; base AF3 weights stay frozen
print(tuner.parameter_summary().describe())
tuner.save_adapter("./af3_lora.npz")

# Merged export requires explicit terms acknowledgement (restricted distribution)
# tuner.export_merged_weights("./merged.bin", acknowledge_weights_terms=True)
```

See [alphafold3/AF3_WEIGHTS_FINETUNING_DESIGN.md](../alphafold3/AF3_WEIGHTS_FINETUNING_DESIGN.md) for architecture,
compliance constraints, and the full test plan.

### List Available Tasks

```python
from finetuning.configs.task_config import list_tasks_by_category

categories = list_tasks_by_category()
for category, tasks in categories.items():
    print(f"{category}: {tasks}")
```

Output:
```
drug_discovery: ['binding_affinity', 'virtual_screening', 'admet']
engineering: ['stability', 'solubility', 'mutation_ddg', 'mutation_fitness']
antibody: ['antibody_affinity', 'antibody_developability', 'humanization']
enzyme: ['enzyme_activity', 'enzyme_specificity', 'enzyme_evolution']
ppi: ['ppi_binding', 'ppi_interface', 'ppi_hotspot']
function: ['function_go', 'function_ec', 'localization']
immunology: ['bcell_epitope', 'tcell_epitope', 'immunogenicity']
```

---

## Configuration

### Full Example

```yaml
# config.yaml
model:
  model_type: boltz2
  pretrained_path: /path/to/weights
  precision: bf16

training:
  learning_rate: 5e-5
  max_steps: 50000
  batch_size: 1
  gradient_accumulation_steps: 8

strategy: lora
task: binding_affinity

lora_rank: 8
lora_alpha: 16.0
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj

loss_weights:
  affinity: 1.0
  fape: 0.1
```

---

## Best Practices

### 1. Task-Specific Learning Rates

| Task | Recommended LR |
|------|---------------|
| Binding Affinity | 5e-5 |
| Stability (ΔΔG) | 1e-4 |
| Antibody Optimization | 5e-5 |
| Function Prediction | 1e-4 |
| Epitope Prediction | 5e-5 |

### 2. Data Requirements

| Task | Minimum Samples | Recommended |
|------|----------------|-------------|
| Binding Affinity | 500 | 5,000+ |
| Mutation Effects | 1,000 | 10,000+ |
| Antibody Design | 100 | 1,000+ |
| Function (GO) | 10,000 | 100,000+ |

### 3. Evaluation Metrics by Task

| Task | Primary Metrics |
|------|----------------|
| Binding Affinity | RMSE, Pearson, Spearman |
| Virtual Screening | AUROC, EF1%, BEDROC |
| Mutation Effects | Spearman, AUROC (pathogenicity) |
| Function | F1-max, AUROC, AUPRC |
| Epitope | AUROC, Precision@L/5 |

### 4. Multi-task Learning

Fine-tune for multiple related tasks simultaneously:

```python
config = FineTuningConfig(
    task="multi_task",
    loss_weights={
        "affinity": 1.0,
        "stability": 0.5,
        "contact": 0.3,
    },
)
```

---

## Architecture Overview

```
finetuning/
├── configs/
│   ├── base_config.py       # Core configuration classes
│   ├── task_config.py       # 25+ task configurations
│   └── lora_config.py       # LoRA settings
├── heads/
│   ├── affinity_head.py     # Drug-target binding
│   ├── antibody_head.py     # Antibody optimization
│   ├── ppi_head.py          # Protein-protein interaction
│   ├── enzyme_head.py       # Enzyme engineering
│   ├── function_head.py     # GO/EC prediction
│   └── epitope_head.py      # Immunology
├── data/
│   └── datasets.py          # Task-specific datasets
├── modules/
│   ├── lora.py              # LoRA implementation
│   └── adapter.py           # Adapter modules
└── trainers/
    └── trainer.py           # Training loop
```

---

## Citation

If you use this fine-tuning framework, please cite:

```bibtex
@software{alphafold_codec_finetuning,
  title={AlphaFold Codec: Fine-tuning Framework},
  author={Chen, Xingqiang and Contributors},
  year={2026},
  url={https://github.com/chenxingqiang/alphafold-notebooks}
}
```
