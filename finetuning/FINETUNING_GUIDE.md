# Fine-tuning Guide for Protein Structure Prediction Models

This guide covers fine-tuning AlphaFold2, AlphaFold3, Boltz-1, and Boltz-2 for various downstream tasks.

## Table of Contents

1. [Overview](#overview)
2. [Supported Models](#supported-models)
3. [Fine-tuning Strategies](#fine-tuning-strategies)
4. [Task Types](#task-types)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Examples](#examples)

## Overview

Fine-tuning allows you to adapt pretrained protein structure prediction models to specific tasks or domains:

- **Domain-specific**: Antibodies, membrane proteins, enzymes
- **Task-specific**: Binding affinity, stability, solubility
- **Data-efficient**: Few-shot learning for rare protein families

### When to Fine-tune

| Scenario | Recommended Strategy |
|----------|---------------------|
| Small dataset (<1000 samples) | LoRA or Adapter |
| Large dataset (>10000 samples) | Full fine-tuning or Freeze backbone |
| New prediction task | Head-only fine-tuning |
| Domain adaptation | LoRA with domain data |

## Supported Models

| Model | Framework | Fine-tuning Support |
|-------|-----------|-------------------|
| AlphaFold2 | JAX/Haiku | ✅ Full, Head-only, LoRA |
| AlphaFold3 | JAX/Haiku | ✅ Full, Head-only, LoRA |
| Boltz-1 | PyTorch | ✅ Full, LoRA, Adapter |
| Boltz-2 | PyTorch | ✅ Full, LoRA, Adapter |
| OpenFold | PyTorch | ✅ Full, LoRA |
| ESMFold | PyTorch | ✅ Full, LoRA |

## Fine-tuning Strategies

### 1. LoRA (Low-Rank Adaptation)

Most parameter-efficient method. Decomposes weight updates into low-rank matrices.

```python
from finetuning import FineTuningConfig, LoRAModule

config = FineTuningConfig(
    strategy="lora",
    lora_rank=8,
    lora_alpha=16.0,
    lora_target_modules=["q_proj", "k_proj", "v_proj"]
)
```

**Pros**: Very few trainable parameters (~0.1% of model)
**Cons**: May not reach full fine-tuning performance

### 2. Adapter

Insert small bottleneck modules between layers.

```python
config = FineTuningConfig(
    strategy="adapter",
    adapter_hidden_dim=64,
)
```

**Pros**: Modular, can add multiple adapters
**Cons**: Slightly increases inference latency

### 3. Head-only

Only train the prediction heads, freeze the backbone.

```python
config = FineTuningConfig(
    strategy="head_only",
    freeze_embeddings=True,
    freeze_evoformer_layers=48,  # Freeze all Evoformer layers
)
```

**Pros**: Fast, prevents catastrophic forgetting
**Cons**: Limited adaptation capability

### 4. Full Fine-tuning

Train all parameters (use with caution).

```python
config = FineTuningConfig(
    strategy="full",
    freeze_evoformer_layers=24,  # Optionally freeze early layers
)
```

**Pros**: Maximum adaptation capability
**Cons**: Requires large dataset, risk of catastrophic forgetting

## Task Types

### Binding Affinity Prediction

Predict protein-ligand binding strength (pKd, pIC50, ΔG).

```python
from finetuning.heads import AffinityHead, AffinityHeadConfig

head_config = AffinityHeadConfig(
    single_channel=384,
    pair_channel=128,
    hidden_dim=256,
    use_gaussian_smearing=True,
)

affinity_head = AffinityHead(head_config)
```

### Property Prediction

Predict protein properties (stability, solubility, expression).

```python
from finetuning.heads import PropertyHead, PropertyHeadConfig

head_config = PropertyHeadConfig(
    num_properties=3,  # stability, solubility, expression
    pooling_strategy="attention",
)

property_head = PropertyHead(head_config)
```

### Contact Prediction

Predict residue-residue contacts.

```python
from finetuning.heads import ContactHead, ContactHeadConfig

head_config = ContactHeadConfig(
    contact_threshold=8.0,  # Angstroms
    min_sequence_separation=6,
)

contact_head = ContactHead(head_config)
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chenxingqiang/alphafold-notebooks.git
cd alphafold-notebooks

# Install dependencies
pip install torch>=2.0 numpy scipy

# Optional: Install for distributed training
pip install deepspeed wandb
```

### Basic Usage

```python
from finetuning import FineTuningConfig, Trainer
from finetuning.modules import LoRAModule
from finetuning.heads import AffinityHead

# 1. Load pretrained model
model = load_pretrained_boltz2()

# 2. Apply LoRA
lora_model = LoRAModule(
    model,
    rank=8,
    alpha=16.0,
    target_modules=["q_proj", "k_proj", "v_proj"],
)

# 3. Add task head
affinity_head = AffinityHead(AffinityHeadConfig())
lora_model.add_head(affinity_head)

# 4. Configure training
config = FineTuningConfig(
    strategy="lora",
    task="binding_affinity",
    training=TrainingConfig(
        learning_rate=5e-5,
        max_steps=10000,
        batch_size=1,
        gradient_accumulation_steps=8,
    ),
)

# 5. Train
trainer = Trainer(lora_model, config, train_dataloader, val_dataloader)
trainer.train()
```

## Configuration

### Full Configuration Example

```yaml
# config.yaml
model:
  model_type: boltz2
  pretrained_path: /path/to/boltz2_weights.pt
  precision: bf16

training:
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 1000
  max_steps: 50000
  batch_size: 1
  gradient_accumulation_steps: 8
  lr_scheduler: warmup_cosine
  output_dir: ./finetuning_output

strategy: lora
task: binding_affinity

lora_rank: 8
lora_alpha: 16.0
lora_dropout: 0.1
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

loss_weights:
  affinity: 1.0
  fape: 0.1

seed: 42
```

### Loading Configuration

```python
config = FineTuningConfig.from_yaml("config.yaml")
```

## Examples

### Example 1: Fine-tune Boltz-2 for Drug Binding Affinity

```python
from finetuning import FineTuningConfig, get_preset_config
from finetuning.data import AffinityDataset
from torch.utils.data import DataLoader

# Load preset configuration
config = get_preset_config("boltz2_affinity_lora")

# Prepare data
train_dataset = AffinityDataset(
    data_path="./data/pdbbind/train",
    affinity_file="./data/pdbbind/affinities.csv",
)
val_dataset = AffinityDataset(
    data_path="./data/pdbbind/val",
    affinity_file="./data/pdbbind/affinities.csv",
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# Train
trainer = Trainer(model, config, train_loader, val_loader)
trainer.train()

# Save LoRA weights only (small file)
trainer.model.save_lora_weights("./lora_weights.pt")
```

### Example 2: Multi-task Property Prediction

```python
from finetuning.heads import PropertyHead, PropertyHeadConfig

# Configure for multiple properties
head_config = PropertyHeadConfig(
    num_properties=5,
    property_names=["stability", "solubility", "aggregation", "expression", "half_life"],
    predict_uncertainty=True,
)

head = PropertyHead(head_config)

# Training will optimize all properties jointly
```

### Example 3: Few-shot Domain Adaptation

```python
# For few-shot scenarios, use aggressive LoRA
config = FineTuningConfig(
    strategy="lora",
    lora_rank=16,  # Higher rank for more capacity
    lora_alpha=32.0,
    training=TrainingConfig(
        learning_rate=1e-4,  # Higher LR for few-shot
        max_steps=1000,  # Fewer steps
        warmup_steps=100,
    ),
)
```

## Evaluation

### Compute Metrics

```python
from finetuning.utils import compute_metrics, evaluate_model

# Evaluate on test set
metrics = evaluate_model(
    model,
    test_dataloader,
    metrics=["rmse", "mae", "pearson", "spearman", "r2"],
)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"Pearson: {metrics['pearson']:.4f}")
```

### Structure Quality Metrics

```python
from finetuning.utils.metrics import compute_lddt, compute_tm_score

# Per-residue lDDT
lddt_scores = compute_lddt(predicted_coords, true_coords)
print(f"Mean lDDT: {lddt_scores.mean():.4f}")

# TM-score
tm = compute_tm_score(predicted_coords, true_coords)
print(f"TM-score: {tm:.4f}")
```

## Tips and Best Practices

### 1. Learning Rate Selection

| Strategy | Recommended LR |
|----------|---------------|
| Full fine-tuning | 1e-5 to 5e-5 |
| LoRA | 5e-5 to 1e-4 |
| Adapter | 1e-4 to 5e-4 |
| Head-only | 1e-4 to 1e-3 |

### 2. Prevent Overfitting

- Use weight decay (0.01-0.1)
- Enable dropout in LoRA (0.1)
- Early stopping based on validation loss
- Data augmentation (structure rotation, MSA subsampling)

### 3. Memory Optimization

```python
# Enable gradient checkpointing
config.training.gradient_checkpointing = True

# Use mixed precision
config.model.precision = "bf16"

# Gradient accumulation for large effective batch size
config.training.gradient_accumulation_steps = 16
```

### 4. Distributed Training

```bash
# Launch distributed training
torchrun --nproc_per_node=4 train.py --config config.yaml
```

## Troubleshooting

### Out of Memory

- Reduce batch size
- Enable gradient checkpointing
- Use LoRA instead of full fine-tuning
- Reduce sequence length

### Training Instability

- Lower learning rate
- Increase warmup steps
- Clip gradients (max_grad_norm=1.0)
- Check for NaN in inputs

### Poor Performance

- Try higher LoRA rank
- Increase training steps
- Add more data augmentation
- Verify data preprocessing
