# Fine-tuning Framework for Protein Structure Prediction Models
# Supports AlphaFold2, AlphaFold3, Boltz-1, and Boltz-2

"""
Fine-tuning Framework for Protein Structure Prediction Models
=============================================================

This module provides comprehensive fine-tuning support for:
- AlphaFold2 (JAX/Haiku)
- AlphaFold3 (JAX/Haiku)  
- Boltz-1 (PyTorch)
- Boltz-2 (PyTorch)

Supported Fine-tuning Strategies:
---------------------------------
1. Full Fine-tuning: Update all model parameters
2. Head-only Fine-tuning: Only update prediction heads
3. LoRA (Low-Rank Adaptation): Efficient parameter-efficient fine-tuning
4. Adapter Fine-tuning: Insert small adapter modules
5. Prompt Tuning: Learn task-specific embeddings

Supported Tasks:
----------------
1. Structure Prediction: Fine-tune for specific protein families
2. Binding Affinity: Predict protein-ligand binding strength
3. Property Prediction: Predict protein properties (stability, solubility, etc.)
4. Contact Prediction: Predict residue-residue contacts
5. Function Prediction: Predict protein function from structure

Usage:
------
>>> from finetuning import FineTuningConfig, LoRAModule, AffinityHead
>>> from finetuning.trainers import Trainer
>>> 
>>> config = FineTuningConfig(
...     model_type="boltz2",
...     strategy="lora",
...     task="binding_affinity",
...     lora_rank=8
... )
>>> trainer = Trainer(config)
>>> trainer.train(train_data, val_data)
"""

from .configs import FineTuningConfig, ModelConfig, TrainingConfig
from .modules import LoRAModule, AdapterModule, PromptTuning
from .heads import AffinityHead, PropertyHead, ContactHead
from .trainers import Trainer, DistributedTrainer
from .utils import load_pretrained, save_checkpoint, evaluate_model

__version__ = "0.1.0"
__all__ = [
    "FineTuningConfig",
    "ModelConfig", 
    "TrainingConfig",
    "LoRAModule",
    "AdapterModule",
    "PromptTuning",
    "AffinityHead",
    "PropertyHead",
    "ContactHead",
    "Trainer",
    "DistributedTrainer",
    "load_pretrained",
    "save_checkpoint",
    "evaluate_model",
]
