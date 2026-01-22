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

Supported Task Categories (50+ tasks):
--------------------------------------
- Drug Discovery: Binding affinity, virtual screening, ADMET
- Protein Engineering: Stability, solubility, mutation effects
- Antibody Design: Affinity maturation, humanization, developability
- Enzyme Engineering: Activity, specificity, directed evolution
- Protein-Protein Interactions: Binding, interface, hot spots
- Function Prediction: GO terms, EC numbers, localization
- Immunology: B-cell epitopes, T-cell epitopes, immunogenicity
- Structure Quality: pLDDT, pAE, disorder, contacts

Quick Start:
------------
>>> from finetuning import TaskRegistry, create_finetuning_pipeline
>>> 
>>> # List all available tasks
>>> print(TaskRegistry.list_all_tasks())
>>> 
>>> # Get task info
>>> info = TaskRegistry.get_task_info("binding_affinity")
>>> 
>>> # Create pipeline
>>> pipeline = create_finetuning_pipeline(
...     task="binding_affinity",
...     base_model=model,
...     strategy="lora",
... )

Detailed Usage:
---------------
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
from .heads import (
    AffinityHead, 
    PropertyHead, 
    ContactHead,
    AntibodyAffinityHead,
    DevelopabilityHead,
    HumannessHead,
    PPIBindingHead,
    PPIInterfaceHead,
    PPIHotspotHead,
    EnzymeActivityHead,
    EnzymeSpecificityHead,
    EnzymeEvolutionHead,
    GOPredictionHead,
    ECNumberHead,
    LocalizationHead,
    BcellEpitopeHead,
    TcellEpitopeHead,
    ImmunogenicityHead,
)
from .trainers import Trainer, DistributedTrainer
from .utils import load_pretrained, save_checkpoint, evaluate_model
from .registry import (
    TaskRegistry,
    TaskCategory,
    TaskInfo,
    create_finetuning_pipeline,
    get_task,
    list_tasks,
)

__version__ = "0.2.0"
__all__ = [
    # Configs
    "FineTuningConfig",
    "ModelConfig", 
    "TrainingConfig",
    # Modules
    "LoRAModule",
    "AdapterModule",
    "PromptTuning",
    # Heads - Core
    "AffinityHead",
    "PropertyHead",
    "ContactHead",
    # Heads - Antibody
    "AntibodyAffinityHead",
    "DevelopabilityHead",
    "HumannessHead",
    # Heads - PPI
    "PPIBindingHead",
    "PPIInterfaceHead",
    "PPIHotspotHead",
    # Heads - Enzyme
    "EnzymeActivityHead",
    "EnzymeSpecificityHead",
    "EnzymeEvolutionHead",
    # Heads - Function
    "GOPredictionHead",
    "ECNumberHead",
    "LocalizationHead",
    # Heads - Immunology
    "BcellEpitopeHead",
    "TcellEpitopeHead",
    "ImmunogenicityHead",
    # Trainers
    "Trainer",
    "DistributedTrainer",
    # Utils
    "load_pretrained",
    "save_checkpoint",
    "evaluate_model",
    # Registry
    "TaskRegistry",
    "TaskCategory",
    "TaskInfo",
    "create_finetuning_pipeline",
    "get_task",
    "list_tasks",
]
