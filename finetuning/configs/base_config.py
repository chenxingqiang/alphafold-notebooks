"""Base configuration classes for fine-tuning protein structure prediction models."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class ModelType(str, Enum):
    """Supported model types."""
    ALPHAFOLD2 = "alphafold2"
    ALPHAFOLD3 = "alphafold3"
    BOLTZ1 = "boltz1"
    BOLTZ2 = "boltz2"
    OPENFOLD = "openfold"
    ESMFOLD = "esmfold"


class FineTuningStrategy(str, Enum):
    """Fine-tuning strategies."""
    FULL = "full"              # Update all parameters
    HEAD_ONLY = "head_only"    # Only update prediction heads
    LORA = "lora"              # Low-Rank Adaptation
    ADAPTER = "adapter"        # Adapter modules
    PROMPT = "prompt"          # Prompt tuning
    FREEZE_BACKBONE = "freeze_backbone"  # Freeze backbone, train heads


class TaskType(str, Enum):
    """Supported fine-tuning tasks."""
    STRUCTURE_PREDICTION = "structure_prediction"
    BINDING_AFFINITY = "binding_affinity"
    PROPERTY_PREDICTION = "property_prediction"
    CONTACT_PREDICTION = "contact_prediction"
    FUNCTION_PREDICTION = "function_prediction"
    MUTATION_EFFECT = "mutation_effect"
    DOCKING = "docking"


@dataclass
class ModelConfig:
    """Configuration for the base model."""

    model_type: ModelType = ModelType.BOLTZ2
    """Type of model to fine-tune."""

    pretrained_path: Optional[str] = None
    """Path to pretrained model weights."""

    # Architecture parameters
    num_evoformer_blocks: int = 48
    """Number of Evoformer/Pairformer blocks."""

    msa_channel: int = 256
    """MSA representation channel dimension."""

    pair_channel: int = 128
    """Pair representation channel dimension."""

    seq_channel: int = 384
    """Sequence representation channel dimension."""

    num_heads: int = 8
    """Number of attention heads."""

    # Diffusion parameters (for AF3/Boltz)
    num_diffusion_steps: int = 200
    """Number of diffusion steps for inference."""

    sigma_data: float = 16.0
    """Data standard deviation for diffusion."""

    # Device configuration
    device: str = "cuda"
    """Device to use (cuda, cpu, tpu)."""

    precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    """Training precision."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_type": self.model_type.value,
            "pretrained_path": self.pretrained_path,
            "num_evoformer_blocks": self.num_evoformer_blocks,
            "msa_channel": self.msa_channel,
            "pair_channel": self.pair_channel,
            "seq_channel": self.seq_channel,
            "num_heads": self.num_heads,
            "num_diffusion_steps": self.num_diffusion_steps,
            "sigma_data": self.sigma_data,
            "device": self.device,
            "precision": self.precision,
        }


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-4
    """Base learning rate."""

    weight_decay: float = 0.01
    """Weight decay for regularization."""

    warmup_steps: int = 1000
    """Number of warmup steps for learning rate."""

    max_steps: int = 100000
    """Maximum number of training steps."""

    batch_size: int = 1
    """Batch size per device."""

    gradient_accumulation_steps: int = 8
    """Number of gradient accumulation steps."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    # Scheduler
    lr_scheduler: Literal["cosine", "linear", "constant", "warmup_cosine"] = "warmup_cosine"
    """Learning rate scheduler type."""

    # Checkpointing
    save_steps: int = 1000
    """Save checkpoint every N steps."""

    eval_steps: int = 500
    """Evaluate every N steps."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep."""

    output_dir: str = "./finetuning_output"
    """Output directory for checkpoints and logs."""

    # Logging
    logging_steps: int = 100
    """Log metrics every N steps."""

    wandb_project: Optional[str] = None
    """Weights & Biases project name."""

    # Distributed training
    distributed: bool = False
    """Whether to use distributed training."""

    num_gpus: int = 1
    """Number of GPUs to use."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler": self.lr_scheduler,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "output_dir": self.output_dir,
            "logging_steps": self.logging_steps,
            "wandb_project": self.wandb_project,
            "distributed": self.distributed,
            "num_gpus": self.num_gpus,
        }


@dataclass
class FineTuningConfig:
    """Main configuration for fine-tuning."""

    # Core configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training configuration."""

    # Fine-tuning strategy
    strategy: FineTuningStrategy = FineTuningStrategy.LORA
    """Fine-tuning strategy to use."""

    task: TaskType = TaskType.BINDING_AFFINITY
    """Task to fine-tune for."""

    # LoRA parameters
    lora_rank: int = 8
    """Rank for LoRA decomposition."""

    lora_alpha: float = 16.0
    """Alpha scaling for LoRA."""

    lora_dropout: float = 0.1
    """Dropout rate for LoRA layers."""

    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "left_projection", "right_projection"
    ])
    """Modules to apply LoRA to."""

    # Adapter parameters
    adapter_hidden_dim: int = 64
    """Hidden dimension for adapter modules."""

    # Prompt tuning parameters
    num_prompt_tokens: int = 10
    """Number of learnable prompt tokens."""

    # Frozen layers
    freeze_embeddings: bool = False
    """Whether to freeze embedding layers."""

    freeze_evoformer_layers: Optional[int] = None
    """Number of Evoformer layers to freeze (from bottom)."""

    # Data augmentation
    use_msa_augmentation: bool = True
    """Whether to use MSA augmentation."""

    use_structure_augmentation: bool = True
    """Whether to use structure augmentation (rotation, translation)."""

    # Loss weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "fape": 1.0,
        "distogram": 0.3,
        "plddt": 0.01,
        "affinity": 1.0,
        "property": 1.0,
    })
    """Weights for different loss components."""

    # Seed
    seed: int = 42
    """Random seed for reproducibility."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.strategy, str):
            self.strategy = FineTuningStrategy(self.strategy)
        if isinstance(self.task, str):
            self.task = TaskType(self.task)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "strategy": self.strategy.value,
            "task": self.task.value,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "adapter_hidden_dim": self.adapter_hidden_dim,
            "num_prompt_tokens": self.num_prompt_tokens,
            "freeze_embeddings": self.freeze_embeddings,
            "freeze_evoformer_layers": self.freeze_evoformer_layers,
            "use_msa_augmentation": self.use_msa_augmentation,
            "use_structure_augmentation": self.use_structure_augmentation,
            "loss_weights": self.loss_weights,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FineTuningConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FineTuningConfig":
        """Load config from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def save_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Preset configurations for common fine-tuning scenarios
PRESET_CONFIGS = {
    "boltz2_affinity_lora": FineTuningConfig(
        model=ModelConfig(model_type=ModelType.BOLTZ2),
        training=TrainingConfig(learning_rate=5e-5, max_steps=50000),
        strategy=FineTuningStrategy.LORA,
        task=TaskType.BINDING_AFFINITY,
        lora_rank=8,
        lora_alpha=16.0,
    ),
    "alphafold2_antibody_full": FineTuningConfig(
        model=ModelConfig(model_type=ModelType.ALPHAFOLD2),
        training=TrainingConfig(learning_rate=1e-5, max_steps=100000),
        strategy=FineTuningStrategy.FULL,
        task=TaskType.STRUCTURE_PREDICTION,
        freeze_evoformer_layers=24,  # Freeze first 24 layers
    ),
    "alphafold3_pairformer_lora": FineTuningConfig(
        model=ModelConfig(
            model_type=ModelType.ALPHAFOLD3,
            num_evoformer_blocks=48,
            pair_channel=128,
            seq_channel=384,
        ),
        training=TrainingConfig(learning_rate=1e-4, max_steps=50000),
        strategy=FineTuningStrategy.LORA,
        task=TaskType.STRUCTURE_PREDICTION,
        lora_rank=8,
        lora_alpha=16.0,
    ),
    "boltz1_property_adapter": FineTuningConfig(
        model=ModelConfig(model_type=ModelType.BOLTZ1),
        training=TrainingConfig(learning_rate=1e-4, max_steps=20000),
        strategy=FineTuningStrategy.ADAPTER,
        task=TaskType.PROPERTY_PREDICTION,
        adapter_hidden_dim=64,
    ),
}


def get_preset_config(preset_name: str) -> FineTuningConfig:
    """Get a preset configuration by name."""
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    return PRESET_CONFIGS[preset_name]
