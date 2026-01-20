"""LoRA (Low-Rank Adaptation) configuration."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation).
    
    LoRA enables efficient fine-tuning by decomposing weight updates into
    low-rank matrices: W = W_0 + BA, where B ∈ R^{d×r} and A ∈ R^{r×k}.
    
    Reference: https://arxiv.org/abs/2106.09685
    """
    
    rank: int = 8
    """Rank of the low-rank decomposition."""
    
    alpha: float = 16.0
    """Scaling factor. The actual scaling is alpha/rank."""
    
    dropout: float = 0.1
    """Dropout probability for LoRA layers."""
    
    target_modules: List[str] = field(default_factory=lambda: [
        # Attention projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        "query_w", "key_w", "value_w", "output_w",
        # Triangle operations
        "left_projection", "right_projection",
        # Transitions
        "transition1", "transition2",
    ])
    """Module names to apply LoRA to."""
    
    modules_to_save: List[str] = field(default_factory=lambda: [
        # Always save these modules in full
        "output_layer_norm",
        "final_layer_norm",
    ])
    """Modules to save in full (not LoRA-adapted)."""
    
    bias: str = "none"
    """Bias handling: 'none', 'all', or 'lora_only'."""
    
    init_lora_weights: bool = True
    """Whether to initialize LoRA weights (A=Gaussian, B=zeros)."""
    
    # Advanced options
    fan_in_fan_out: bool = False
    """Set True if the layer weights are stored as (fan_out, fan_in)."""
    
    merge_weights: bool = True
    """Whether to merge LoRA weights into base weights after training."""
    
    # Per-layer rank (for layer-wise LoRA)
    layer_ranks: Optional[Dict[str, int]] = None
    """Optional per-layer rank override."""
    
    def get_scaling(self) -> float:
        """Get the LoRA scaling factor."""
        return self.alpha / self.rank
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "modules_to_save": self.modules_to_save,
            "bias": self.bias,
            "init_lora_weights": self.init_lora_weights,
            "fan_in_fan_out": self.fan_in_fan_out,
            "merge_weights": self.merge_weights,
            "layer_ranks": self.layer_ranks,
        }


# Preset LoRA configurations for different scenarios
LORA_PRESETS = {
    "small": LoRAConfig(rank=4, alpha=8.0),
    "medium": LoRAConfig(rank=8, alpha=16.0),
    "large": LoRAConfig(rank=16, alpha=32.0),
    "xlarge": LoRAConfig(rank=32, alpha=64.0),
    
    # Task-specific presets
    "affinity": LoRAConfig(
        rank=8,
        alpha=16.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "left_projection", "right_projection",
        ],
    ),
    "structure": LoRAConfig(
        rank=16,
        alpha=32.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "left_projection", "right_projection",
            "transition1", "transition2",
        ],
    ),
}
