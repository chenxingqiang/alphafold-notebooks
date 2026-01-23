"""LoRA (Low-Rank Adaptation) module for efficient fine-tuning.

This module implements LoRA for both PyTorch (Boltz) and JAX/Haiku (AlphaFold).
LoRA decomposes weight updates into low-rank matrices, dramatically reducing
the number of trainable parameters.

Reference: "LoRA: Low-Rank Adaptation of Large Language Models"
           https://arxiv.org/abs/2106.09685
"""

import math
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

# PyTorch implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# JAX implementation
try:
    import jax
    import jax.numpy as jnp
    import haiku as hk
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

import numpy as np


# =============================================================================
# PyTorch Implementation
# =============================================================================

if TORCH_AVAILABLE:

    class LoRALinear(nn.Module):
        """Linear layer with LoRA adaptation.

        Implements: y = (W + BA/r) @ x + b
        where W is frozen and B, A are trainable low-rank matrices.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
            bias: bool = True,
            merge_weights: bool = True,
        ):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.merge_weights = merge_weights
            self.merged = False

            # Original weight (frozen)
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features),
                requires_grad=False
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(out_features),
                    requires_grad=False
                )
            else:
                self.register_parameter('bias', None)

            # LoRA weights (trainable)
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

            # Dropout
            if dropout > 0:
                self.lora_dropout = nn.Dropout(p=dropout)
            else:
                self.lora_dropout = nn.Identity()

            # Initialize
            self.reset_parameters()

        def reset_parameters(self):
            """Initialize LoRA weights."""
            # Initialize A with Kaiming uniform
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Initialize B with zeros (so initial LoRA contribution is 0)
            nn.init.zeros_(self.lora_B)

        def merge(self):
            """Merge LoRA weights into the main weights."""
            if not self.merged:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

        def unmerge(self):
            """Unmerge LoRA weights from main weights."""
            if self.merged:
                self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            if self.merged:
                return F.linear(x, self.weight, self.bias)
            else:
                # Original forward + LoRA contribution
                result = F.linear(x, self.weight, self.bias)
                lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
                return result + lora_out * self.scaling

        @classmethod
        def from_linear(
            cls,
            linear: nn.Linear,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
        ) -> "LoRALinear":
            """Create LoRALinear from existing Linear layer."""
            lora_linear = cls(
                in_features=linear.in_features,
                out_features=linear.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=linear.bias is not None,
            )
            # Copy original weights
            lora_linear.weight.data = linear.weight.data.clone()
            if linear.bias is not None:
                lora_linear.bias.data = linear.bias.data.clone()
            return lora_linear


    class LoRAModule(nn.Module):
        """Wrapper module that applies LoRA to specified layers."""

        def __init__(
            self,
            model: nn.Module,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
            target_modules: Optional[List[str]] = None,
        ):
            super().__init__()

            self.model = model
            self.rank = rank
            self.alpha = alpha
            self.dropout = dropout
            self.target_modules = target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj"
            ]

            # Apply LoRA to target modules
            self._apply_lora()

        def _apply_lora(self):
            """Apply LoRA to target modules."""
            for name, module in self.model.named_modules():
                if any(target in name for target in self.target_modules):
                    if isinstance(module, nn.Linear):
                        # Replace with LoRA version
                        parent_name = ".".join(name.split(".")[:-1])
                        child_name = name.split(".")[-1]
                        parent = self.model.get_submodule(parent_name) if parent_name else self.model

                        lora_linear = LoRALinear.from_linear(
                            module,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout,
                        )
                        setattr(parent, child_name, lora_linear)

        def forward(self, *args, **kwargs):
            """Forward pass through the model."""
            return self.model(*args, **kwargs)

        def get_trainable_parameters(self) -> List[nn.Parameter]:
            """Get only the trainable LoRA parameters."""
            params = []
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    params.append(param)
            return params

        def merge_and_unload(self) -> nn.Module:
            """Merge LoRA weights and return the model."""
            for module in self.model.modules():
                if isinstance(module, LoRALinear):
                    module.merge()
            return self.model

        def save_lora_weights(self, path: str):
            """Save only the LoRA weights."""
            lora_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    lora_state[name] = param.data
            torch.save(lora_state, path)

        def load_lora_weights(self, path: str):
            """Load LoRA weights."""
            lora_state = torch.load(path)
            for name, param in self.model.named_parameters():
                if name in lora_state:
                    param.data = lora_state[name]


# =============================================================================
# JAX/Haiku Implementation
# =============================================================================

if JAX_AVAILABLE:

    class LoRALinearHaiku(hk.Module):
        """LoRA Linear layer for JAX/Haiku models."""

        def __init__(
            self,
            output_size: int,
            rank: int = 8,
            alpha: float = 16.0,
            with_bias: bool = True,
            w_init: Optional[hk.initializers.Initializer] = None,
            name: Optional[str] = None,
        ):
            super().__init__(name=name)

            self.output_size = output_size
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.with_bias = with_bias
            self.w_init = w_init or hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
            """Forward pass with LoRA."""
            input_size = inputs.shape[-1]

            # Original weight (frozen in practice via stop_gradient)
            w = hk.get_parameter(
                "w",
                shape=[input_size, self.output_size],
                init=self.w_init,
            )

            # LoRA weights
            lora_a = hk.get_parameter(
                "lora_a",
                shape=[input_size, self.rank],
                init=hk.initializers.RandomNormal(stddev=1.0 / math.sqrt(input_size)),
            )
            lora_b = hk.get_parameter(
                "lora_b",
                shape=[self.rank, self.output_size],
                init=hk.initializers.Constant(0.0),
            )

            # Compute output
            # Original: inputs @ w
            # LoRA: inputs @ lora_a @ lora_b * scaling
            original_out = jnp.einsum("...i,io->...o", inputs, jax.lax.stop_gradient(w))
            lora_out = jnp.einsum("...i,ir,ro->...o", inputs, lora_a, lora_b) * self.scaling
            out = original_out + lora_out

            if self.with_bias:
                b = hk.get_parameter(
                    "b",
                    shape=[self.output_size],
                    init=hk.initializers.Constant(0.0),
                )
                out = out + b

            return out


def apply_lora_to_model(
    model: Any,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Any:
    """Apply LoRA to a model.

    Args:
        model: The model to apply LoRA to (PyTorch nn.Module or JAX function)
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout rate for LoRA layers
        target_modules: List of module names to apply LoRA to

    Returns:
        Model with LoRA applied
    """
    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        return LoRAModule(
            model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )
    else:
        raise NotImplementedError(
            "LoRA application for this model type is not implemented. "
            "For JAX models, use LoRALinearHaiku directly in the model definition."
        )


# =============================================================================
# NumPy Reference Implementation (for understanding)
# =============================================================================

class LoRALinearNumPy:
    """NumPy reference implementation of LoRA for educational purposes."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Initialize weights
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.bias = np.zeros(out_features)

        # LoRA weights
        self.lora_A = np.random.randn(rank, in_features) / np.sqrt(in_features)
        self.lora_B = np.zeros((out_features, rank))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        y = W @ x + b + (B @ A) @ x * scaling
        """
        # Original linear transformation
        out = x @ self.W.T + self.bias

        # LoRA contribution
        lora_out = x @ self.lora_A.T @ self.lora_B.T * self.scaling

        return out + lora_out

    def merge_weights(self) -> np.ndarray:
        """Merge LoRA weights into main weights."""
        return self.W + (self.lora_B @ self.lora_A) * self.scaling

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        original = self.in_features * self.out_features + self.out_features
        lora = self.rank * (self.in_features + self.out_features)
        return {
            "original": original,
            "lora": lora,
            "ratio": lora / original,
        }


def demonstrate_lora():
    """Demonstrate LoRA parameter efficiency."""
    print("LoRA Parameter Efficiency Demonstration")
    print("=" * 50)

    # Example: Attention layer dimensions
    in_features = 384   # seq_channel
    out_features = 384
    rank = 8

    layer = LoRALinearNumPy(in_features, out_features, rank=rank)
    params = layer.count_parameters()

    print(f"Input features: {in_features}")
    print(f"Output features: {out_features}")
    print(f"LoRA rank: {rank}")
    print(f"\nOriginal parameters: {params['original']:,}")
    print(f"LoRA parameters: {params['lora']:,}")
    print(f"Parameter ratio: {params['ratio']:.4f} ({params['ratio']*100:.2f}%)")
    print(f"\nParameter reduction: {(1-params['ratio'])*100:.2f}%")

    # Test forward pass
    x = np.random.randn(10, in_features)  # Batch of 10
    y = layer.forward(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Verify initial LoRA contribution is zero
    initial_lora = x @ layer.lora_A.T @ layer.lora_B.T
    print(f"\nInitial LoRA contribution (should be ~0): {np.abs(initial_lora).max():.6f}")


if __name__ == "__main__":
    demonstrate_lora()
