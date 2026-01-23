"""Adapter modules for efficient fine-tuning.

Adapters are small bottleneck modules inserted between layers of a pretrained
model. Only the adapter parameters are trained, keeping the original model frozen.

Reference: "Parameter-Efficient Transfer Learning for NLP"
           https://arxiv.org/abs/1902.00751
"""

from typing import Optional, List, Callable
import math

# PyTorch implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


if TORCH_AVAILABLE:

    class AdapterLayer(nn.Module):
        """A single adapter layer (bottleneck architecture).

        Architecture: input -> down_proj -> activation -> up_proj -> residual + input
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            activation: str = "relu",
            dropout: float = 0.0,
            init_scale: float = 1e-3,
        ):
            super().__init__()

            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            # Down projection (input_dim -> hidden_dim)
            self.down_proj = nn.Linear(input_dim, hidden_dim)

            # Up projection (hidden_dim -> input_dim)
            self.up_proj = nn.Linear(hidden_dim, input_dim)

            # Activation
            self.activation = self._get_activation(activation)

            # Dropout
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            # Initialize with small weights
            self._init_weights(init_scale)

        def _get_activation(self, name: str) -> nn.Module:
            """Get activation function by name."""
            activations = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "swish": nn.SiLU(),
                "tanh": nn.Tanh(),
            }
            return activations.get(name, nn.ReLU())

        def _init_weights(self, scale: float):
            """Initialize weights with small values."""
            nn.init.normal_(self.down_proj.weight, std=scale)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.normal_(self.up_proj.weight, std=scale)
            nn.init.zeros_(self.up_proj.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with residual connection."""
            # Bottleneck
            h = self.down_proj(x)
            h = self.activation(h)
            h = self.dropout(h)
            h = self.up_proj(h)

            # Residual connection
            return x + h


    class AdapterModule(nn.Module):
        """Module that wraps a model with adapters.

        Inserts adapter layers after specified modules in the model.
        """

        def __init__(
            self,
            model: nn.Module,
            hidden_dim: int = 64,
            activation: str = "relu",
            dropout: float = 0.0,
            target_modules: Optional[List[str]] = None,
        ):
            super().__init__()

            self.model = model
            self.hidden_dim = hidden_dim
            self.activation = activation
            self.dropout = dropout
            self.target_modules = target_modules or [
                "attention", "transition", "outer_product"
            ]

            # Store adapters
            self.adapters = nn.ModuleDict()

            # Insert adapters
            self._insert_adapters()

        def _insert_adapters(self):
            """Insert adapters after target modules."""
            for name, module in self.model.named_modules():
                if any(target in name for target in self.target_modules):
                    # Infer input dimension from the module
                    input_dim = self._get_output_dim(module)
                    if input_dim is not None:
                        adapter_name = name.replace(".", "_")
                        self.adapters[adapter_name] = AdapterLayer(
                            input_dim=input_dim,
                            hidden_dim=self.hidden_dim,
                            activation=self.activation,
                            dropout=self.dropout,
                        )

        def _get_output_dim(self, module: nn.Module) -> Optional[int]:
            """Get the output dimension of a module."""
            if hasattr(module, "out_features"):
                return module.out_features
            elif hasattr(module, "output_dim"):
                return module.output_dim
            elif hasattr(module, "hidden_size"):
                return module.hidden_size
            return None

        def forward(self, *args, **kwargs):
            """Forward pass."""
            # This is a simplified implementation
            # In practice, you'd need to hook into the forward pass
            return self.model(*args, **kwargs)

        def get_trainable_parameters(self) -> List[nn.Parameter]:
            """Get only the trainable adapter parameters."""
            params = []
            for adapter in self.adapters.values():
                params.extend(adapter.parameters())
            return params

        def save_adapters(self, path: str):
            """Save adapter weights."""
            torch.save(self.adapters.state_dict(), path)

        def load_adapters(self, path: str):
            """Load adapter weights."""
            self.adapters.load_state_dict(torch.load(path))


# =============================================================================
# NumPy Reference Implementation
# =============================================================================

class AdapterLayerNumPy:
    """NumPy reference implementation of an adapter layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        init_scale: float = 1e-3,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.down_weight = np.random.randn(input_dim, hidden_dim) * init_scale
        self.down_bias = np.zeros(hidden_dim)
        self.up_weight = np.random.randn(hidden_dim, input_dim) * init_scale
        self.up_bias = np.zeros(input_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with residual."""
        # Down projection
        h = x @ self.down_weight + self.down_bias

        # ReLU activation
        h = np.maximum(0, h)

        # Up projection
        h = h @ self.up_weight + self.up_bias

        # Residual connection
        return x + h

    def count_parameters(self) -> dict:
        """Count adapter parameters."""
        total = (
            self.input_dim * self.hidden_dim +  # down_weight
            self.hidden_dim +                    # down_bias
            self.hidden_dim * self.input_dim +  # up_weight
            self.input_dim                       # up_bias
        )
        return {
            "total": total,
            "bottleneck_ratio": self.hidden_dim / self.input_dim,
        }


def demonstrate_adapter():
    """Demonstrate adapter architecture."""
    print("Adapter Architecture Demonstration")
    print("=" * 50)

    input_dim = 384  # seq_channel
    hidden_dim = 64  # bottleneck dimension

    adapter = AdapterLayerNumPy(input_dim, hidden_dim)
    params = adapter.count_parameters()

    print(f"Input dimension: {input_dim}")
    print(f"Hidden (bottleneck) dimension: {hidden_dim}")
    print(f"Total adapter parameters: {params['total']:,}")
    print(f"Bottleneck ratio: {params['bottleneck_ratio']:.4f}")

    # Compare to full layer
    full_layer_params = input_dim * input_dim
    print(f"\nFull layer parameters: {full_layer_params:,}")
    print(f"Adapter parameter ratio: {params['total'] / full_layer_params:.4f}")

    # Test forward pass
    x = np.random.randn(10, input_dim)
    y = adapter.forward(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Verify residual connection works
    residual = y - x
    print(f"Residual norm: {np.linalg.norm(residual):.6f}")


if __name__ == "__main__":
    demonstrate_adapter()
