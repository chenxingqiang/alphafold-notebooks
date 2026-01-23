"""Prompt tuning module for efficient fine-tuning.

Prompt tuning learns a small set of continuous embeddings (soft prompts)
that are prepended to the input, while keeping the model frozen.

Reference: "The Power of Scale for Parameter-Efficient Prompt Tuning"
           https://arxiv.org/abs/2104.08691
"""

from typing import Optional, List
import math

# PyTorch implementation
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


if TORCH_AVAILABLE:

    class SoftPrompt(nn.Module):
        """Learnable soft prompt embeddings."""

        def __init__(
            self,
            num_tokens: int = 10,
            embed_dim: int = 384,
            init_from_vocab: bool = False,
            init_text: Optional[str] = None,
            tokenizer: Optional[object] = None,
        ):
            super().__init__()

            self.num_tokens = num_tokens
            self.embed_dim = embed_dim

            # Initialize prompt embeddings
            if init_from_vocab and tokenizer is not None and init_text is not None:
                # Initialize from text tokens
                tokens = tokenizer.encode(init_text)[:num_tokens]
                # Pad if necessary
                while len(tokens) < num_tokens:
                    tokens.append(0)
                self.prompt_embeddings = nn.Parameter(
                    torch.zeros(num_tokens, embed_dim)
                )
            else:
                # Random initialization
                self.prompt_embeddings = nn.Parameter(
                    torch.randn(num_tokens, embed_dim) * 0.02
                )

        def forward(self, batch_size: int) -> torch.Tensor:
            """Get prompt embeddings for a batch.

            Returns: [batch_size, num_tokens, embed_dim]
            """
            return self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)


    class PromptTuning(nn.Module):
        """Prompt tuning wrapper for a model.

        Prepends learnable prompt tokens to the input sequence.
        """

        def __init__(
            self,
            model: nn.Module,
            num_prompt_tokens: int = 10,
            embed_dim: int = 384,
            prompt_position: str = "prefix",  # "prefix" or "suffix"
        ):
            super().__init__()

            self.model = model
            self.num_prompt_tokens = num_prompt_tokens
            self.embed_dim = embed_dim
            self.prompt_position = prompt_position

            # Freeze the model
            for param in self.model.parameters():
                param.requires_grad = False

            # Initialize soft prompts
            self.soft_prompt = SoftPrompt(
                num_tokens=num_prompt_tokens,
                embed_dim=embed_dim,
            )

        def forward(
            self,
            input_embeds: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            """Forward pass with prompt prepending.

            Args:
                input_embeds: Input embeddings [batch, seq_len, embed_dim]
                attention_mask: Attention mask [batch, seq_len]
                **kwargs: Additional arguments passed to model

            Returns:
                Model output
            """
            batch_size = input_embeds.shape[0]

            # Get prompt embeddings
            prompt_embeds = self.soft_prompt(batch_size)

            # Concatenate with input
            if self.prompt_position == "prefix":
                combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
            else:  # suffix
                combined_embeds = torch.cat([input_embeds, prompt_embeds], dim=1)

            # Update attention mask if provided
            if attention_mask is not None:
                prompt_mask = torch.ones(
                    batch_size, self.num_prompt_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                if self.prompt_position == "prefix":
                    combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
                else:
                    combined_mask = torch.cat([attention_mask, prompt_mask], dim=1)
                kwargs["attention_mask"] = combined_mask

            # Forward through model
            return self.model(combined_embeds, **kwargs)

        def get_trainable_parameters(self) -> List[nn.Parameter]:
            """Get only the trainable prompt parameters."""
            return [self.soft_prompt.prompt_embeddings]

        def save_prompts(self, path: str):
            """Save prompt embeddings."""
            torch.save(self.soft_prompt.state_dict(), path)

        def load_prompts(self, path: str):
            """Load prompt embeddings."""
            self.soft_prompt.load_state_dict(torch.load(path))


# =============================================================================
# Protein-Specific Prompt Tuning
# =============================================================================

if TORCH_AVAILABLE:

    class ProteinPromptTuning(nn.Module):
        """Prompt tuning specialized for protein structure prediction.

        Learns task-specific prompts that can be:
        - Prepended to MSA representations
        - Added to pair representations
        - Injected into single representations
        """

        def __init__(
            self,
            msa_channel: int = 256,
            pair_channel: int = 128,
            seq_channel: int = 384,
            num_msa_prompts: int = 4,
            num_pair_prompts: int = 4,
            num_single_prompts: int = 8,
        ):
            super().__init__()

            # MSA prompts (prepended as additional sequences)
            self.msa_prompts = nn.Parameter(
                torch.randn(num_msa_prompts, msa_channel) * 0.02
            )

            # Pair prompts (bias added to pair representations)
            self.pair_prompts = nn.Parameter(
                torch.randn(num_pair_prompts, pair_channel) * 0.02
            )

            # Single prompts (prepended to single representation)
            self.single_prompts = nn.Parameter(
                torch.randn(num_single_prompts, seq_channel) * 0.02
            )

            # Task-specific embeddings
            self.task_embedding = nn.Embedding(10, seq_channel)  # Up to 10 tasks

        def get_msa_prompts(self, num_res: int) -> torch.Tensor:
            """Get MSA prompts expanded to residue length.

            Returns: [num_msa_prompts, num_res, msa_channel]
            """
            return self.msa_prompts.unsqueeze(1).expand(-1, num_res, -1)

        def get_pair_bias(self, num_res: int) -> torch.Tensor:
            """Get pair representation bias.

            Returns: [num_res, num_res, pair_channel]
            """
            # Use outer sum of prompts as pair bias
            bias = self.pair_prompts[None, :, :] + self.pair_prompts[:, None, :]
            return bias.mean(dim=0).unsqueeze(0).expand(num_res, num_res, -1)

        def get_single_prompts(self, batch_size: int) -> torch.Tensor:
            """Get single representation prompts.

            Returns: [batch_size, num_single_prompts, seq_channel]
            """
            return self.single_prompts.unsqueeze(0).expand(batch_size, -1, -1)


# =============================================================================
# NumPy Reference Implementation
# =============================================================================

class SoftPromptNumPy:
    """NumPy reference implementation of soft prompts."""

    def __init__(
        self,
        num_tokens: int = 10,
        embed_dim: int = 384,
    ):
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # Initialize prompt embeddings
        self.prompt_embeddings = np.random.randn(num_tokens, embed_dim) * 0.02

    def forward(self, batch_size: int) -> np.ndarray:
        """Get prompt embeddings for a batch."""
        return np.tile(
            self.prompt_embeddings[np.newaxis, :, :],
            (batch_size, 1, 1)
        )

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return self.num_tokens * self.embed_dim


def demonstrate_prompt_tuning():
    """Demonstrate prompt tuning."""
    print("Prompt Tuning Demonstration")
    print("=" * 50)

    num_tokens = 10
    embed_dim = 384
    seq_len = 100

    soft_prompt = SoftPromptNumPy(num_tokens, embed_dim)
    num_params = soft_prompt.count_parameters()

    print(f"Number of prompt tokens: {num_tokens}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Total prompt parameters: {num_params:,}")

    # Compare to typical model size
    evoformer_block_params = 2_000_000  # Approximate
    print(f"\nApproximate Evoformer block params: {evoformer_block_params:,}")
    print(f"Prompt params / Block params: {num_params / evoformer_block_params:.6f}")

    # Test forward
    batch_size = 4
    prompts = soft_prompt.forward(batch_size)
    print(f"\nPrompt shape: {prompts.shape}")

    # Simulate prepending to input
    input_embeds = np.random.randn(batch_size, seq_len, embed_dim)
    combined = np.concatenate([prompts, input_embeds], axis=1)
    print(f"Combined shape: {combined.shape}")
    print(f"New sequence length: {combined.shape[1]} (was {seq_len})")


if __name__ == "__main__":
    demonstrate_prompt_tuning()
