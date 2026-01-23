"""Checkpoint utilities."""

from typing import Optional, Dict, Any
import os

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_pretrained(
    model: Any,
    checkpoint_path: str,
    strict: bool = False,
    map_location: Optional[str] = None,
) -> Any:
    """Load pretrained weights into model.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Whether to require exact match
        map_location: Device to map tensors to

    Returns:
        Model with loaded weights
    """
    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        state_dict = torch.load(checkpoint_path, map_location=map_location)

        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Remove DDP prefix if present
        cleaned_state = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]
            cleaned_state[key] = value

        model.load_state_dict(cleaned_state, strict=strict)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    return model


def save_checkpoint(
    model: Any,
    path: str,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
):
    """Save model checkpoint.

    Args:
        model: Model to save
        path: Save path
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch
        step: Current step
        metrics: Optional metrics to save
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
