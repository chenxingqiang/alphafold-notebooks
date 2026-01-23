"""Trainer for fine-tuning protein structure prediction models."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Iterator
import os
import json
import time
import math

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


@dataclass
class TrainingState:
    """State of training."""

    global_step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")
    best_step: int = 0

    # Metrics history
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "metrics_history": self.metrics_history,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        return cls(**d)


if TORCH_AVAILABLE:

    class Trainer:
        """Trainer for fine-tuning models.

        Supports:
        - Mixed precision training
        - Gradient accumulation
        - Learning rate scheduling
        - Checkpoint saving/loading
        - Metrics logging
        """

        def __init__(
            self,
            model: nn.Module,
            config: Any,  # FineTuningConfig
            train_dataloader: Optional[DataLoader] = None,
            val_dataloader: Optional[DataLoader] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[Any] = None,
            callbacks: Optional[List[Any]] = None,
        ):
            self.model = model
            self.config = config
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            self.callbacks = callbacks or []

            # Device
            self.device = torch.device(
                config.model.device if hasattr(config, 'model') else "cuda"
            )
            self.model.to(self.device)

            # Optimizer
            self.optimizer = optimizer or self._create_optimizer()

            # Scheduler
            self.scheduler = scheduler or self._create_scheduler()

            # Mixed precision
            self.scaler = torch.amp.GradScaler('cuda') if self._use_amp() else None

            # Training state
            self.state = TrainingState()

            # Output directory
            self.output_dir = config.training.output_dir if hasattr(config, 'training') else "./output"
            os.makedirs(self.output_dir, exist_ok=True)

        def _use_amp(self) -> bool:
            """Whether to use automatic mixed precision."""
            precision = getattr(getattr(self.config, 'model', None), 'precision', 'fp32')
            return precision in ("fp16", "bf16") and torch.cuda.is_available()

        def _create_optimizer(self) -> torch.optim.Optimizer:
            """Create optimizer with weight decay."""
            training_config = getattr(self.config, 'training', self.config)

            # Get trainable parameters
            params = [p for p in self.model.parameters() if p.requires_grad]

            # Separate parameters for weight decay
            decay_params = []
            no_decay_params = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "bias" in name or "norm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            optimizer_grouped_params = [
                {"params": decay_params, "weight_decay": training_config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]

            return AdamW(
                optimizer_grouped_params,
                lr=training_config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        def _create_scheduler(self):
            """Create learning rate scheduler."""
            training_config = getattr(self.config, 'training', self.config)

            warmup_steps = training_config.warmup_steps
            max_steps = training_config.max_steps

            # Warmup scheduler
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )

            # Main scheduler
            if training_config.lr_scheduler == "cosine":
                main_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=max_steps - warmup_steps,
                    eta_min=training_config.learning_rate * 0.01,
                )
            else:
                # Linear decay
                main_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=max_steps - warmup_steps,
                )

            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )

        def _get_trainable_parameters_count(self) -> Dict[str, int]:
            """Count trainable parameters."""
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            return {
                "total": total,
                "trainable": trainable,
                "ratio": trainable / total if total > 0 else 0,
            }

        def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
            """Single training step."""
            self.model.train()

            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=self._use_amp()):
                outputs = self.model(**batch)
                loss = outputs.get("loss", outputs.get("total_loss", torch.tensor(0.0)))

            # Scale loss for gradient accumulation
            training_config = getattr(self.config, 'training', self.config)
            loss = loss / training_config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            return {
                "loss": loss.item() * training_config.gradient_accumulation_steps,
                **{k: v.item() if isinstance(v, torch.Tensor) else v
                   for k, v in outputs.items() if k != "loss"},
            }

        def optimizer_step(self):
            """Optimizer step with gradient clipping."""
            training_config = getattr(self.config, 'training', self.config)

            # Unscale gradients for clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                training_config.max_grad_norm,
            )

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Scheduler step
            self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

        @torch.no_grad()
        def evaluate(self) -> Dict[str, float]:
            """Evaluate on validation set."""
            if self.val_dataloader is None:
                return {}

            self.model.eval()

            total_loss = 0.0
            total_samples = 0
            all_metrics = {}

            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                with torch.amp.autocast('cuda', enabled=self._use_amp()):
                    outputs = self.model(**batch)

                loss = outputs.get("loss", outputs.get("total_loss", torch.tensor(0.0)))
                batch_size = next(iter(batch.values())).shape[0]

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Accumulate metrics
                for k, v in outputs.items():
                    if k != "loss" and isinstance(v, (int, float, torch.Tensor)):
                        if k not in all_metrics:
                            all_metrics[k] = 0.0
                        all_metrics[k] += (v.item() if isinstance(v, torch.Tensor) else v) * batch_size

            # Average metrics
            avg_metrics = {
                "val_loss": total_loss / total_samples,
                **{f"val_{k}": v / total_samples for k, v in all_metrics.items()},
            }

            return avg_metrics

        def train(
            self,
            num_steps: Optional[int] = None,
            num_epochs: Optional[int] = None,
        ):
            """Run training loop."""
            training_config = getattr(self.config, 'training', self.config)

            num_steps = num_steps or training_config.max_steps

            # Log training info
            param_count = self._get_trainable_parameters_count()
            print(f"Training started")
            print(f"  Total parameters: {param_count['total']:,}")
            print(f"  Trainable parameters: {param_count['trainable']:,}")
            print(f"  Trainable ratio: {param_count['ratio']:.4f}")
            print(f"  Max steps: {num_steps}")

            # Call callbacks
            for callback in self.callbacks:
                if hasattr(callback, "on_train_begin"):
                    callback.on_train_begin(self)

            # Training loop
            accumulation_steps = 0
            epoch_loss = 0.0

            while self.state.global_step < num_steps:
                self.state.epoch += 1

                for batch in self.train_dataloader:
                    # Training step
                    step_metrics = self.train_step(batch)
                    epoch_loss += step_metrics["loss"]
                    accumulation_steps += 1

                    # Optimizer step after accumulation
                    if accumulation_steps >= training_config.gradient_accumulation_steps:
                        self.optimizer_step()
                        self.state.global_step += 1
                        accumulation_steps = 0

                        # Logging
                        if self.state.global_step % training_config.logging_steps == 0:
                            avg_loss = epoch_loss / training_config.logging_steps
                            lr = self.scheduler.get_last_lr()[0]
                            print(f"Step {self.state.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                            self.state.train_losses.append(avg_loss)
                            epoch_loss = 0.0

                        # Evaluation
                        if (self.state.global_step % training_config.eval_steps == 0
                            and self.val_dataloader is not None):
                            val_metrics = self.evaluate()
                            print(f"  Validation: {val_metrics}")

                            # Track best
                            if val_metrics.get("val_loss", float("inf")) < self.state.best_metric:
                                self.state.best_metric = val_metrics["val_loss"]
                                self.state.best_step = self.state.global_step
                                self.save_checkpoint("best")

                        # Save checkpoint
                        if self.state.global_step % training_config.save_steps == 0:
                            self.save_checkpoint(f"step_{self.state.global_step}")

                        # Check if done
                        if self.state.global_step >= num_steps:
                            break

                if self.state.global_step >= num_steps:
                    break

            # Final save
            self.save_checkpoint("final")

            # Call callbacks
            for callback in self.callbacks:
                if hasattr(callback, "on_train_end"):
                    callback.on_train_end(self)

            print(f"Training completed. Best step: {self.state.best_step}")
            return self.state

        def save_checkpoint(self, name: str):
            """Save checkpoint."""
            checkpoint_dir = os.path.join(self.output_dir, name)
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save model
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

            # Save optimizer
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

            # Save scheduler
            torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

            # Save state
            with open(os.path.join(checkpoint_dir, "state.json"), "w") as f:
                json.dump(self.state.to_dict(), f)

            print(f"Saved checkpoint to {checkpoint_dir}")

        def load_checkpoint(self, path: str):
            """Load checkpoint."""
            self.model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
            self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt")))
            self.scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt")))

            with open(os.path.join(path, "state.json"), "r") as f:
                self.state = TrainingState.from_dict(json.load(f))

            print(f"Loaded checkpoint from {path}")


# =============================================================================
# NumPy Reference Trainer (for educational purposes)
# =============================================================================

class TrainerNumPy:
    """Simplified NumPy trainer for educational purposes."""

    def __init__(
        self,
        model_params: Dict[str, np.ndarray],
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.params = model_params
        self.lr = learning_rate
        self.weight_decay = weight_decay

        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in model_params.items()}
        self.v = {k: np.zeros_like(v) for k, v in model_params.items()}
        self.t = 0

    def adam_step(self, grads: Dict[str, np.ndarray], beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer step."""
        self.t += 1

        for key in self.params:
            if key not in grads:
                continue

            g = grads[key]

            # Momentum
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g

            # RMS
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)

            # Update
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            # Weight decay
            self.params[key] -= self.lr * self.weight_decay * self.params[key]

    def train_step(self, x: np.ndarray, y: np.ndarray, forward_fn, loss_fn, grad_fn):
        """Single training step."""
        # Forward
        pred = forward_fn(x, self.params)
        loss = loss_fn(pred, y)

        # Backward (numerical gradient for simplicity)
        grads = grad_fn(x, y, self.params)

        # Update
        self.adam_step(grads)

        return loss


def demonstrate_trainer():
    """Demonstrate trainer functionality."""
    print("Trainer Demonstration")
    print("=" * 50)

    # Dummy model parameters
    params = {
        "w1": np.random.randn(10, 5) * 0.1,
        "b1": np.zeros(5),
        "w2": np.random.randn(5, 1) * 0.1,
        "b2": np.zeros(1),
    }

    trainer = TrainerNumPy(params, learning_rate=0.01)

    print(f"Initial parameters:")
    print(f"  w1 norm: {np.linalg.norm(params['w1']):.4f}")
    print(f"  w2 norm: {np.linalg.norm(params['w2']):.4f}")

    # Simulate gradients
    grads = {
        "w1": np.random.randn(10, 5) * 0.01,
        "b1": np.random.randn(5) * 0.01,
        "w2": np.random.randn(5, 1) * 0.01,
        "b2": np.random.randn(1) * 0.01,
    }

    trainer.adam_step(grads)

    print(f"\nAfter one Adam step:")
    print(f"  w1 norm: {np.linalg.norm(trainer.params['w1']):.4f}")
    print(f"  w2 norm: {np.linalg.norm(trainer.params['w2']):.4f}")


if __name__ == "__main__":
    demonstrate_trainer()
