"""Training callbacks."""

from typing import Optional, Any, Dict
import os
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer: Any):
        pass
    
    def on_train_end(self, trainer: Any):
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int):
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        pass
    
    def on_step_begin(self, trainer: Any, step: int):
        pass
    
    def on_step_end(self, trainer: Any, step: int, logs: Dict[str, float]):
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping based on validation metric."""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered after {epoch} epochs")


class ModelCheckpointCallback(Callback):
    """Save model checkpoints."""
    
    def __init__(
        self,
        save_path: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
    ):
        self.save_path = save_path
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        
        os.makedirs(save_path, exist_ok=True)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        
        if self.save_best_only:
            if current is None:
                return
            
            if self.mode == "min":
                improved = current < self.best_value
            else:
                improved = current > self.best_value
            
            if improved:
                self.best_value = current
                trainer.save_checkpoint(os.path.join(self.save_path, "best"))
        else:
            trainer.save_checkpoint(os.path.join(self.save_path, f"epoch_{epoch}"))


class WandbCallback(Callback):
    """Weights & Biases logging callback."""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        try:
            import wandb
            self.wandb = wandb
            self.enabled = True
        except ImportError:
            self.wandb = None
            self.enabled = False
            print("wandb not installed, logging disabled")
        
        self.project = project
        self.name = name
        self.config = config
    
    def on_train_begin(self, trainer: Any):
        if not self.enabled:
            return
        
        self.wandb.init(
            project=self.project,
            name=self.name,
            config=self.config or trainer.config.to_dict(),
        )
    
    def on_step_end(self, trainer: Any, step: int, logs: Dict[str, float]):
        if not self.enabled:
            return
        
        self.wandb.log(logs, step=step)
    
    def on_train_end(self, trainer: Any):
        if not self.enabled:
            return
        
        self.wandb.finish()
