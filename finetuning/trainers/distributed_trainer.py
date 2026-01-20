"""Distributed trainer for multi-GPU training."""

from typing import Optional, Any, List
import os

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    
    class DistributedTrainer:
        """Distributed trainer using PyTorch DDP."""
        
        def __init__(
            self,
            model: nn.Module,
            config: Any,
            train_dataloader: Any = None,
            val_dataloader: Any = None,
            callbacks: Optional[List[Any]] = None,
        ):
            self.config = config
            self.callbacks = callbacks or []
            
            # Initialize distributed
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.rank = int(os.environ.get("RANK", 0))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            if self.world_size > 1:
                dist.init_process_group(backend="nccl")
                torch.cuda.set_device(self.local_rank)
            
            # Setup model
            self.device = torch.device(f"cuda:{self.local_rank}")
            model = model.to(self.device)
            
            if self.world_size > 1:
                self.model = DDP(model, device_ids=[self.local_rank])
            else:
                self.model = model
            
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            
            self.is_main = self.rank == 0
        
        def log(self, msg: str):
            """Log only from main process."""
            if self.is_main:
                print(msg)
        
        def train(self, num_steps: int):
            """Distributed training loop."""
            from .trainer import Trainer
            
            # Use base trainer for actual training logic
            base_trainer = Trainer(
                model=self.model,
                config=self.config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                callbacks=self.callbacks,
            )
            
            return base_trainer.train(num_steps=num_steps)
        
        def cleanup(self):
            """Cleanup distributed resources."""
            if self.world_size > 1:
                dist.destroy_process_group()
