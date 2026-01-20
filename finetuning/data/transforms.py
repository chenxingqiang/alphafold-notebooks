"""Data transformations for protein structure prediction."""

from typing import Dict, Any, Optional
import numpy as np


class StructureAugmentation:
    """Augmentations for protein structures."""
    
    def __init__(
        self,
        random_rotation: bool = True,
        random_translation: bool = True,
        noise_scale: float = 0.0,
    ):
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.noise_scale = noise_scale
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentations to structure."""
        if "coords" in data or "positions" in data:
            key = "coords" if "coords" in data else "positions"
            coords = data[key]
            
            if self.random_rotation:
                coords = self._rotate(coords)
            
            if self.random_translation:
                coords = self._translate(coords)
            
            if self.noise_scale > 0:
                coords = self._add_noise(coords)
            
            data[key] = coords
        
        return data
    
    def _rotate(self, coords: np.ndarray) -> np.ndarray:
        """Apply random 3D rotation."""
        # Generate random rotation matrix using QR decomposition
        random_matrix = np.random.randn(3, 3)
        q, r = np.linalg.qr(random_matrix)
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        
        return coords @ q.T
    
    def _translate(self, coords: np.ndarray) -> np.ndarray:
        """Apply random translation."""
        translation = np.random.randn(3) * 10  # 10 Angstrom scale
        return coords + translation
    
    def _add_noise(self, coords: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to coordinates."""
        noise = np.random.randn(*coords.shape) * self.noise_scale
        return coords + noise


class MSAAugmentation:
    """Augmentations for Multiple Sequence Alignments."""
    
    def __init__(
        self,
        dropout_rate: float = 0.1,
        mask_rate: float = 0.15,
        shuffle_sequences: bool = True,
        max_sequences: Optional[int] = None,
    ):
        self.dropout_rate = dropout_rate
        self.mask_rate = mask_rate
        self.shuffle_sequences = shuffle_sequences
        self.max_sequences = max_sequences
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentations to MSA."""
        if "msa" in data:
            msa = data["msa"]
            
            if self.shuffle_sequences:
                msa = self._shuffle(msa)
            
            if self.max_sequences is not None:
                msa = msa[:self.max_sequences]
            
            if self.dropout_rate > 0:
                msa = self._dropout(msa)
            
            if self.mask_rate > 0:
                msa, mask = self._mask(msa)
                data["bert_mask"] = mask
            
            data["msa"] = msa
        
        return data
    
    def _shuffle(self, msa: np.ndarray) -> np.ndarray:
        """Shuffle MSA sequences (keep first sequence fixed)."""
        if len(msa) <= 1:
            return msa
        
        indices = np.random.permutation(len(msa) - 1) + 1
        indices = np.concatenate([[0], indices])
        return msa[indices]
    
    def _dropout(self, msa: np.ndarray) -> np.ndarray:
        """Randomly drop sequences from MSA."""
        if len(msa) <= 1:
            return msa
        
        keep_mask = np.random.rand(len(msa)) > self.dropout_rate
        keep_mask[0] = True  # Always keep first sequence
        return msa[keep_mask]
    
    def _mask(self, msa: np.ndarray) -> tuple:
        """Mask residues for BERT-style pretraining."""
        mask = np.random.rand(*msa.shape) < self.mask_rate
        mask[0] = False  # Don't mask query sequence
        
        masked_msa = msa.copy()
        masked_msa[mask] = 21  # BERT mask token
        
        return masked_msa, mask


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data
