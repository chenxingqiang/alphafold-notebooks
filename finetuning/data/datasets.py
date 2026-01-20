"""Dataset classes for fine-tuning."""

from typing import Optional, Dict, Any, List, Callable
import os

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


if TORCH_AVAILABLE:
    
    class ProteinDataset(Dataset):
        """Base dataset for protein structures."""
        
        def __init__(
            self,
            data_path: str,
            transform: Optional[Callable] = None,
            max_length: int = 512,
        ):
            self.data_path = data_path
            self.transform = transform
            self.max_length = max_length
            
            # Load data index
            self.samples = self._load_index()
        
        def _load_index(self) -> List[str]:
            """Load list of sample paths."""
            if os.path.isdir(self.data_path):
                return [os.path.join(self.data_path, f) 
                        for f in os.listdir(self.data_path) 
                        if f.endswith(('.npz', '.pt', '.pdb'))]
            else:
                # Assume it's an index file
                with open(self.data_path, 'r') as f:
                    return [line.strip() for line in f]
        
        def __len__(self) -> int:
            return len(self.samples)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            path = self.samples[idx]
            
            if path.endswith('.npz'):
                data = dict(np.load(path, allow_pickle=True))
            elif path.endswith('.pt'):
                data = torch.load(path)
            else:
                data = self._load_pdb(path)
            
            # Apply transforms
            if self.transform is not None:
                data = self.transform(data)
            
            # Convert to tensors
            data = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v 
                    for k, v in data.items()}
            
            return data
        
        def _load_pdb(self, path: str) -> Dict[str, np.ndarray]:
            """Load PDB file (placeholder)."""
            # In practice, use a PDB parser
            return {"path": path}
    
    
    class AffinityDataset(ProteinDataset):
        """Dataset for binding affinity prediction."""
        
        def __init__(
            self,
            data_path: str,
            affinity_file: Optional[str] = None,
            transform: Optional[Callable] = None,
            max_length: int = 512,
        ):
            super().__init__(data_path, transform, max_length)
            
            # Load affinity labels
            self.affinities = {}
            if affinity_file is not None:
                self._load_affinities(affinity_file)
        
        def _load_affinities(self, path: str):
            """Load affinity values from file."""
            import csv
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.affinities[row['id']] = float(row['affinity'])
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            data = super().__getitem__(idx)
            
            # Add affinity label
            sample_id = os.path.basename(self.samples[idx]).split('.')[0]
            if sample_id in self.affinities:
                data['affinity_value'] = torch.tensor(self.affinities[sample_id])
                data['has_affinity'] = torch.tensor(1.0)
            else:
                data['affinity_value'] = torch.tensor(0.0)
                data['has_affinity'] = torch.tensor(0.0)
            
            return data
    
    
    class PropertyDataset(ProteinDataset):
        """Dataset for protein property prediction."""
        
        def __init__(
            self,
            data_path: str,
            property_file: Optional[str] = None,
            property_names: List[str] = None,
            transform: Optional[Callable] = None,
            max_length: int = 512,
        ):
            super().__init__(data_path, transform, max_length)
            
            self.property_names = property_names or ["stability"]
            self.properties = {}
            
            if property_file is not None:
                self._load_properties(property_file)
        
        def _load_properties(self, path: str):
            """Load property values."""
            import csv
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    props = {name: float(row.get(name, 0)) 
                             for name in self.property_names}
                    self.properties[row['id']] = props
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            data = super().__getitem__(idx)
            
            sample_id = os.path.basename(self.samples[idx]).split('.')[0]
            if sample_id in self.properties:
                props = self.properties[sample_id]
                data['targets'] = torch.tensor([props[name] for name in self.property_names])
            else:
                data['targets'] = torch.zeros(len(self.property_names))
            
            return data


# NumPy reference
class ProteinDatasetNumPy:
    """NumPy reference dataset."""
    
    def __init__(self, sequences: List[str], labels: Optional[np.ndarray] = None):
        self.sequences = sequences
        self.labels = labels
        
        # Amino acid vocabulary
        self.vocab = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        seq = self.sequences[idx]
        
        # Encode sequence
        encoded = np.array([self.vocab.get(aa, 20) for aa in seq])
        
        data = {"sequence": encoded}
        
        if self.labels is not None:
            data["label"] = self.labels[idx]
        
        return data
