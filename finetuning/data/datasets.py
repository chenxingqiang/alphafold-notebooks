"""Dataset classes for fine-tuning protein structure prediction models.

Comprehensive dataset support for all task types:
- Drug Discovery: Binding affinity, virtual screening
- Protein Engineering: Stability, solubility, mutation effects
- Antibody Design: CDR optimization, humanization
- Enzyme Engineering: Activity, specificity
- Protein-Protein Interaction: Binding, interface
- Function Prediction: GO terms, EC numbers
- Immunology: Epitopes, immunogenicity
"""

from typing import Optional, Dict, Any, List, Callable, Tuple
from abc import ABC, abstractmethod
import os
import json

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


# =============================================================================
# Base Dataset Classes
# =============================================================================

if TORCH_AVAILABLE:
    
    class ProteinDataset(Dataset):
        """Base dataset for protein structures."""
        
        def __init__(
            self,
            data_path: str,
            transform: Optional[Callable] = None,
            max_length: int = 512,
            cache_data: bool = True,
        ):
            self.data_path = data_path
            self.transform = transform
            self.max_length = max_length
            self.cache_data = cache_data
            
            self.samples = self._load_index()
            self._cache = {} if cache_data else None
        
        def _load_index(self) -> List[str]:
            """Load list of sample paths."""
            if os.path.isdir(self.data_path):
                extensions = ('.npz', '.pt', '.pdb', '.cif', '.json')
                samples = [
                    os.path.join(self.data_path, f) 
                    for f in os.listdir(self.data_path) 
                    if f.endswith(extensions)
                ]
                return sorted(samples)
            elif os.path.isfile(self.data_path):
                with open(self.data_path, 'r') as f:
                    if self.data_path.endswith('.json'):
                        data = json.load(f)
                        return data if isinstance(data, list) else list(data.keys())
                    else:
                        return [line.strip() for line in f if line.strip()]
            return []
        
        def __len__(self) -> int:
            return len(self.samples)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            if self._cache is not None and idx in self._cache:
                return self._cache[idx]
            
            sample_path = self.samples[idx]
            data = self._load_sample(sample_path)
            
            if self.transform is not None:
                data = self.transform(data)
            
            # Convert to tensors
            data = self._to_tensors(data)
            
            if self._cache is not None:
                self._cache[idx] = data
            
            return data
        
        def _load_sample(self, path: str) -> Dict[str, Any]:
            """Load a single sample."""
            if path.endswith('.npz'):
                return dict(np.load(path, allow_pickle=True))
            elif path.endswith('.pt'):
                return torch.load(path)
            elif path.endswith('.json'):
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                return {"path": path}
        
        def _to_tensors(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
            """Convert numpy arrays to tensors."""
            result = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    result[k] = torch.from_numpy(v)
                elif isinstance(v, (int, float)):
                    result[k] = torch.tensor(v)
                elif isinstance(v, torch.Tensor):
                    result[k] = v
                else:
                    result[k] = v
            return result


# =============================================================================
# Drug Discovery Datasets
# =============================================================================

if TORCH_AVAILABLE:
    
    class AffinityDataset(ProteinDataset):
        """Dataset for protein-ligand binding affinity prediction.
        
        Supports: PDBbind, BindingDB, ChEMBL data formats.
        """
        
        def __init__(
            self,
            data_path: str,
            affinity_file: Optional[str] = None,
            affinity_type: str = "pic50",  # pkd, pki, pic50, delta_g
            include_ligand: bool = True,
            transform: Optional[Callable] = None,
            max_length: int = 512,
        ):
            super().__init__(data_path, transform, max_length)
            
            self.affinity_type = affinity_type
            self.include_ligand = include_ligand
            
            # Load affinity labels
            self.affinities = {}
            self.ligands = {}
            if affinity_file:
                self._load_affinities(affinity_file)
        
        def _load_affinities(self, path: str):
            """Load affinity values and ligand info."""
            if path.endswith('.csv'):
                import csv
                with open(path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sample_id = row.get('id') or row.get('pdb_id')
                        if sample_id:
                            self.affinities[sample_id] = float(row.get(self.affinity_type, row.get('affinity', 0)))
                            if 'smiles' in row:
                                self.ligands[sample_id] = row['smiles']
            elif path.endswith('.json'):
                with open(path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        sample_id = item.get('id')
                        if sample_id:
                            self.affinities[sample_id] = item.get(self.affinity_type, item.get('affinity', 0))
                            if 'smiles' in item:
                                self.ligands[sample_id] = item['smiles']
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            data = super().__getitem__(idx)
            
            sample_id = os.path.basename(self.samples[idx]).split('.')[0]
            
            if sample_id in self.affinities:
                data['affinity_value'] = torch.tensor(self.affinities[sample_id], dtype=torch.float32)
                data['has_affinity'] = torch.tensor(1.0)
            else:
                data['affinity_value'] = torch.tensor(0.0)
                data['has_affinity'] = torch.tensor(0.0)
            
            if sample_id in self.ligands:
                data['ligand_smiles'] = self.ligands[sample_id]
            
            return data
    
    
    class VirtualScreeningDataset(ProteinDataset):
        """Dataset for virtual screening / hit identification."""
        
        def __init__(
            self,
            data_path: str,
            compounds_file: str,
            activity_threshold: float = 6.0,
            include_decoys: bool = True,
            transform: Optional[Callable] = None,
        ):
            super().__init__(data_path, transform)
            
            self.activity_threshold = activity_threshold
            self.compounds = self._load_compounds(compounds_file)
        
        def _load_compounds(self, path: str) -> List[Dict]:
            """Load compound library."""
            with open(path, 'r') as f:
                return json.load(f)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            data = super().__getitem__(idx)
            compound = self.compounds[idx % len(self.compounds)]
            
            data['compound_smiles'] = compound.get('smiles', '')
            data['is_active'] = torch.tensor(
                1.0 if compound.get('activity', 0) > self.activity_threshold else 0.0
            )
            
            return data


# =============================================================================
# Protein Engineering Datasets
# =============================================================================

if TORCH_AVAILABLE:
    
    class StabilityDataset(ProteinDataset):
        """Dataset for protein stability prediction."""
        
        def __init__(
            self,
            data_path: str,
            labels_file: Optional[str] = None,
            stability_type: str = "ddg",
            transform: Optional[Callable] = None,
        ):
            super().__init__(data_path, transform)
            
            self.stability_type = stability_type
            self.labels = {}
            if labels_file:
                self._load_labels(labels_file)
        
        def _load_labels(self, path: str):
            """Load stability labels."""
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.labels[item['id']] = {
                        'ddg': item.get('ddg', 0),
                        'tm': item.get('tm', 0),
                        'mutations': item.get('mutations', []),
                    }
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            data = super().__getitem__(idx)
            sample_id = os.path.basename(self.samples[idx]).split('.')[0]
            
            if sample_id in self.labels:
                label = self.labels[sample_id]
                data['ddg'] = torch.tensor(label['ddg'], dtype=torch.float32)
                data['mutations'] = label['mutations']
            
            return data
    
    
    class MutationDataset(ProteinDataset):
        """Dataset for mutation effect prediction."""
        
        def __init__(
            self,
            data_path: str,
            mutations_file: str,
            effect_type: str = "ddg",
            transform: Optional[Callable] = None,
        ):
            super().__init__(data_path, transform)
            
            self.effect_type = effect_type
            self.mutations = self._load_mutations(mutations_file)
        
        def _load_mutations(self, path: str) -> List[Dict]:
            """Load mutation data."""
            with open(path, 'r') as f:
                return json.load(f)
        
        def __len__(self) -> int:
            return len(self.mutations)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            mutation = self.mutations[idx]
            
            # Load wild-type structure
            wt_path = mutation.get('wt_structure')
            if wt_path and os.path.exists(wt_path):
                data = self._load_sample(wt_path)
                data = self._to_tensors(data)
            else:
                data = {}
            
            # Add mutation info
            data['mutation_positions'] = torch.tensor(mutation.get('positions', []))
            data['mutation_wt_aa'] = mutation.get('wt_aa', [])
            data['mutation_mt_aa'] = mutation.get('mt_aa', [])
            data['effect'] = torch.tensor(mutation.get(self.effect_type, 0), dtype=torch.float32)
            
            return data


# =============================================================================
# Antibody Datasets
# =============================================================================

if TORCH_AVAILABLE:
    
    class AntibodyDataset(ProteinDataset):
        """Dataset for antibody sequences and structures."""
        
        def __init__(
            self,
            data_path: str,
            annotations_file: Optional[str] = None,
            include_antigen: bool = True,
            transform: Optional[Callable] = None,
        ):
            super().__init__(data_path, transform)
            
            self.include_antigen = include_antigen
            self.annotations = {}
            if annotations_file:
                self._load_annotations(annotations_file)
        
        def _load_annotations(self, path: str):
            """Load antibody annotations (CDR regions, species, etc.)."""
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.annotations[item['id']] = {
                        'cdr_h1': item.get('cdr_h1', []),
                        'cdr_h2': item.get('cdr_h2', []),
                        'cdr_h3': item.get('cdr_h3', []),
                        'cdr_l1': item.get('cdr_l1', []),
                        'cdr_l2': item.get('cdr_l2', []),
                        'cdr_l3': item.get('cdr_l3', []),
                        'species': item.get('species', 'unknown'),
                        'affinity': item.get('affinity', None),
                    }
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            data = super().__getitem__(idx)
            sample_id = os.path.basename(self.samples[idx]).split('.')[0]
            
            if sample_id in self.annotations:
                ann = self.annotations[sample_id]
                
                # Create CDR mask
                if 'num_res' in data:
                    num_res = data['num_res']
                else:
                    num_res = data.get('single', data.get('sequence', np.zeros(100))).shape[0]
                
                cdr_mask = torch.zeros(num_res)
                cdr_types = torch.full((num_res,), -1, dtype=torch.long)
                
                for i, cdr_name in enumerate(['cdr_h1', 'cdr_h2', 'cdr_h3', 'cdr_l1', 'cdr_l2', 'cdr_l3']):
                    cdr_region = ann.get(cdr_name, [])
                    if len(cdr_region) == 2:
                        start, end = cdr_region
                        cdr_mask[start:end] = 1
                        cdr_types[start:end] = i
                
                data['cdr_mask'] = cdr_mask
                data['cdr_types'] = cdr_types
                
                if ann.get('affinity') is not None:
                    data['affinity'] = torch.tensor(ann['affinity'], dtype=torch.float32)
            
            return data


# =============================================================================
# PPI Datasets
# =============================================================================

if TORCH_AVAILABLE:
    
    class PPIDataset(ProteinDataset):
        """Dataset for protein-protein interaction data."""
        
        def __init__(
            self,
            data_path: str,
            interactions_file: Optional[str] = None,
            include_negatives: bool = True,
            transform: Optional[Callable] = None,
        ):
            super().__init__(data_path, transform)
            
            self.interactions = []
            if interactions_file:
                self._load_interactions(interactions_file)
        
        def _load_interactions(self, path: str):
            """Load PPI data."""
            with open(path, 'r') as f:
                self.interactions = json.load(f)
        
        def __len__(self) -> int:
            return len(self.interactions) if self.interactions else len(self.samples)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            if self.interactions:
                interaction = self.interactions[idx]
                
                # Load complex structure
                complex_path = interaction.get('complex_path')
                if complex_path and os.path.exists(complex_path):
                    data = self._load_sample(complex_path)
                    data = self._to_tensors(data)
                else:
                    data = {}
                
                # Add chain masks
                data['chain_a_mask'] = torch.tensor(interaction.get('chain_a_residues', []))
                data['chain_b_mask'] = torch.tensor(interaction.get('chain_b_residues', []))
                data['binding_affinity'] = torch.tensor(
                    interaction.get('kd', interaction.get('affinity', 0)), 
                    dtype=torch.float32
                )
                data['is_interacting'] = torch.tensor(interaction.get('interacting', 1.0))
                
                return data
            else:
                return super().__getitem__(idx)


# =============================================================================
# Enzyme Datasets
# =============================================================================

if TORCH_AVAILABLE:
    
    class EnzymeDataset(ProteinDataset):
        """Dataset for enzyme structures with kinetic data."""
        
        def __init__(
            self,
            data_path: str,
            kinetics_file: Optional[str] = None,
            substrates_file: Optional[str] = None,
            transform: Optional[Callable] = None,
        ):
            super().__init__(data_path, transform)
            
            self.kinetics = {}
            self.substrates = {}
            
            if kinetics_file:
                self._load_kinetics(kinetics_file)
            if substrates_file:
                self._load_substrates(substrates_file)
        
        def _load_kinetics(self, path: str):
            """Load kinetic parameters."""
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.kinetics[item['id']] = {
                        'kcat': item.get('kcat'),
                        'km': item.get('km'),
                        'kcat_km': item.get('kcat_km'),
                        'substrate': item.get('substrate'),
                        'conditions': item.get('conditions', {}),
                    }
        
        def _load_substrates(self, path: str):
            """Load substrate library."""
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.substrates[item['id']] = item
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            data = super().__getitem__(idx)
            sample_id = os.path.basename(self.samples[idx]).split('.')[0]
            
            if sample_id in self.kinetics:
                kin = self.kinetics[sample_id]
                if kin['kcat'] is not None:
                    data['log_kcat'] = torch.tensor(np.log(kin['kcat']), dtype=torch.float32)
                if kin['km'] is not None:
                    data['log_km'] = torch.tensor(np.log(kin['km']), dtype=torch.float32)
            
            return data


# =============================================================================
# Function Prediction Datasets
# =============================================================================

if TORCH_AVAILABLE:
    
    class FunctionDataset(ProteinDataset):
        """Dataset for protein function prediction (GO, EC)."""
        
        def __init__(
            self,
            data_path: str,
            annotations_file: str,
            ontology: str = "go_mf",  # go_mf, go_bp, go_cc, ec
            num_classes: int = 1000,
            transform: Optional[Callable] = None,
        ):
            super().__init__(data_path, transform)
            
            self.ontology = ontology
            self.num_classes = num_classes
            self.annotations = {}
            self.term_to_idx = {}
            
            self._load_annotations(annotations_file)
        
        def _load_annotations(self, path: str):
            """Load function annotations."""
            with open(path, 'r') as f:
                data = json.load(f)
                
                # Build term vocabulary
                all_terms = set()
                for item in data:
                    terms = item.get(self.ontology, item.get('terms', []))
                    all_terms.update(terms)
                
                self.term_to_idx = {term: i for i, term in enumerate(sorted(all_terms)[:self.num_classes])}
                
                # Store annotations
                for item in data:
                    sample_id = item['id']
                    terms = item.get(self.ontology, item.get('terms', []))
                    self.annotations[sample_id] = [
                        self.term_to_idx[t] for t in terms if t in self.term_to_idx
                    ]
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            data = super().__getitem__(idx)
            sample_id = os.path.basename(self.samples[idx]).split('.')[0]
            
            # Create multi-hot label vector
            labels = torch.zeros(self.num_classes)
            if sample_id in self.annotations:
                for term_idx in self.annotations[sample_id]:
                    labels[term_idx] = 1
            
            data['labels'] = labels
            
            return data


# =============================================================================
# Epitope Datasets
# =============================================================================

if TORCH_AVAILABLE:
    
    class EpitopeDataset(ProteinDataset):
        """Dataset for epitope prediction."""
        
        def __init__(
            self,
            data_path: str,
            epitopes_file: Optional[str] = None,
            epitope_type: str = "bcell",  # bcell, tcell
            transform: Optional[Callable] = None,
        ):
            super().__init__(data_path, transform)
            
            self.epitope_type = epitope_type
            self.epitopes = {}
            
            if epitopes_file:
                self._load_epitopes(epitopes_file)
        
        def _load_epitopes(self, path: str):
            """Load epitope annotations."""
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.epitopes[item['id']] = {
                        'epitope_residues': item.get('epitope_residues', []),
                        'epitope_sequences': item.get('epitope_sequences', []),
                        'mhc_allele': item.get('mhc_allele'),
                        'immunogenicity': item.get('immunogenicity'),
                    }
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            data = super().__getitem__(idx)
            sample_id = os.path.basename(self.samples[idx]).split('.')[0]
            
            if sample_id in self.epitopes:
                epi = self.epitopes[sample_id]
                
                # Create epitope mask
                if 'num_res' in data:
                    num_res = data['num_res']
                else:
                    num_res = 500  # Default
                
                epitope_mask = torch.zeros(num_res)
                for pos in epi.get('epitope_residues', []):
                    if pos < num_res:
                        epitope_mask[pos] = 1
                
                data['epitope_mask'] = epitope_mask
            
            return data


# =============================================================================
# Dataset Factory
# =============================================================================

DATASET_REGISTRY = {
    # Drug Discovery
    "affinity": AffinityDataset if TORCH_AVAILABLE else None,
    "virtual_screening": VirtualScreeningDataset if TORCH_AVAILABLE else None,
    
    # Protein Engineering
    "stability": StabilityDataset if TORCH_AVAILABLE else None,
    "mutation": MutationDataset if TORCH_AVAILABLE else None,
    
    # Antibody
    "antibody": AntibodyDataset if TORCH_AVAILABLE else None,
    
    # PPI
    "ppi": PPIDataset if TORCH_AVAILABLE else None,
    
    # Enzyme
    "enzyme": EnzymeDataset if TORCH_AVAILABLE else None,
    
    # Function
    "function": FunctionDataset if TORCH_AVAILABLE else None,
    
    # Epitope
    "epitope": EpitopeDataset if TORCH_AVAILABLE else None,
}


def get_dataset(task: str, **kwargs):
    """Get dataset for a specific task."""
    if task not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown task: {task}. Available: {available}")
    
    dataset_cls = DATASET_REGISTRY[task]
    if dataset_cls is None:
        raise ImportError("PyTorch is required for datasets")
    
    return dataset_cls(**kwargs)
