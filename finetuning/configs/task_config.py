"""Task-specific configuration for fine-tuning."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class PropertyType(str, Enum):
    """Types of protein properties to predict."""
    STABILITY = "stability"          # Thermodynamic stability (ΔG)
    SOLUBILITY = "solubility"        # Protein solubility
    EXPRESSION = "expression"        # Expression level
    AGGREGATION = "aggregation"      # Aggregation propensity
    IMMUNOGENICITY = "immunogenicity"  # Immunogenicity score
    HALF_LIFE = "half_life"          # Protein half-life
    BINDING_CONSTANT = "kd"          # Dissociation constant
    IC50 = "ic50"                    # Half-maximal inhibitory concentration
    CUSTOM = "custom"                # Custom property


@dataclass
class TaskConfig:
    """Configuration for specific fine-tuning tasks."""
    
    # General task settings
    task_name: str = "binding_affinity"
    """Name of the task."""
    
    task_description: str = ""
    """Description of the task."""
    
    # Output configuration
    num_outputs: int = 1
    """Number of output values (1 for regression, N for classification)."""
    
    output_type: Literal["regression", "classification", "multi_label"] = "regression"
    """Type of output."""
    
    # For classification tasks
    num_classes: Optional[int] = None
    """Number of classes for classification."""
    
    class_labels: Optional[List[str]] = None
    """Labels for each class."""
    
    # Loss function
    loss_type: Literal["mse", "mae", "huber", "cross_entropy", "focal", "custom"] = "mse"
    """Type of loss function."""
    
    loss_weight: float = 1.0
    """Weight for this task's loss."""
    
    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: ["rmse", "mae", "r2", "pearson"])
    """Metrics to compute during evaluation."""
    
    # Task-specific parameters
    normalize_targets: bool = True
    """Whether to normalize target values."""
    
    target_mean: Optional[float] = None
    """Mean for target normalization."""
    
    target_std: Optional[float] = None
    """Standard deviation for target normalization."""
    
    # Auxiliary outputs
    predict_uncertainty: bool = False
    """Whether to predict uncertainty (epistemic/aleatoric)."""
    
    use_ensemble: bool = False
    """Whether to use ensemble predictions."""
    
    num_ensemble_members: int = 5
    """Number of ensemble members."""


@dataclass
class BindingAffinityConfig(TaskConfig):
    """Configuration for binding affinity prediction."""
    
    task_name: str = "binding_affinity"
    task_description: str = "Predict protein-ligand binding affinity (pKd, pIC50)"
    
    # Affinity-specific settings
    affinity_type: Literal["pkd", "pki", "pic50", "delta_g"] = "pic50"
    """Type of affinity measurement."""
    
    use_pocket_features: bool = True
    """Whether to use binding pocket features."""
    
    pocket_radius: float = 10.0
    """Radius (Å) for defining binding pocket."""
    
    use_ligand_features: bool = True
    """Whether to use ligand molecular features."""
    
    ligand_representation: Literal["smiles", "fingerprint", "graph", "3d"] = "graph"
    """How to represent ligands."""
    
    # For relative binding affinity
    predict_relative: bool = False
    """Whether to predict relative binding affinities (ΔΔG)."""


@dataclass
class PropertyPredictionConfig(TaskConfig):
    """Configuration for protein property prediction."""
    
    task_name: str = "property_prediction"
    task_description: str = "Predict protein physicochemical or functional properties"
    
    # Property type
    property_type: PropertyType = PropertyType.STABILITY
    """Type of property to predict."""
    
    # Feature extraction
    use_sequence_features: bool = True
    """Whether to use sequence-level features."""
    
    use_structure_features: bool = True
    """Whether to use structure-level features."""
    
    use_residue_features: bool = True
    """Whether to use per-residue features."""
    
    pooling_strategy: Literal["mean", "max", "attention", "cls"] = "attention"
    """How to pool residue features to protein-level."""


@dataclass
class ContactPredictionConfig(TaskConfig):
    """Configuration for contact prediction."""
    
    task_name: str = "contact_prediction"
    task_description: str = "Predict residue-residue contacts"
    
    output_type: Literal["regression", "classification", "multi_label"] = "classification"
    num_classes: int = 2  # Contact / No contact
    
    # Contact definition
    contact_threshold: float = 8.0
    """Distance threshold (Å) for defining contacts."""
    
    min_sequence_separation: int = 6
    """Minimum sequence separation for contact pairs."""
    
    # Output format
    output_symmetric: bool = True
    """Whether to enforce symmetric contact matrix."""


@dataclass
class MutationEffectConfig(TaskConfig):
    """Configuration for mutation effect prediction."""
    
    task_name: str = "mutation_effect"
    task_description: str = "Predict the effect of mutations on protein properties"
    
    # Mutation-specific settings
    effect_type: Literal["ddg", "fitness", "pathogenicity"] = "ddg"
    """Type of mutation effect to predict."""
    
    use_wild_type_structure: bool = True
    """Whether to use wild-type structure as reference."""
    
    use_evolutionary_features: bool = True
    """Whether to use evolutionary conservation features."""
    
    predict_multiple_mutations: bool = True
    """Whether to handle multiple mutations."""
    
    max_mutations: int = 10
    """Maximum number of mutations to handle."""


# Preset task configurations
TASK_PRESETS = {
    "binding_affinity": BindingAffinityConfig(),
    "stability": PropertyPredictionConfig(property_type=PropertyType.STABILITY),
    "solubility": PropertyPredictionConfig(property_type=PropertyType.SOLUBILITY),
    "contact": ContactPredictionConfig(),
    "mutation_ddg": MutationEffectConfig(effect_type="ddg"),
    "mutation_fitness": MutationEffectConfig(effect_type="fitness"),
}
