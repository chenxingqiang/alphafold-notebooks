"""Task-specific configuration for fine-tuning.

Comprehensive task support inspired by ProteinBase.com business logic:
- Drug Discovery: Binding affinity, virtual screening, ADMET
- Protein Engineering: Stability, solubility, expression optimization
- Antibody Design: CDR optimization, humanization, developability
- Enzyme Engineering: Activity, specificity, thermostability
- Protein-Protein Interaction: Binding, interface, hot spots
- Function Prediction: GO terms, EC numbers, localization
- Safety Assessment: Immunogenicity, aggregation, toxicity
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal, Tuple
from enum import Enum


# =============================================================================
# Enumerations for Task Types
# =============================================================================

class PropertyType(str, Enum):
    """Types of protein properties to predict."""
    # Thermodynamic Properties
    STABILITY = "stability"              # ΔG of folding
    THERMAL_STABILITY = "thermal_stability"  # Tm
    
    # Expression & Production
    SOLUBILITY = "solubility"            # Protein solubility
    EXPRESSION = "expression"            # Expression level
    AGGREGATION = "aggregation"          # Aggregation propensity
    
    # Pharmacokinetics
    HALF_LIFE = "half_life"              # Serum half-life
    CLEARANCE = "clearance"              # Renal/hepatic clearance
    
    # Safety
    IMMUNOGENICITY = "immunogenicity"    # Immunogenicity score
    TOXICITY = "toxicity"                # Toxicity prediction
    
    # Binding
    BINDING_CONSTANT = "kd"              # Dissociation constant
    IC50 = "ic50"                        # Half-maximal inhibitory concentration
    EC50 = "ec50"                        # Half-maximal effective concentration
    
    # Custom
    CUSTOM = "custom"


class AntibodyRegion(str, Enum):
    """Antibody regions."""
    CDR_H1 = "cdr_h1"
    CDR_H2 = "cdr_h2"
    CDR_H3 = "cdr_h3"
    CDR_L1 = "cdr_l1"
    CDR_L2 = "cdr_l2"
    CDR_L3 = "cdr_l3"
    FRAMEWORK = "framework"
    FULL = "full"


class EnzymeProperty(str, Enum):
    """Enzyme properties to predict."""
    KCAT = "kcat"                        # Turnover number
    KM = "km"                            # Michaelis constant
    KCAT_KM = "kcat_km"                  # Catalytic efficiency
    ACTIVITY = "activity"                # Relative activity
    SPECIFICITY = "specificity"          # Substrate specificity
    ENANTIOSELECTIVITY = "enantioselectivity"  # Stereo selectivity
    THERMOSTABILITY = "thermostability"  # Temperature optimum


class FunctionOntology(str, Enum):
    """Function annotation ontologies."""
    GO_MF = "go_molecular_function"      # GO Molecular Function
    GO_BP = "go_biological_process"      # GO Biological Process
    GO_CC = "go_cellular_component"      # GO Cellular Component
    EC_NUMBER = "ec_number"              # Enzyme Commission number
    PFAM = "pfam"                        # Pfam domains
    INTERPRO = "interpro"                # InterPro annotations


class LocalizationType(str, Enum):
    """Subcellular localization types."""
    NUCLEUS = "nucleus"
    CYTOPLASM = "cytoplasm"
    MEMBRANE = "membrane"
    MITOCHONDRIA = "mitochondria"
    ER = "endoplasmic_reticulum"
    GOLGI = "golgi"
    LYSOSOME = "lysosome"
    EXTRACELLULAR = "extracellular"
    PEROXISOME = "peroxisome"


# =============================================================================
# Base Task Configuration
# =============================================================================

@dataclass
class TaskConfig:
    """Base configuration for fine-tuning tasks."""
    
    # General settings
    task_name: str = "base_task"
    task_description: str = ""
    task_category: str = "general"  # drug_discovery, engineering, antibody, etc.
    
    # Output configuration
    num_outputs: int = 1
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "regression"
    num_classes: Optional[int] = None
    class_labels: Optional[List[str]] = None
    
    # Loss function
    loss_type: Literal["mse", "mae", "huber", "cross_entropy", "focal", "bce", "custom"] = "mse"
    loss_weight: float = 1.0
    
    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: ["rmse", "mae", "r2", "pearson"])
    
    # Normalization
    normalize_targets: bool = True
    target_mean: Optional[float] = None
    target_std: Optional[float] = None
    
    # Uncertainty
    predict_uncertainty: bool = False
    use_ensemble: bool = False
    num_ensemble_members: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v.value if isinstance(v, Enum) else v 
                for k, v in self.__dict__.items()}


# =============================================================================
# Drug Discovery Tasks
# =============================================================================

@dataclass
class BindingAffinityConfig(TaskConfig):
    """Configuration for binding affinity prediction (Drug-Target Interaction)."""
    
    task_name: str = "binding_affinity"
    task_description: str = "Predict protein-ligand binding affinity"
    task_category: str = "drug_discovery"
    
    # Affinity settings
    affinity_type: Literal["pkd", "pki", "pic50", "delta_g", "kd_nm"] = "pic50"
    predict_relative: bool = False  # ΔΔG prediction
    
    # Pocket features
    use_pocket_features: bool = True
    pocket_radius: float = 10.0
    pocket_detection: Literal["fpocket", "p2rank", "manual"] = "fpocket"
    
    # Ligand features
    use_ligand_features: bool = True
    ligand_representation: Literal["smiles", "fingerprint", "graph", "3d", "hybrid"] = "graph"
    fingerprint_type: Literal["morgan", "rdkit", "maccs"] = "morgan"
    fingerprint_bits: int = 2048
    
    # Interaction features
    use_interaction_fingerprint: bool = True
    interaction_types: List[str] = field(default_factory=lambda: [
        "hydrophobic", "hbond_donor", "hbond_acceptor", "ionic", "aromatic"
    ])


@dataclass
class VirtualScreeningConfig(TaskConfig):
    """Configuration for virtual screening / hit identification."""
    
    task_name: str = "virtual_screening"
    task_description: str = "Rank compounds by predicted binding"
    task_category: str = "drug_discovery"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "classification"
    num_classes: int = 2  # Active / Inactive
    
    # Screening settings
    activity_threshold: float = 6.0  # pIC50 threshold for active
    use_decoys: bool = True
    decoy_ratio: float = 50.0  # Decoys per active
    
    # Ranking metrics
    metrics: List[str] = field(default_factory=lambda: [
        "auroc", "auprc", "ef1", "ef5", "ef10", "bedroc"
    ])


@dataclass
class ADMETConfig(TaskConfig):
    """Configuration for ADMET property prediction."""
    
    task_name: str = "admet"
    task_description: str = "Predict ADMET properties"
    task_category: str = "drug_discovery"
    
    # ADMET endpoints
    endpoints: List[str] = field(default_factory=lambda: [
        "absorption", "distribution", "metabolism", "excretion", "toxicity"
    ])
    
    # Specific properties
    absorption_properties: List[str] = field(default_factory=lambda: [
        "caco2", "pgp_inhibitor", "pgp_substrate", "hia"
    ])
    metabolism_properties: List[str] = field(default_factory=lambda: [
        "cyp2d6_inhibitor", "cyp3a4_inhibitor", "cyp2c9_substrate"
    ])
    toxicity_properties: List[str] = field(default_factory=lambda: [
        "herg", "ames", "hepatotoxicity", "skin_sensitization"
    ])


# =============================================================================
# Protein Engineering Tasks
# =============================================================================

@dataclass
class StabilityConfig(TaskConfig):
    """Configuration for protein stability prediction."""
    
    task_name: str = "stability"
    task_description: str = "Predict protein thermodynamic stability"
    task_category: str = "engineering"
    
    # Stability type
    stability_type: Literal["ddg", "tm", "t50", "half_life"] = "ddg"
    
    # Reference state
    use_wild_type: bool = True
    reference_conditions: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 298.15,  # K
        "ph": 7.0,
        "ionic_strength": 0.15  # M
    })
    
    # Features
    use_evolutionary_features: bool = True
    use_structure_features: bool = True
    use_energy_features: bool = True  # Rosetta energies


@dataclass
class SolubilityConfig(TaskConfig):
    """Configuration for protein solubility prediction."""
    
    task_name: str = "solubility"
    task_description: str = "Predict protein solubility and expression"
    task_category: str = "engineering"
    
    # Solubility type
    solubility_type: Literal["binary", "continuous", "class"] = "continuous"
    
    # Expression system
    expression_system: Literal["ecoli", "yeast", "mammalian", "insect"] = "ecoli"
    
    # Features
    use_sequence_features: bool = True
    use_composition_features: bool = True
    use_disorder_features: bool = True
    use_aggregation_features: bool = True


@dataclass
class MutationEffectConfig(TaskConfig):
    """Configuration for mutation effect prediction."""
    
    task_name: str = "mutation_effect"
    task_description: str = "Predict effect of mutations on protein properties"
    task_category: str = "engineering"
    
    # Effect type
    effect_type: Literal["ddg", "fitness", "pathogenicity", "activity"] = "ddg"
    
    # Mutation handling
    max_mutations: int = 10
    predict_epistasis: bool = False  # Non-additive effects
    
    # Features
    use_wild_type_structure: bool = True
    use_evolutionary_features: bool = True
    use_pssm: bool = True
    use_blosum: bool = True


# =============================================================================
# Antibody Design Tasks
# =============================================================================

@dataclass
class AntibodyOptimizationConfig(TaskConfig):
    """Configuration for antibody optimization."""
    
    task_name: str = "antibody_optimization"
    task_description: str = "Optimize antibody properties"
    task_category: str = "antibody"
    
    # Target property
    optimization_target: Literal["affinity", "specificity", "stability", "developability"] = "affinity"
    
    # Regions
    optimize_regions: List[AntibodyRegion] = field(default_factory=lambda: [
        AntibodyRegion.CDR_H3, AntibodyRegion.CDR_L3
    ])
    
    # Constraints
    preserve_framework: bool = True
    max_mutations_per_cdr: int = 5
    
    # Developability filters
    check_aggregation: bool = True
    check_immunogenicity: bool = True
    check_expression: bool = True


@dataclass
class HumanizationConfig(TaskConfig):
    """Configuration for antibody humanization."""
    
    task_name: str = "humanization"
    task_description: str = "Humanize non-human antibodies"
    task_category: str = "antibody"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "sequence"
    
    # Humanization method
    method: Literal["cdr_grafting", "resurfacing", "superhumanization"] = "cdr_grafting"
    
    # Human germline
    target_germline: Optional[str] = None  # Auto-select if None
    
    # Constraints
    preserve_binding: bool = True
    minimize_immunogenicity: bool = True
    
    # Scoring
    humanness_threshold: float = 0.8


@dataclass
class AntibodyDevelopabilityConfig(TaskConfig):
    """Configuration for antibody developability assessment."""
    
    task_name: str = "developability"
    task_description: str = "Predict antibody developability properties"
    task_category: str = "antibody"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "multi_label"
    
    # Properties to assess
    properties: List[str] = field(default_factory=lambda: [
        "aggregation_propensity",
        "viscosity",
        "self_interaction",
        "polyreactivity",
        "clearance",
        "immunogenicity",
        "expression",
        "stability"
    ])
    
    # Thresholds for flags
    aggregation_threshold: float = 0.5
    viscosity_threshold: float = 20.0  # cP at 150 mg/mL


# =============================================================================
# Enzyme Engineering Tasks
# =============================================================================

@dataclass
class EnzymeActivityConfig(TaskConfig):
    """Configuration for enzyme activity prediction."""
    
    task_name: str = "enzyme_activity"
    task_description: str = "Predict enzyme kinetic parameters"
    task_category: str = "enzyme"
    
    # Activity type
    activity_type: EnzymeProperty = EnzymeProperty.KCAT_KM
    
    # Substrate
    use_substrate_features: bool = True
    substrate_representation: Literal["smiles", "fingerprint", "graph"] = "graph"
    
    # Conditions
    include_conditions: bool = True
    conditions: List[str] = field(default_factory=lambda: [
        "temperature", "ph", "cofactor"
    ])


@dataclass
class EnzymeSpecificityConfig(TaskConfig):
    """Configuration for enzyme substrate specificity prediction."""
    
    task_name: str = "enzyme_specificity"
    task_description: str = "Predict enzyme substrate specificity"
    task_category: str = "enzyme"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "multi_label"
    
    # Specificity type
    specificity_type: Literal["substrate", "product", "regioselectivity", "stereoselectivity"] = "substrate"
    
    # Substrate library
    substrate_library: Optional[str] = None  # Path to substrate library
    
    # Molecular docking
    use_docking_features: bool = True


@dataclass  
class EnzymeEngineeringConfig(TaskConfig):
    """Configuration for directed enzyme evolution."""
    
    task_name: str = "enzyme_engineering"
    task_description: str = "Guide directed evolution experiments"
    task_category: str = "enzyme"
    
    # Optimization target
    optimization_target: EnzymeProperty = EnzymeProperty.ACTIVITY
    
    # Evolution strategy
    mutation_strategy: Literal["random", "focused", "recombination"] = "focused"
    
    # Active site
    focus_active_site: bool = True
    active_site_radius: float = 8.0
    
    # Screening
    predict_expressibility: bool = True
    predict_stability: bool = True


# =============================================================================
# Protein-Protein Interaction Tasks
# =============================================================================

@dataclass
class PPIBindingConfig(TaskConfig):
    """Configuration for protein-protein interaction prediction."""
    
    task_name: str = "ppi_binding"
    task_description: str = "Predict protein-protein binding affinity"
    task_category: str = "ppi"
    
    # Binding type
    binding_type: Literal["kd", "ic50", "binary"] = "kd"
    
    # Interface
    use_interface_features: bool = True
    interface_distance: float = 10.0
    
    # Complex features
    use_complex_structure: bool = True
    use_evolutionary_covariance: bool = True


@dataclass
class PPIInterfaceConfig(TaskConfig):
    """Configuration for PPI interface prediction."""
    
    task_name: str = "ppi_interface"
    task_description: str = "Predict protein-protein interface residues"
    task_category: str = "ppi"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "classification"
    num_classes: int = 2  # Interface / Non-interface
    
    # Interface definition
    interface_threshold: float = 5.0  # Å
    
    # Features
    use_surface_features: bool = True
    use_conservation: bool = True
    use_coevolution: bool = True


@dataclass
class PPIHotspotConfig(TaskConfig):
    """Configuration for PPI hot spot prediction."""
    
    task_name: str = "ppi_hotspot"
    task_description: str = "Predict binding hot spots"
    task_category: str = "ppi"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "classification"
    num_classes: int = 2  # Hot spot / Non-hot spot
    
    # Hot spot definition
    ddg_threshold: float = 2.0  # kcal/mol for hot spot
    
    # Features
    use_alanine_scanning: bool = True
    use_energetics: bool = True


# =============================================================================
# Function Prediction Tasks
# =============================================================================

@dataclass
class FunctionPredictionConfig(TaskConfig):
    """Configuration for protein function prediction."""
    
    task_name: str = "function_prediction"
    task_description: str = "Predict protein function annotations"
    task_category: str = "function"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "multi_label"
    
    # Ontology
    ontology: FunctionOntology = FunctionOntology.GO_MF
    
    # GO settings
    go_namespace: Literal["all", "mf", "bp", "cc"] = "all"
    min_go_depth: int = 3
    max_go_depth: int = 10
    
    # Features
    use_sequence_features: bool = True
    use_structure_features: bool = True
    use_domain_features: bool = True
    use_homology_features: bool = True


@dataclass
class LocalizationConfig(TaskConfig):
    """Configuration for subcellular localization prediction."""
    
    task_name: str = "localization"
    task_description: str = "Predict subcellular localization"
    task_category: str = "function"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "multi_label"
    
    # Localizations
    localizations: List[LocalizationType] = field(default_factory=lambda: [
        LocalizationType.NUCLEUS,
        LocalizationType.CYTOPLASM,
        LocalizationType.MEMBRANE,
        LocalizationType.MITOCHONDRIA,
        LocalizationType.EXTRACELLULAR,
    ])
    
    # Features
    use_signal_peptide: bool = True
    use_transmembrane: bool = True
    use_sorting_signals: bool = True


# =============================================================================
# Epitope & Immunology Tasks
# =============================================================================

@dataclass
class BcellEpitopeConfig(TaskConfig):
    """Configuration for B-cell epitope prediction."""
    
    task_name: str = "bcell_epitope"
    task_description: str = "Predict B-cell epitopes"
    task_category: str = "immunology"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "classification"
    num_classes: int = 2
    
    # Epitope type
    epitope_type: Literal["linear", "conformational", "both"] = "conformational"
    
    # Features
    use_accessibility: bool = True
    use_flexibility: bool = True
    use_protrusion: bool = True
    use_antigenicity: bool = True


@dataclass
class TcellEpitopeConfig(TaskConfig):
    """Configuration for T-cell epitope prediction."""
    
    task_name: str = "tcell_epitope"
    task_description: str = "Predict T-cell epitopes and MHC binding"
    task_category: str = "immunology"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "regression"
    
    # MHC type
    mhc_class: Literal["I", "II", "both"] = "I"
    
    # Alleles
    alleles: List[str] = field(default_factory=lambda: [
        "HLA-A*02:01", "HLA-A*01:01", "HLA-B*07:02"
    ])
    
    # Prediction type
    predict_binding: bool = True
    predict_immunogenicity: bool = True
    predict_processing: bool = True


@dataclass
class ImmunogenicityConfig(TaskConfig):
    """Configuration for immunogenicity prediction."""
    
    task_name: str = "immunogenicity"
    task_description: str = "Predict therapeutic protein immunogenicity"
    task_category: str = "immunology"
    
    # Immunogenicity components
    predict_tcell_response: bool = True
    predict_bcell_response: bool = True
    predict_ada: bool = True  # Anti-drug antibodies
    
    # Features
    use_t_cell_epitopes: bool = True
    use_sequence_humanness: bool = True
    use_aggregation_prone: bool = True


# =============================================================================
# Structure Quality Tasks
# =============================================================================

@dataclass
class StructureQualityConfig(TaskConfig):
    """Configuration for structure quality assessment."""
    
    task_name: str = "structure_quality"
    task_description: str = "Assess predicted structure quality"
    task_category: str = "quality"
    
    # Quality metrics
    metrics_to_predict: List[str] = field(default_factory=lambda: [
        "plddt", "pae", "ptm", "lddt"
    ])
    
    # Per-residue vs global
    per_residue: bool = True
    global_score: bool = True


@dataclass
class DisorderPredictionConfig(TaskConfig):
    """Configuration for intrinsic disorder prediction."""
    
    task_name: str = "disorder"
    task_description: str = "Predict intrinsically disordered regions"
    task_category: str = "quality"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "regression"
    
    # Disorder type
    disorder_type: Literal["binary", "score", "region"] = "score"
    
    # Region detection
    min_region_length: int = 10
    disorder_threshold: float = 0.5


# =============================================================================
# Contact & Distance Tasks
# =============================================================================

@dataclass
class ContactPredictionConfig(TaskConfig):
    """Configuration for contact prediction."""
    
    task_name: str = "contact_prediction"
    task_description: str = "Predict residue-residue contacts"
    task_category: str = "structure"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "classification"
    num_classes: int = 2
    
    # Contact definition
    contact_threshold: float = 8.0
    min_sequence_separation: int = 6
    
    # Output
    output_symmetric: bool = True
    predict_distance: bool = False


@dataclass
class DistanceMapConfig(TaskConfig):
    """Configuration for distance map prediction."""
    
    task_name: str = "distance_map"
    task_description: str = "Predict inter-residue distances"
    task_category: str = "structure"
    
    output_type: Literal["regression", "classification", "multi_label", "sequence"] = "regression"
    
    # Distance bins (if classification)
    use_bins: bool = True
    distance_bins: List[float] = field(default_factory=lambda: [
        4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0
    ])
    
    # Atom type
    atom_type: Literal["ca", "cb", "min"] = "cb"


# =============================================================================
# Preset Task Configurations (ProteinBase-style)
# =============================================================================

TASK_PRESETS = {
    # Drug Discovery
    "binding_affinity": BindingAffinityConfig(),
    "virtual_screening": VirtualScreeningConfig(),
    "admet": ADMETConfig(),
    
    # Protein Engineering
    "stability": StabilityConfig(),
    "solubility": SolubilityConfig(),
    "mutation_ddg": MutationEffectConfig(effect_type="ddg"),
    "mutation_fitness": MutationEffectConfig(effect_type="fitness"),
    
    # Antibody Design
    "antibody_affinity": AntibodyOptimizationConfig(optimization_target="affinity"),
    "antibody_developability": AntibodyDevelopabilityConfig(),
    "humanization": HumanizationConfig(),
    
    # Enzyme Engineering
    "enzyme_activity": EnzymeActivityConfig(),
    "enzyme_specificity": EnzymeSpecificityConfig(),
    "enzyme_evolution": EnzymeEngineeringConfig(),
    
    # Protein-Protein Interaction
    "ppi_binding": PPIBindingConfig(),
    "ppi_interface": PPIInterfaceConfig(),
    "ppi_hotspot": PPIHotspotConfig(),
    
    # Function Prediction
    "function_go": FunctionPredictionConfig(ontology=FunctionOntology.GO_MF),
    "function_ec": FunctionPredictionConfig(ontology=FunctionOntology.EC_NUMBER),
    "localization": LocalizationConfig(),
    
    # Immunology
    "bcell_epitope": BcellEpitopeConfig(),
    "tcell_epitope": TcellEpitopeConfig(),
    "immunogenicity": ImmunogenicityConfig(),
    
    # Structure Quality
    "structure_quality": StructureQualityConfig(),
    "disorder": DisorderPredictionConfig(),
    
    # Contact & Distance
    "contact": ContactPredictionConfig(),
    "distance_map": DistanceMapConfig(),
}


def get_task_config(task_name: str) -> TaskConfig:
    """Get preset task configuration."""
    if task_name not in TASK_PRESETS:
        available = ", ".join(sorted(TASK_PRESETS.keys()))
        raise ValueError(f"Unknown task: {task_name}. Available: {available}")
    return TASK_PRESETS[task_name]


def list_tasks_by_category() -> Dict[str, List[str]]:
    """List all available tasks grouped by category."""
    categories = {}
    for name, config in TASK_PRESETS.items():
        category = config.task_category
        if category not in categories:
            categories[category] = []
        categories[category].append(name)
    return categories
