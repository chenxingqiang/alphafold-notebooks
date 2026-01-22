"""
Task Registry and Factory Pattern for AlphaFold Codec Fine-tuning Framework.

This module provides a centralized registry for all supported tasks,
making it easy to create models, heads, and datasets for any task type.

Usage:
    from finetuning.registry import TaskRegistry, create_finetuning_pipeline
    
    # List all tasks
    TaskRegistry.list_all_tasks()
    
    # Get task info
    info = TaskRegistry.get_task_info("binding_affinity")
    
    # Create complete pipeline
    pipeline = create_finetuning_pipeline(
        task="binding_affinity",
        base_model=model,
        strategy="lora",
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any, Callable
from enum import Enum

# Import task configs
from .configs.task_config import (
    TASK_PRESETS,
    BindingAffinityConfig,
    VirtualScreeningConfig,
    ADMETConfig,
    StabilityConfig,
    SolubilityConfig,
    MutationEffectConfig,
    AntibodyOptimizationConfig,
    HumanizationConfig,
    AntibodyDevelopabilityConfig,
    EnzymeActivityConfig,
    EnzymeSpecificityConfig,
    EnzymeEngineeringConfig,
    PPIBindingConfig,
    PPIInterfaceConfig,
    PPIHotspotConfig,
    FunctionPredictionConfig,
    LocalizationConfig,
    BcellEpitopeConfig,
    TcellEpitopeConfig,
    ImmunogenicityConfig,
    StructureQualityConfig,
    DisorderPredictionConfig,
    ContactPredictionConfig,
    DistanceMapConfig,
)


class TaskCategory(Enum):
    """Categories of fine-tuning tasks."""
    DRUG_DISCOVERY = "drug_discovery"
    PROTEIN_ENGINEERING = "protein_engineering"
    ANTIBODY_DESIGN = "antibody_design"
    ENZYME_ENGINEERING = "enzyme_engineering"
    PROTEIN_PROTEIN_INTERACTION = "protein_protein_interaction"
    FUNCTION_PREDICTION = "function_prediction"
    IMMUNOLOGY = "immunology"
    STRUCTURE_QUALITY = "structure_quality"


@dataclass
class TaskInfo:
    """Complete information about a task."""
    name: str
    category: TaskCategory
    description: str
    config_class: Type
    head_class: Optional[str] = None  # String reference to avoid circular imports
    dataset_class: Optional[str] = None
    outputs: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    example_datasets: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    min_samples: int = 100
    recommended_rank: int = 8
    recommended_lr: float = 5e-5


# Registry of all tasks
TASK_REGISTRY: Dict[str, TaskInfo] = {
    # Drug Discovery
    "binding_affinity": TaskInfo(
        name="binding_affinity",
        category=TaskCategory.DRUG_DISCOVERY,
        description="Predict protein-ligand binding affinity (pKd, pIC50, ΔG)",
        config_class=BindingAffinityConfig,
        head_class="AffinityHead",
        dataset_class="AffinityDataset",
        outputs=["pKd", "pIC50", "delta_G", "Ki"],
        metrics=["rmse", "mae", "pearson", "r2", "spearman"],
        example_datasets=["PDBbind", "BindingDB", "ChEMBL"],
        difficulty="medium",
        min_samples=500,
        recommended_rank=8,
        recommended_lr=5e-5,
    ),
    "virtual_screening": TaskInfo(
        name="virtual_screening",
        category=TaskCategory.DRUG_DISCOVERY,
        description="Rank compounds by binding probability for high-throughput screening",
        config_class=VirtualScreeningConfig,
        head_class="AffinityHead",
        dataset_class="VirtualScreeningDataset",
        outputs=["hit_probability", "rank", "enrichment"],
        metrics=["auroc", "auprc", "ef_1", "ef_5", "bedroc"],
        example_datasets=["DUD-E", "LIT-PCBA", "MUV"],
        difficulty="medium",
        min_samples=1000,
        recommended_rank=8,
        recommended_lr=1e-4,
    ),
    "admet": TaskInfo(
        name="admet",
        category=TaskCategory.DRUG_DISCOVERY,
        description="Predict ADMET properties (absorption, distribution, metabolism, excretion, toxicity)",
        config_class=ADMETConfig,
        head_class="PropertyHead",
        dataset_class="AffinityDataset",
        outputs=["solubility", "permeability", "metabolism", "clearance", "toxicity"],
        metrics=["auroc", "rmse", "accuracy"],
        example_datasets=["ADMET-AI", "TDC"],
        difficulty="hard",
        min_samples=1000,
        recommended_rank=16,
        recommended_lr=1e-4,
    ),
    
    # Protein Engineering
    "stability": TaskInfo(
        name="stability",
        category=TaskCategory.PROTEIN_ENGINEERING,
        description="Predict protein stability changes upon mutation (ΔΔG)",
        config_class=StabilityConfig,
        head_class="PropertyHead",
        dataset_class="StabilityDataset",
        outputs=["ddG", "Tm_shift", "aggregation_propensity"],
        metrics=["rmse", "mae", "pearson", "spearman"],
        example_datasets=["ProTherm", "FireProtDB", "Megascale"],
        difficulty="medium",
        min_samples=300,
        recommended_rank=8,
        recommended_lr=5e-5,
    ),
    "solubility": TaskInfo(
        name="solubility",
        category=TaskCategory.PROTEIN_ENGINEERING,
        description="Predict protein solubility and expression",
        config_class=SolubilityConfig,
        head_class="PropertyHead",
        dataset_class="StabilityDataset",
        outputs=["solubility_score", "expression_level"],
        metrics=["rmse", "pearson", "accuracy"],
        example_datasets=["eSol", "PSI:Biology"],
        difficulty="easy",
        min_samples=200,
        recommended_rank=4,
        recommended_lr=1e-4,
    ),
    "mutation_effects": TaskInfo(
        name="mutation_effects",
        category=TaskCategory.PROTEIN_ENGINEERING,
        description="Predict functional effects of mutations (fitness, pathogenicity)",
        config_class=MutationEffectConfig,
        head_class="PropertyHead",
        dataset_class="MutationDataset",
        outputs=["fitness", "pathogenicity", "functional_score"],
        metrics=["spearman", "auroc", "auprc"],
        example_datasets=["DMS datasets", "ClinVar", "gnomAD"],
        difficulty="medium",
        min_samples=500,
        recommended_rank=8,
        recommended_lr=5e-5,
    ),
    
    # Antibody Design
    "affinity_maturation": TaskInfo(
        name="affinity_maturation",
        category=TaskCategory.ANTIBODY_DESIGN,
        description="Optimize antibody-antigen binding affinity",
        config_class=AntibodyOptimizationConfig,
        head_class="AntibodyAffinityHead",
        dataset_class="AntibodyDataset",
        outputs=["ddG", "fold_improvement", "binding_score"],
        metrics=["spearman", "top_k_accuracy", "ndcg"],
        example_datasets=["SAbDab", "SKEMPI"],
        difficulty="hard",
        min_samples=200,
        recommended_rank=16,
        recommended_lr=5e-5,
    ),
    "humanization": TaskInfo(
        name="humanization",
        category=TaskCategory.ANTIBODY_DESIGN,
        description="Assess and improve antibody humanness for therapeutic development",
        config_class=HumanizationConfig,
        head_class="HumannessHead",
        dataset_class="AntibodyDataset",
        outputs=["humanness_score", "immunogenicity_risk"],
        metrics=["auroc", "accuracy", "pearson"],
        example_datasets=["OAS", "Observed Antibody Space"],
        difficulty="medium",
        min_samples=500,
        recommended_rank=8,
        recommended_lr=1e-4,
    ),
    "developability": TaskInfo(
        name="developability",
        category=TaskCategory.ANTIBODY_DESIGN,
        description="Predict antibody developability (aggregation, viscosity, expression)",
        config_class=AntibodyDevelopabilityConfig,
        head_class="DevelopabilityHead",
        dataset_class="AntibodyDataset",
        outputs=["aggregation_score", "viscosity", "expression_yield"],
        metrics=["auroc", "rmse", "accuracy"],
        example_datasets=["TAP", "Internal assays"],
        difficulty="hard",
        min_samples=300,
        recommended_rank=16,
        recommended_lr=5e-5,
    ),
    
    # Enzyme Engineering
    "enzyme_activity": TaskInfo(
        name="enzyme_activity",
        category=TaskCategory.ENZYME_ENGINEERING,
        description="Predict enzyme kinetic parameters (kcat, Km, kcat/Km)",
        config_class=EnzymeActivityConfig,
        head_class="EnzymeActivityHead",
        dataset_class="EnzymeDataset",
        outputs=["log_kcat", "log_Km", "log_kcat_over_Km"],
        metrics=["rmse", "pearson", "spearman"],
        example_datasets=["BRENDA", "SABIO-RK"],
        difficulty="hard",
        min_samples=200,
        recommended_rank=16,
        recommended_lr=5e-5,
    ),
    "enzyme_specificity": TaskInfo(
        name="enzyme_specificity",
        category=TaskCategory.ENZYME_ENGINEERING,
        description="Predict substrate specificity and selectivity",
        config_class=EnzymeSpecificityConfig,
        head_class="EnzymeSpecificityHead",
        dataset_class="EnzymeDataset",
        outputs=["substrate_profile", "selectivity_ratio"],
        metrics=["spearman", "auroc", "top_k"],
        example_datasets=["MEROPS", "CAZy"],
        difficulty="hard",
        min_samples=300,
        recommended_rank=16,
        recommended_lr=5e-5,
    ),
    "directed_evolution": TaskInfo(
        name="directed_evolution",
        category=TaskCategory.ENZYME_ENGINEERING,
        description="Guide directed evolution by predicting fitness landscapes",
        config_class=EnzymeEngineeringConfig,
        head_class="EnzymeEvolutionHead",
        dataset_class="EnzymeDataset",
        outputs=["fitness", "epistasis", "hot_spots"],
        metrics=["spearman", "ndcg", "hit_rate"],
        example_datasets=["DMS datasets", "FLIP"],
        difficulty="hard",
        min_samples=500,
        recommended_rank=16,
        recommended_lr=5e-5,
    ),
    
    # Protein-Protein Interactions
    "ppi_binding": TaskInfo(
        name="ppi_binding",
        category=TaskCategory.PROTEIN_PROTEIN_INTERACTION,
        description="Predict protein-protein binding affinity",
        config_class=PPIBindingConfig,
        head_class="PPIBindingHead",
        dataset_class="PPIDataset",
        outputs=["Kd", "ddG", "binding_energy"],
        metrics=["rmse", "pearson", "spearman"],
        example_datasets=["SKEMPI", "AB-Bind"],
        difficulty="medium",
        min_samples=300,
        recommended_rank=8,
        recommended_lr=5e-5,
    ),
    "ppi_interface": TaskInfo(
        name="ppi_interface",
        category=TaskCategory.PROTEIN_PROTEIN_INTERACTION,
        description="Predict protein-protein interface residues",
        config_class=PPIInterfaceConfig,
        head_class="PPIInterfaceHead",
        dataset_class="PPIDataset",
        outputs=["interface_residues", "buried_surface_area"],
        metrics=["auroc", "auprc", "precision", "recall"],
        example_datasets=["DOCKGROUND", "PIFACE"],
        difficulty="medium",
        min_samples=200,
        recommended_rank=8,
        recommended_lr=1e-4,
    ),
    "ppi_hotspot": TaskInfo(
        name="ppi_hotspot",
        category=TaskCategory.PROTEIN_PROTEIN_INTERACTION,
        description="Identify binding hot spots at protein-protein interfaces",
        config_class=PPIHotspotConfig,
        head_class="PPIHotspotHead",
        dataset_class="PPIDataset",
        outputs=["hotspot_residues", "ddG_per_residue", "druggability"],
        metrics=["auroc", "auprc", "recall_at_k"],
        example_datasets=["ASEdb", "BID"],
        difficulty="hard",
        min_samples=100,
        recommended_rank=16,
        recommended_lr=5e-5,
    ),
    
    # Function Prediction
    "go_prediction": TaskInfo(
        name="go_prediction",
        category=TaskCategory.FUNCTION_PREDICTION,
        description="Predict Gene Ontology terms (MF, BP, CC)",
        config_class=FunctionPredictionConfig,
        head_class="GOPredictionHead",
        dataset_class="FunctionDataset",
        outputs=["MF_terms", "BP_terms", "CC_terms"],
        metrics=["auroc", "auprc", "f1_max", "precision", "recall"],
        example_datasets=["UniProt-GO", "CAFA"],
        difficulty="medium",
        min_samples=1000,
        recommended_rank=8,
        recommended_lr=1e-4,
    ),
    "ec_number": TaskInfo(
        name="ec_number",
        category=TaskCategory.FUNCTION_PREDICTION,
        description="Predict EC enzyme classification numbers",
        config_class=FunctionPredictionConfig,
        head_class="ECNumberHead",
        dataset_class="FunctionDataset",
        outputs=["ec_class", "ec_subclass", "ec_full"],
        metrics=["accuracy", "macro_f1", "hierarchical_f1"],
        example_datasets=["UniProt", "BRENDA"],
        difficulty="medium",
        min_samples=500,
        recommended_rank=8,
        recommended_lr=1e-4,
    ),
    "localization": TaskInfo(
        name="localization",
        category=TaskCategory.FUNCTION_PREDICTION,
        description="Predict subcellular localization",
        config_class=LocalizationConfig,
        head_class="LocalizationHead",
        dataset_class="FunctionDataset",
        outputs=["localization", "signal_peptide", "transmembrane"],
        metrics=["accuracy", "macro_f1", "mcc"],
        example_datasets=["DeepLoc", "TargetP"],
        difficulty="easy",
        min_samples=500,
        recommended_rank=4,
        recommended_lr=1e-4,
    ),
    
    # Immunology
    "bcell_epitope": TaskInfo(
        name="bcell_epitope",
        category=TaskCategory.IMMUNOLOGY,
        description="Predict B-cell epitopes (linear and conformational)",
        config_class=BcellEpitopeConfig,
        head_class="BcellEpitopeHead",
        dataset_class="EpitopeDataset",
        outputs=["epitope_probability", "epitope_type"],
        metrics=["auroc", "auprc", "precision", "recall"],
        example_datasets=["IEDB", "BepiPred", "DiscoTope"],
        difficulty="medium",
        min_samples=500,
        recommended_rank=8,
        recommended_lr=1e-4,
    ),
    "tcell_epitope": TaskInfo(
        name="tcell_epitope",
        category=TaskCategory.IMMUNOLOGY,
        description="Predict T-cell epitopes and MHC binding",
        config_class=TcellEpitopeConfig,
        head_class="TcellEpitopeHead",
        dataset_class="EpitopeDataset",
        outputs=["mhc_binding", "immunogenicity", "presentation"],
        metrics=["auroc", "auprc", "ppv"],
        example_datasets=["IEDB", "NetMHCpan"],
        difficulty="hard",
        min_samples=1000,
        recommended_rank=16,
        recommended_lr=5e-5,
    ),
    "immunogenicity": TaskInfo(
        name="immunogenicity",
        category=TaskCategory.IMMUNOLOGY,
        description="Predict therapeutic immunogenicity risk",
        config_class=ImmunogenicityConfig,
        head_class="ImmunogenicityHead",
        dataset_class="EpitopeDataset",
        outputs=["immunogenicity_score", "ada_risk"],
        metrics=["auroc", "auprc", "accuracy"],
        example_datasets=["IEDB", "Clinical data"],
        difficulty="hard",
        min_samples=200,
        recommended_rank=16,
        recommended_lr=5e-5,
    ),
    
    # Structure Quality
    "structure_quality": TaskInfo(
        name="structure_quality",
        category=TaskCategory.STRUCTURE_QUALITY,
        description="Predict structure quality metrics (pLDDT, pAE, pTM)",
        config_class=StructureQualityConfig,
        head_class="PropertyHead",
        dataset_class="ProteinDataset",
        outputs=["plddt", "pae", "ptm", "lddt"],
        metrics=["pearson", "spearman", "rmse"],
        example_datasets=["AlphaFold DB", "PDB"],
        difficulty="easy",
        min_samples=100,
        recommended_rank=4,
        recommended_lr=1e-4,
    ),
    "disorder_prediction": TaskInfo(
        name="disorder_prediction",
        category=TaskCategory.STRUCTURE_QUALITY,
        description="Predict intrinsically disordered regions",
        config_class=DisorderPredictionConfig,
        head_class="PropertyHead",
        dataset_class="ProteinDataset",
        outputs=["disorder_probability"],
        metrics=["auroc", "auprc", "mcc"],
        example_datasets=["DisProt", "MobiDB"],
        difficulty="easy",
        min_samples=500,
        recommended_rank=4,
        recommended_lr=1e-4,
    ),
    "contact_prediction": TaskInfo(
        name="contact_prediction",
        category=TaskCategory.STRUCTURE_QUALITY,
        description="Predict residue-residue contacts",
        config_class=ContactPredictionConfig,
        head_class="ContactHead",
        dataset_class="ProteinDataset",
        outputs=["contact_map", "contact_probability"],
        metrics=["precision_L", "precision_L5", "auroc"],
        example_datasets=["PDB", "CASP"],
        difficulty="medium",
        min_samples=500,
        recommended_rank=8,
        recommended_lr=5e-5,
    ),
    "distance_prediction": TaskInfo(
        name="distance_prediction",
        category=TaskCategory.STRUCTURE_QUALITY,
        description="Predict inter-residue distance maps",
        config_class=DistanceMapConfig,
        head_class="ContactHead",
        dataset_class="ProteinDataset",
        outputs=["distance_map", "distance_bins"],
        metrics=["mae", "rmse", "lddt"],
        example_datasets=["PDB", "AlphaFold DB"],
        difficulty="medium",
        min_samples=500,
        recommended_rank=8,
        recommended_lr=5e-5,
    ),
}


class TaskRegistry:
    """Central registry for all fine-tuning tasks."""
    
    @staticmethod
    def list_all_tasks() -> List[str]:
        """List all registered task names."""
        return list(TASK_REGISTRY.keys())
    
    @staticmethod
    def list_tasks_by_category(category: TaskCategory) -> List[str]:
        """List tasks in a specific category."""
        return [
            name for name, info in TASK_REGISTRY.items()
            if info.category == category
        ]
    
    @staticmethod
    def get_task_info(task_name: str) -> TaskInfo:
        """Get complete information about a task."""
        if task_name not in TASK_REGISTRY:
            available = ", ".join(TASK_REGISTRY.keys())
            raise ValueError(f"Unknown task: {task_name}. Available: {available}")
        return TASK_REGISTRY[task_name]
    
    @staticmethod
    def get_recommended_config(task_name: str) -> dict:
        """Get recommended hyperparameters for a task."""
        info = TaskRegistry.get_task_info(task_name)
        return {
            "lora_rank": info.recommended_rank,
            "learning_rate": info.recommended_lr,
            "min_samples": info.min_samples,
        }
    
    @staticmethod
    def search_tasks(query: str) -> List[str]:
        """Search tasks by keyword in name or description."""
        query_lower = query.lower()
        matches = []
        for name, info in TASK_REGISTRY.items():
            if query_lower in name.lower() or query_lower in info.description.lower():
                matches.append(name)
        return matches
    
    @staticmethod
    def get_tasks_for_output(output_type: str) -> List[str]:
        """Find tasks that predict a specific output type."""
        output_lower = output_type.lower()
        matches = []
        for name, info in TASK_REGISTRY.items():
            for output in info.outputs:
                if output_lower in output.lower():
                    matches.append(name)
                    break
        return matches
    
    @staticmethod
    def print_task_summary():
        """Print a summary of all tasks by category."""
        print("=" * 70)
        print("AlphaFold Codec Fine-tuning Task Registry")
        print("=" * 70)
        
        for category in TaskCategory:
            tasks = TaskRegistry.list_tasks_by_category(category)
            if tasks:
                print(f"\n{category.value.replace('_', ' ').title()}")
                print("-" * 50)
                for task_name in tasks:
                    info = TASK_REGISTRY[task_name]
                    print(f"  {task_name}: {info.description[:50]}...")


@dataclass
class FineTuningPipeline:
    """Complete fine-tuning pipeline for a task."""
    task_name: str
    task_info: TaskInfo
    config: Any
    head: Optional[Any] = None
    dataset: Optional[Any] = None
    model: Optional[Any] = None
    trainer: Optional[Any] = None


def create_finetuning_pipeline(
    task: str,
    base_model: Any = None,
    strategy: str = "lora",
    **kwargs
) -> FineTuningPipeline:
    """Factory function to create a complete fine-tuning pipeline.
    
    Args:
        task: Task name from registry
        base_model: Pre-trained model to fine-tune
        strategy: Fine-tuning strategy (lora, adapter, full)
        **kwargs: Additional configuration options
        
    Returns:
        Configured FineTuningPipeline
    """
    # Get task info
    task_info = TaskRegistry.get_task_info(task)
    
    # Get recommended config and merge with kwargs
    recommended = TaskRegistry.get_recommended_config(task)
    config_dict = {**recommended, **kwargs}
    
    # Create task config
    if task in TASK_PRESETS:
        config = TASK_PRESETS[task]
    else:
        config = task_info.config_class(**config_dict)
    
    # Create pipeline
    pipeline = FineTuningPipeline(
        task_name=task,
        task_info=task_info,
        config=config,
    )
    
    # If base model provided, set up LoRA/adapter
    if base_model is not None:
        try:
            if strategy == "lora":
                from .modules.lora import LoRAModule
                pipeline.model = LoRAModule(
                    base_model,
                    rank=config_dict.get("lora_rank", 8),
                    alpha=config_dict.get("lora_alpha", 16.0),
                )
            else:
                pipeline.model = base_model
        except ImportError:
            pipeline.model = base_model
    
    return pipeline


# Convenience function for quick access
def get_task(task_name: str) -> TaskInfo:
    """Quick access to task information."""
    return TaskRegistry.get_task_info(task_name)


def list_tasks() -> List[str]:
    """Quick access to list all tasks."""
    return TaskRegistry.list_all_tasks()


if __name__ == "__main__":
    # Demo
    TaskRegistry.print_task_summary()
    
    print("\n\nExample: Getting task info")
    print("-" * 40)
    info = get_task("binding_affinity")
    print(f"Task: {info.name}")
    print(f"Category: {info.category.value}")
    print(f"Description: {info.description}")
    print(f"Outputs: {info.outputs}")
    print(f"Recommended LoRA rank: {info.recommended_rank}")
