# Fine-tuning Framework for Protein Structure Prediction Models

from .configs import FineTuningConfig, ModelConfig, TrainingConfig

try:
    from .modules import LoRAModule, AdapterModule, PromptTuning
except ImportError:
    LoRAModule = None  # type: ignore[misc, assignment]
    AdapterModule = None  # type: ignore[misc, assignment]
    PromptTuning = None  # type: ignore[misc, assignment]

try:
    from .heads import (
        AffinityHead,
        PropertyHead,
        ContactHead,
        AntibodyAffinityHead,
        DevelopabilityHead,
        HumannessHead,
        PPIBindingHead,
        PPIInterfaceHead,
        PPIHotspotHead,
        EnzymeActivityHead,
        EnzymeSpecificityHead,
        EnzymeEvolutionHead,
        GOPredictionHead,
        ECNumberHead,
        LocalizationHead,
        BcellEpitopeHead,
        TcellEpitopeHead,
        ImmunogenicityHead,
    )
except ImportError:
    AffinityHead = None  # type: ignore[misc, assignment]
    PropertyHead = None  # type: ignore[misc, assignment]
    ContactHead = None  # type: ignore[misc, assignment]
    AntibodyAffinityHead = None  # type: ignore[misc, assignment]
    DevelopabilityHead = None  # type: ignore[misc, assignment]
    HumannessHead = None  # type: ignore[misc, assignment]
    PPIBindingHead = None  # type: ignore[misc, assignment]
    PPIInterfaceHead = None  # type: ignore[misc, assignment]
    PPIHotspotHead = None  # type: ignore[misc, assignment]
    EnzymeActivityHead = None  # type: ignore[misc, assignment]
    EnzymeSpecificityHead = None  # type: ignore[misc, assignment]
    EnzymeEvolutionHead = None  # type: ignore[misc, assignment]
    GOPredictionHead = None  # type: ignore[misc, assignment]
    ECNumberHead = None  # type: ignore[misc, assignment]
    LocalizationHead = None  # type: ignore[misc, assignment]
    BcellEpitopeHead = None  # type: ignore[misc, assignment]
    TcellEpitopeHead = None  # type: ignore[misc, assignment]
    ImmunogenicityHead = None  # type: ignore[misc, assignment]

try:
    from .trainers import Trainer, DistributedTrainer
except ImportError:
    Trainer = None  # type: ignore[misc, assignment]
    DistributedTrainer = None  # type: ignore[misc, assignment]

try:
    from .utils import load_pretrained, save_checkpoint, evaluate_model
except ImportError:
    load_pretrained = None  # type: ignore[misc, assignment]
    save_checkpoint = None  # type: ignore[misc, assignment]
    evaluate_model = None  # type: ignore[misc, assignment]

try:
    from .registry import (
        TaskRegistry,
        TaskCategory,
        TaskInfo,
        create_finetuning_pipeline,
        get_task,
        list_tasks,
    )
except ImportError:
    TaskRegistry = None  # type: ignore[misc, assignment]
    TaskCategory = None  # type: ignore[misc, assignment]
    TaskInfo = None  # type: ignore[misc, assignment]
    create_finetuning_pipeline = None  # type: ignore[misc, assignment]
    get_task = None  # type: ignore[misc, assignment]
    list_tasks = None  # type: ignore[misc, assignment]

__version__ = "0.2.0"
__all__ = [
    "FineTuningConfig",
    "ModelConfig",
    "TrainingConfig",
    "LoRAModule",
    "AdapterModule",
    "PromptTuning",
    "AffinityHead",
    "PropertyHead",
    "ContactHead",
    "AntibodyAffinityHead",
    "DevelopabilityHead",
    "HumannessHead",
    "PPIBindingHead",
    "PPIInterfaceHead",
    "PPIHotspotHead",
    "EnzymeActivityHead",
    "EnzymeSpecificityHead",
    "EnzymeEvolutionHead",
    "GOPredictionHead",
    "ECNumberHead",
    "LocalizationHead",
    "BcellEpitopeHead",
    "TcellEpitopeHead",
    "ImmunogenicityHead",
    "Trainer",
    "DistributedTrainer",
    "load_pretrained",
    "save_checkpoint",
    "evaluate_model",
    "TaskRegistry",
    "TaskCategory",
    "TaskInfo",
    "create_finetuning_pipeline",
    "get_task",
    "list_tasks",
]
