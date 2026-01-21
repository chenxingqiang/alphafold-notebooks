# Task-Specific Prediction Heads
# Comprehensive coverage for protein analysis tasks (ProteinBase-style)

# Drug Discovery
from .affinity_head import AffinityHead, AffinityHeadConfig

# Protein Engineering
from .property_head import PropertyHead, PropertyHeadConfig
from .contact_head import ContactHead, ContactHeadConfig

# Antibody Design
from .antibody_head import (
    AntibodyHeadConfig,
    AntibodyAffinityHead,
    DevelopabilityHead,
    HumannessHead,
)

# Protein-Protein Interaction
from .ppi_head import (
    PPIHeadConfig,
    PPIBindingHead,
    PPIInterfaceHead,
    PPIHotspotHead,
)

# Enzyme Engineering
from .enzyme_head import (
    EnzymeHeadConfig,
    EnzymeActivityHead,
    EnzymeSpecificityHead,
    EnzymeEvolutionHead,
)

# Function Prediction
from .function_head import (
    FunctionHeadConfig,
    GOPredictionHead,
    ECNumberHead,
    LocalizationHead,
)

# Immunology / Epitope
from .epitope_head import (
    EpitopeHeadConfig,
    BcellEpitopeHead,
    TcellEpitopeHead,
    ImmunogenicityHead,
)

__all__ = [
    # Drug Discovery
    "AffinityHead",
    "AffinityHeadConfig",
    
    # Protein Engineering
    "PropertyHead",
    "PropertyHeadConfig",
    "ContactHead",
    "ContactHeadConfig",
    
    # Antibody
    "AntibodyHeadConfig",
    "AntibodyAffinityHead",
    "DevelopabilityHead",
    "HumannessHead",
    
    # PPI
    "PPIHeadConfig",
    "PPIBindingHead",
    "PPIInterfaceHead",
    "PPIHotspotHead",
    
    # Enzyme
    "EnzymeHeadConfig",
    "EnzymeActivityHead",
    "EnzymeSpecificityHead",
    "EnzymeEvolutionHead",
    
    # Function
    "FunctionHeadConfig",
    "GOPredictionHead",
    "ECNumberHead",
    "LocalizationHead",
    
    # Epitope / Immunology
    "EpitopeHeadConfig",
    "BcellEpitopeHead",
    "TcellEpitopeHead",
    "ImmunogenicityHead",
]
