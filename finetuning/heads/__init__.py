# Task-Specific Prediction Heads

try:
    from .affinity_head import AffinityHead, AffinityHeadConfig
except ImportError:
    AffinityHead = None  # type: ignore[misc, assignment]
    AffinityHeadConfig = None  # type: ignore[misc, assignment]

try:
    from .property_head import PropertyHead, PropertyHeadConfig
    from .contact_head import ContactHead, ContactHeadConfig
except ImportError:
    PropertyHead = None  # type: ignore[misc, assignment]
    PropertyHeadConfig = None  # type: ignore[misc, assignment]
    ContactHead = None  # type: ignore[misc, assignment]
    ContactHeadConfig = None  # type: ignore[misc, assignment]

try:
    from .antibody_head import (
        AntibodyHeadConfig,
        AntibodyAffinityHead,
        DevelopabilityHead,
        HumannessHead,
    )
except ImportError:
    AntibodyHeadConfig = None  # type: ignore[misc, assignment]
    AntibodyAffinityHead = None  # type: ignore[misc, assignment]
    DevelopabilityHead = None  # type: ignore[misc, assignment]
    HumannessHead = None  # type: ignore[misc, assignment]

try:
    from .ppi_head import (
        PPIHeadConfig,
        PPIBindingHead,
        PPIInterfaceHead,
        PPIHotspotHead,
    )
except ImportError:
    PPIHeadConfig = None  # type: ignore[misc, assignment]
    PPIBindingHead = None  # type: ignore[misc, assignment]
    PPIInterfaceHead = None  # type: ignore[misc, assignment]
    PPIHotspotHead = None  # type: ignore[misc, assignment]

try:
    from .enzyme_head import (
        EnzymeHeadConfig,
        EnzymeActivityHead,
        EnzymeSpecificityHead,
        EnzymeEvolutionHead,
    )
except ImportError:
    EnzymeHeadConfig = None  # type: ignore[misc, assignment]
    EnzymeActivityHead = None  # type: ignore[misc, assignment]
    EnzymeSpecificityHead = None  # type: ignore[misc, assignment]
    EnzymeEvolutionHead = None  # type: ignore[misc, assignment]

try:
    from .function_head import (
        FunctionHeadConfig,
        GOPredictionHead,
        ECNumberHead,
        LocalizationHead,
    )
except ImportError:
    FunctionHeadConfig = None  # type: ignore[misc, assignment]
    GOPredictionHead = None  # type: ignore[misc, assignment]
    ECNumberHead = None  # type: ignore[misc, assignment]
    LocalizationHead = None  # type: ignore[misc, assignment]

try:
    from .epitope_head import (
        EpitopeHeadConfig,
        BcellEpitopeHead,
        TcellEpitopeHead,
        ImmunogenicityHead,
    )
except ImportError:
    EpitopeHeadConfig = None  # type: ignore[misc, assignment]
    BcellEpitopeHead = None  # type: ignore[misc, assignment]
    TcellEpitopeHead = None  # type: ignore[misc, assignment]
    ImmunogenicityHead = None  # type: ignore[misc, assignment]

__all__ = [
    "AffinityHead",
    "AffinityHeadConfig",
    "PropertyHead",
    "PropertyHeadConfig",
    "ContactHead",
    "ContactHeadConfig",
    "AntibodyHeadConfig",
    "AntibodyAffinityHead",
    "DevelopabilityHead",
    "HumannessHead",
    "PPIHeadConfig",
    "PPIBindingHead",
    "PPIInterfaceHead",
    "PPIHotspotHead",
    "EnzymeHeadConfig",
    "EnzymeActivityHead",
    "EnzymeSpecificityHead",
    "EnzymeEvolutionHead",
    "FunctionHeadConfig",
    "GOPredictionHead",
    "ECNumberHead",
    "LocalizationHead",
    "EpitopeHeadConfig",
    "BcellEpitopeHead",
    "TcellEpitopeHead",
    "ImmunogenicityHead",
]
