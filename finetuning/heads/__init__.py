# Task-Specific Prediction Heads

from .affinity_head import AffinityHead, AffinityHeadConfig
from .property_head import PropertyHead, PropertyHeadConfig
from .contact_head import ContactHead, ContactHeadConfig

__all__ = [
    "AffinityHead",
    "AffinityHeadConfig",
    "PropertyHead",
    "PropertyHeadConfig",
    "ContactHead",
    "ContactHeadConfig",
]
