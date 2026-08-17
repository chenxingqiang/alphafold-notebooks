# Fine-tuning Modules

try:
    from .lora import LoRALinear, apply_lora_to_model, LoRAModule
except ImportError:
    LoRALinear = None  # type: ignore[misc, assignment]
    LoRAModule = None  # type: ignore[misc, assignment]
    apply_lora_to_model = None  # type: ignore[misc, assignment]

try:
    from .adapter import AdapterLayer, AdapterModule
except ImportError:
    AdapterLayer = None  # type: ignore[misc, assignment]
    AdapterModule = None  # type: ignore[misc, assignment]

try:
    from .prompt_tuning import PromptTuning, SoftPrompt
except ImportError:
    PromptTuning = None  # type: ignore[misc, assignment]
    SoftPrompt = None  # type: ignore[misc, assignment]

__all__ = [
    "LoRAModule",
    "LoRALinear",
    "apply_lora_to_model",
    "AdapterModule",
    "AdapterLayer",
    "PromptTuning",
    "SoftPrompt",
]
