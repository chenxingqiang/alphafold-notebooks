"""AlphaFold 3 weight loading, validation, and parameter-efficient fine-tuning."""

from .finetuner import (
    AF3FineTuneConfig,
    AlphaFold3FineTuner,
    ParameterSummary,
    WeightsComplianceError,
)
from .lora import AF3LoRA, LoRAConfig, count_lora_params
from .param_groups import (
    DEFAULT_LORA_TARGET_PATTERNS,
    ParamGroup,
    classify,
    group_param_counts,
    is_linear_weight,
    is_transposed_weight,
    matrix_layout,
    select_lora_targets,
    stack_dims,
)
from .record_io import (
    MissingDependencyError,
    RecordError,
    encode_record,
    flatten,
    read_params,
    read_records,
    select_model_files,
    unflatten,
    write_params,
)
from .schema import ParamSpec, load_schema, summarize, validate_params
from .weights import AF3_WEIGHTS_URL, check_weights, download_weights, load_weights

__all__ = [
    "AF3FineTuneConfig",
    "AF3LoRA",
    "AF3_WEIGHTS_URL",
    "AlphaFold3FineTuner",
    "DEFAULT_LORA_TARGET_PATTERNS",
    "LoRAConfig",
    "MissingDependencyError",
    "ParamGroup",
    "ParamSpec",
    "ParameterSummary",
    "RecordError",
    "WeightsComplianceError",
    "classify",
    "check_weights",
    "count_lora_params",
    "download_weights",
    "encode_record",
    "flatten",
    "group_param_counts",
    "is_linear_weight",
    "is_transposed_weight",
    "load_schema",
    "load_weights",
    "matrix_layout",
    "read_params",
    "read_records",
    "select_lora_targets",
    "select_model_files",
    "stack_dims",
    "summarize",
    "unflatten",
    "validate_params",
    "write_params",
]
