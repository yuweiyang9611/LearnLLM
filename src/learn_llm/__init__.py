"""Small, readable building blocks used by the LearnLLM exercises.

The package intentionally favors explicit implementations over highly optimized
ones.  Each module is small enough to read beside the corresponding lesson.
"""

from .attention import scaled_dot_product_attention
from .lora import (
    LoRALinear,
    inject_lora,
    load_lora_adapter_state_dict,
    lora_adapter_state_dict,
    lora_parameter_names,
)
from .math_utils import cross_entropy, cross_entropy_from_logits, stable_softmax
from .model import TinyGPT, TinyGPTConfig
from .rag import Document, RetrievalResult, TfidfRetriever
from .tokenizer import CharTokenizer

__all__ = [
    "CharTokenizer",
    "Document",
    "LoRALinear",
    "RetrievalResult",
    "TfidfRetriever",
    "TinyGPT",
    "TinyGPTConfig",
    "cross_entropy",
    "cross_entropy_from_logits",
    "inject_lora",
    "load_lora_adapter_state_dict",
    "lora_adapter_state_dict",
    "lora_parameter_names",
    "scaled_dot_product_attention",
    "stable_softmax",
]
