"""Utils for model executor."""
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(weight, key, value)


def post_preprocessing(param: torch.Tensor, loaded_weight: torch.Tensor) -> torch.Tensor:
    """Post preprocessing a weight tensor.

    This method is used to preprocess a weight tensor to make it satisfy some
    constraints of a specific quantized kernel. This is because some quantization
    kernel may assume some weights be organized in some special layouts to achieve
    better performance.

    Args:
        param: The parameter
        weight: The loaded weight tensor.
    """
    preprocessor = getattr(param, "preprocessor", None)
    if preprocessor is not None:
        loaded_weight = preprocessor(loaded_weight)
    dumper = getattr(param, "dumper", None)
    if dumper is not None:
        dumper(loaded_weight)
    return loaded_weight
