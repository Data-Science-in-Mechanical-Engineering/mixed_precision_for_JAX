"""
Mixed Precision for JAX - A library for mixed precision training in JAX
"""

__version__ = "0.1.7"

from ._cast import (
    cast_tree,
    cast_to_float32,
    cast_to_float16,
    cast_to_bfloat16,
    cast_to_full_precision,
    cast_to_half_precision,
    force_full_precision,
    cast_function,
)
from ._dtypes import half_precision_datatype, set_half_precision_datatype, HALF_PRECISION_DATATYPE, set_forward_backward_precision, forward_datatype, backward_datatype  # , FLOAT16_MAX, BFLOAT16_MAX
from ._loss_scaling import DynamicLossScaling, all_finite, scaled
from ._grad_tools import select_tree, filter_grad, filter_value_and_grad, optimizer_update, calculate_scaled_grad


__all__ = [
    # Cast functions
    'cast_tree',
    'cast_to_float32',
    'cast_to_float16',
    'cast_to_bfloat16',
    'cast_to_full_precision',
    'cast_to_half_precision',
    'force_full_precision',
    'cast_function',
]
