"""
Mixed Precision for JAX - A library for mixed precision training in JAX

# build: python3 -m build --sdist
# upload: twine upload --repository pypi dist/* 
"""

__version__ = "0.1.5"

from .cast import (
    cast_tree,
    cast_to_float32,
    cast_to_float16,
    cast_to_bfloat16,
    cast_to_full_precision,
    cast_to_half_precision,
    force_full_precision,
)
from .dtypes import half_precision_datatype, set_half_precision_datatype, FLOAT16_MAX, BFLOAT16_MAX
from .loss_scaling import DynamicLossScaling, all_finite, scaled
from .grad_tools import select_tree, filter_grad, filter_value_and_grad, optimizer_update

__all__ = [
    # Cast functions
    'cast_tree',
    'cast_to_float32',
    'cast_to_float16',
    'cast_to_bfloat16',
    'cast_to_full_precision',
    'cast_to_half_precision',
    'force_full_precision',
    
    # Dtype functions
    'half_precision_datatype',
    'set_half_precision_datatype',
    
    # Loss scaling functions
    'DynamicLossScaling',
    'all_finite',
    'scaled',
    
    # Gradient tools
    'select_tree',
    'filter_grad',
    'filter_value_and_grad',
    'optimizer_update',
]
