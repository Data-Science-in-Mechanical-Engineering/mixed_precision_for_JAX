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
from ._dtypes import half_precision_datatype, set_half_precision_datatype, HALF_PRECISION_DATATYPE  # , FLOAT16_MAX, BFLOAT16_MAX
from ._loss_scaling import DynamicLossScaling, all_finite, scaled
from ._grad_tools import select_tree, filter_grad, filter_value_and_grad, optimizer_update, calculate_scaled_grad

import sys
import types

import jax.numpy as jnp
# We do to avoid that jax is directly called when importing this module.
# This is to ensure that the constants are lazily initialized.
class _MaxConstantsLazyInit(types.ModuleType):    
    @property
    def FLOAT16_MAX(self):
        return jnp.ones([], dtype=jnp.float32) * (2 - 2**(-10)) * 2**15
    
    @property
    def BFLOAT16_MAX(self):
        return jnp.array([((2**8 - 1) * 2**(120))], dtype=jnp.float32)[0]

sys.modules[__name__].__class__ = _MaxConstantsLazyInit

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
    'calculate_scaled_grad',
]
