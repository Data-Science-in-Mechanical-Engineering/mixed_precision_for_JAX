from . import utils
from . import layers
from . import optimizers

"""
Mixed Precision for JAX (mpfj)

This package provides utilities for mixed precision training in JAX.
"""

__version__ = "0.1.0"

from .dtypes import set_half_precision_datatype