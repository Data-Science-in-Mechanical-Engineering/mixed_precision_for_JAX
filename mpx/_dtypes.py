import jax.numpy as jnp

import sys
import types

HALF_PRECISION_DATATYPE = jnp.float16  # Default half precision datatype

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

def set_half_precision_datatype(datatype):
    """
    Set the half precision datatype for the module.
    
    Args:
        datatype: The datatype to set as half precision (e.g., jnp.float16).
    """
    global HALF_PRECISION_DATATYPE
    if isinstance(datatype, str):
        if datatype == 'float16':
            datatype = jnp.float16
        elif datatype == 'bfloat16':
            datatype = jnp.bfloat16
        else:
            raise ValueError(f"Unsupported datatype: {datatype}. Use 'float16' or 'bfloat16'.")
    elif datatype in (jnp.float16, jnp.bfloat16):
        HALF_PRECISION_DATATYPE = datatype
    else:
        raise TypeError("Datatype must be a string or in (jnp.float16, jnp.bfloat16).")

def half_precision_datatype():
    return HALF_PRECISION_DATATYPE
