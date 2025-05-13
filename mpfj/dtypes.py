import jax.numpy as jnp

HALF_PRECISION_DATATYPE = jnp.float16

def set_half_precision_datatype(datatype):
    """
    Set the half precision datatype for the module.
    
    Args:
        datatype: The datatype to set as half precision (e.g., jnp.float16).
    """
    global HALF_PRECISION_DATATYPE
    HALF_PRECISION_DATATYPE = datatype

def half_precision_datatype():
    return HALF_PRECISION_DATATYPE
