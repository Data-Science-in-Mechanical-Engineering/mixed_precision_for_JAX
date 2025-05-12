import jax.numpy as jnp

HALF_PRECISION_DATATYPE = jnp.float16

def set_half_precision_datatype(datatype):
    """
    Set the half precision datatype for the module.
    
    Args:
        datatype: The datatype to set as half precision (e.g., jnp.float16).
    """
    HALF_PRECISION_DATATYPE = datatype