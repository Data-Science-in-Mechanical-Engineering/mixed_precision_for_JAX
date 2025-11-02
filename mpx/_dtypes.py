import jax.numpy as jnp
import logging

HALF_PRECISION_DATATYPE = jnp.float16  # Default half precision datatype

EXPERIMENTAL_ACTIVATED = False  # Flag for experimental features
FORWARD_PRECISION_DATATYPE = None  # Default forward precision datatype
BACKWARD_PRECISION_DATATYPE = None  # Default backward precision datatype

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


def set_forward_backward_precision(forward_datatype, backward_datatype):
    """
    Set the forward and backward precision datatypes for experimental features.

    Args:
        forward_datatype: The datatype to use for forward computations.
        backward_datatype: The datatype to use for backward computations.
    """
    global EXPERIMENTAL_ACTIVATED
    global FORWARD_PRECISION_DATATYPE
    global BACKWARD_PRECISION_DATATYPE
    logging.warning("Setting forward backward precision is an experimental feature and may lead to unexpected behavior.")
    EXPERIMENTAL_ACTIVATED = True
    FORWARD_PRECISION_DATATYPE = forward_datatype
    BACKWARD_PRECISION_DATATYPE = backward_datatype
    assert backward_datatype == jnp.float32, "Currently only float32 is supported as backward datatype."

def forward_datatype():
    assert EXPERIMENTAL_ACTIVATED, "Experimental features not activated. Call set_forward_backward_precision first."
    return FORWARD_PRECISION_DATATYPE

def backward_datatype():
    assert EXPERIMENTAL_ACTIVATED, "Experimental features not activated. Call set_forward_backward_precision first."
    return BACKWARD_PRECISION_DATATYPE
