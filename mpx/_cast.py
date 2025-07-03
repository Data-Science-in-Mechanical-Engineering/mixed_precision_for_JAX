"""
Functions for casting of Pytrees.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray 

from ._dtypes import half_precision_datatype

def cast_tree(tree: PyTree, dtype):
    """
    Casts all array elements in a PyTree to a specified data type.
    This function traverses a PyTree and applies a type casting operation to all array elements with dtype float (float16, bfloat16, float32), leaving  all other elements unchanged.
    Args:
        tree (PyTree): The input PyTree containing arrays and other objects.
        dtype (numpy.dtype or str): The target data type to cast the arrays to.
    Returns:
        PyTree: A new PyTree with all array elements cast to the specified data type.
    """
    
    def _cast(x):
        if eqx.is_array(x):
            if x.dtype == jnp.float16 or x.dtype == jnp.bfloat16 or x.dtype == jnp.float32:
                return x.astype(dtype)
            else:
                return x
        else:
            return x
    return jax.tree_util.tree_map(_cast, tree)


def cast_to_float32(x: PyTree) -> PyTree:
    """
    Cast the input PyTree to `float32` data type.

    This function takes a PyTree and casts all its elements to the `float32` data type.

    Args:
        x (PyTree): The input PyTree containing elements to be cast.

    Returns:
        PyTree: A new PyTree with all elements cast to `float32`.
    """
    """Cast to float32."""
    return cast_tree(x, jnp.float32)


def cast_to_float16(x: PyTree) -> PyTree:
    """
    Casts all elements of a PyTree to the float16 data type.

    Args:
        x (PyTree): A PyTree containing numerical data to be cast to float16.

    Returns:
        PyTree: A new PyTree with all numerical elements cast to float16.
    """
    return cast_tree(x, jnp.float16)


def cast_to_bfloat16(x: PyTree) -> PyTree:
    """
    Casts the input PyTree to the bfloat16 data type.

    Args:
        x (PyTree): A PyTree structure containing arrays or tensors to be cast.

    Returns:
        PyTree: A PyTree with all arrays or tensors cast to the bfloat16 data type.
    """
    return cast_tree(x, jnp.bfloat16)


def cast_to_full_precision(x: PyTree) -> PyTree:
    """
    Casts all elements of a PyTree to full precision (float32).

    Args:
        x (PyTree): The input PyTree containing elements to be cast.

    Returns:
        PyTree: A new PyTree with all elements cast to float32 precision.
    """
    """Cast to full precision (float32)."""
    return cast_tree(x, jnp.float32)

def cast_to_half_precision(x: PyTree) -> PyTree:
    """
    Cast the input PyTree to half precision.

    This function converts all elements in the input PyTree to the half-precision
    datatype (either `float16` or `bfloat16`), depending on the configuration set
    by `set_half_precision_datatype`.

    Args:
        x (PyTree): The input PyTree containing elements to be cast to half precision.

    Returns:
        PyTree: A new PyTree with all elements cast to the half-precision datatype.
    """
    """Cast to half precision (float16/bfloat16, depending on with what we called set_half_precision_datatype)."""
    return cast_tree(x, half_precision_datatype())


def cast_function(func, dtype, return_dtype=None):
    """
    Casts the function to the specified data type.
    """

    if return_dtype is None:
        return_dtype = dtype

    def wrapper(*args, **kwargs):
        args_cast = []
        for arg in args:
            args_cast.append(cast_tree(arg, dtype))
        args_cast = tuple(args_cast)

        kwargs_cast = {}
        for key, value in kwargs.items():
            kwargs_cast[key] = cast_tree(value, dtype)

        results = func(*args_cast, **kwargs_cast)

        if type(results) == tuple:
            results_converted = []
            for r in results:
                results_converted.append(cast_tree(r, return_dtype))
            return tuple(results_converted)
        elif eqx.is_array(results):
            return cast_tree(results, return_dtype)
        return results
    
    return wrapper


def force_full_precision(func, return_dtype=jnp.float16):
    """
    A decorator to enforce full precision (float32) for the inputs and outputs of a function.
    This decorator ensures that all array arguments passed to the decorated function are 
    converted to float32 precision before the function is executed. Additionally, it converts 
    the outputs of the function to the specified `return_dtype` if they are arrays.
    Args:
        func (callable): The function to be decorated.
        return_dtype (dtype): The desired data type for the function's output arrays.
    Returns:
        callable: The wrapped function with enforced input and output precision.
    Example:
        @force_full_precision
        def my_function(x, y):
            return x + y
        # All array inputs to `my_function` will be cast to float32, and the output
        # will be cast to the specified `return_dtype` if it is an array.
    """

    return cast_function(func, jnp.float32, return_dtype)
