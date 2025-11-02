"""
Functions for casting of Pytrees.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

import quax

from functools import partial

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray, ArrayLike

from .._dtypes import forward_datatype, backward_datatype
from .._cast import cast_tree


def max_val(dtype):
    return (jnp.finfo(dtype).max).astype(jnp.float32)

@partial(jax.custom_vjp, nondiff_argnames=("dtype8", 'dimension_numbers', 'precision', 'preferred_element_type', 'out_sharding'))
def quantized_multiplication(a: ArrayLike, b: ArrayLike, dtype8, dimension_numbers, precision, preferred_element_type, out_sharding):
    a_max = jnp.max(jnp.abs(a))
    b_max = jnp.max(jnp.abs(b))
    max_dtype = max_val(dtype8)
    scaling_a = max_dtype / (a_max + 1e-8)
    scaling_b = max_dtype / (b_max + 1e-8)

    a_q = (a * scaling_a).astype(dtype8)
    b_q = (b * scaling_b).astype(dtype8)

    result_q = jax.lax.dot_general_p.bind(a_q, b_q, dimension_numbers=dimension_numbers, precision=precision, preferred_element_type=preferred_element_type, out_sharding=out_sharding)

    result = (result_q.astype(jnp.float32)) / (scaling_a * scaling_b)
    return result


def quantized_multiplication_fwd(a: ArrayLike, b: ArrayLike, dtype8, dimension_numbers, precision, preferred_element_type, out_sharding):
    a_max = jnp.max(jnp.abs(a))
    b_max = jnp.max(jnp.abs(b))
    max_dtype = max_val(dtype8)
    scaling_a = max_dtype / (a_max + 1e-8)
    scaling_b = max_dtype / (b_max + 1e-8)

    a_q = (a * scaling_a).astype(dtype8)
    b_q = (b * scaling_b).astype(dtype8)
    # we want to save the quantized versions for the backward pass to save memory
    return quantized_multiplication(a, b, dtype8, dimension_numbers, precision, preferred_element_type, out_sharding), (a_q, b_q, scaling_a, scaling_b)

# f_bwd :: (c, CT b) -> CT a
def quantized_multiplication_bwd(dtype8, dimension_numbers, precision, preferred_element_type, out_sharding, c, dy_dc):
  a_q, b_q, scaling_a, scaling_b = c
  backward_dtype = backward_datatype()
  # backward is performed in fp32 TODO allow to change it.
  a = a_q.astype(backward_dtype) / scaling_a
  b = b_q.astype(backward_dtype) / scaling_b
  dy_da = jax.lax.dot_general_p.bind(dy_dc, b.T, dimension_numbers=dimension_numbers, precision=precision, preferred_element_type=preferred_element_type, out_sharding=out_sharding)
  dy_db = jax.lax.dot_general_p.bind(a.T, dy_dc, dimension_numbers=dimension_numbers, precision=precision, preferred_element_type=preferred_element_type, out_sharding=out_sharding)

  return (dy_da, dy_db)

quantized_multiplication.defvjp(quantized_multiplication_fwd, quantized_multiplication_bwd)

@quax.register(jax.lax.dot_general_p)
def _(lhs: ArrayLike, rhs: ArrayLike, **params):
    return quantized_multiplication(lhs, rhs, jnp.float8_e4m3, **params)



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


