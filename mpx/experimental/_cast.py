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
from .._cast import cast_function


def max_val(dtype):
    return (jnp.finfo(dtype).max).astype(jnp.float32)

@partial(jax.custom_vjp, nondiff_argnames=('dimension_numbers', 'precision', 'preferred_element_type', 'out_sharding'))
def quantized_multiplication(a: ArrayLike, b: ArrayLike, dimension_numbers, precision, preferred_element_type, out_sharding):
    a_max = jnp.max(jnp.abs(a))
    b_max = jnp.max(jnp.abs(b))
    fwd_dtype = forward_datatype()
    max_dtype = max_val(fwd_dtype)
    scaling_a = max_dtype / (a_max + 1e-8)
    scaling_b = max_dtype / (b_max + 1e-8)

    a_q = (a * scaling_a).astype(fwd_dtype)
    b_q = (b * scaling_b).astype(fwd_dtype)

    result_q = jax.lax.dot_general_p.bind(a_q, b_q, dimension_numbers=dimension_numbers, precision=precision, preferred_element_type=preferred_element_type, out_sharding=out_sharding)
    result = (result_q.astype(backward_datatype())) / (scaling_a * scaling_b)
    return result


def quantized_multiplication_fwd(a: ArrayLike, b: ArrayLike, dimension_numbers, precision, preferred_element_type, out_sharding):
    a_max = jnp.max(jnp.abs(a))
    b_max = jnp.max(jnp.abs(b))
    fwd_dtype = forward_datatype()
    max_dtype = max_val(fwd_dtype)
    scaling_a = max_dtype / (a_max + 1e-8)
    scaling_b = max_dtype / (b_max + 1e-8)

    a_q = (a * scaling_a).astype(fwd_dtype)
    b_q = (b * scaling_b).astype(fwd_dtype)
    # we want to save the quantized versions for the backward pass to save memory
    return quantized_multiplication(a, b, dimension_numbers, precision, preferred_element_type, out_sharding), (a_q, b_q, scaling_a, scaling_b)

# f_bwd :: (c, CT b) -> CT a
def quantized_multiplication_bwd(dimension_numbers, precision, preferred_element_type, out_sharding, c, dy_dc):
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
    return quantized_multiplication(lhs, rhs, **params)


def cast_function_fwd_bwd(f: callable) -> callable:
    """
    Casts a function to use the specified forward and backward data types.
    Args:
        f (callable): The function to be cast.
    Returns:
        callable: A new function that uses the specified data types for forward and backward passes.
    """

    # cast inuts to bwd_dtype. This makes all non multiply operations to be in bwd_dtype
    f = cast_function(f, backward_datatype())

    f = quax.quaxify(f)

    return f