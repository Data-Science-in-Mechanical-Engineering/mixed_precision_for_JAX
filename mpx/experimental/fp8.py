from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx

from collections.abc import Sequence
from jaxtyping import ArrayLike  

import quax

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
  # backward is performed in fp32 TODO allow to change it.
  a = a_q.astype(jnp.float32) / scaling_a
  b = b_q.astype(jnp.float32) / scaling_b
  dy_da = jax.lax.dot_general_p.bind(dy_dc, b.T, dimension_numbers=dimension_numbers, precision=precision, preferred_element_type=preferred_element_type, out_sharding=out_sharding)
  dy_db = jax.lax.dot_general_p.bind(a.T, dy_dc, dimension_numbers=dimension_numbers, precision=precision, preferred_element_type=preferred_element_type, out_sharding=out_sharding)

  return (dy_da, dy_db)

quantized_multiplication.defvjp(quantized_multiplication_fwd, quantized_multiplication_bwd)

@quax.register(jax.lax.dot_general_p)
def _(lhs: ArrayLike, rhs: ArrayLike, **params):
    return quantized_multiplication(lhs, rhs, jnp.float8_e4m3, **params)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    a = jax.random.normal(subkey, (4, 4)) * 0 + 1.2
    key, subkey = jax.random.split(key)
    b = jax.random.normal(subkey, (4, 4)) * 0 + 1.4

    a = jnp.array([[1, 2, 3, 4.0]])
    b = jnp.array([[5, 6, 7, 8.0]]).T

    def multiply(a, b):
        return a @ b
    
    multiply_grad = jax.grad(lambda a, b: jnp.sum(multiply(a, b)), argnums=(0,1))
    multiply_f8 = quax.quaxify(multiply)
    multiply_f8_grad = jax.grad(lambda a, b: jnp.sum(multiply_f8(a, b)), argnums=(0,1))

    result_normal = multiply(a, b)

    result_f8 = multiply_f8(a, b)

    result_grad_normal = multiply_grad(a, b)
    result_grad_f8 = multiply_f8_grad(a, b)

    print("Normal result:")
    print(result_normal)
    print("FP8 result:")
    print(result_f8)
    print("Normal grad:")
    print(result_grad_normal)
    print("FP8 grad:")
    print(result_grad_f8)

    traced = jax.jit(multiply_f8_grad).trace(a, b)

    print(traced.jaxpr)

    # print(multiply(a, b))
    
    # multiply_f8 = quax.quaxify(multiply)
    # # result = jax.jit(multiply_f8)(FP8Array(a), b)
    # traced = jax.jit(multiply_f8).trace(FP8Array(a), b)
    # print(traced.jaxpr)
    # print(result.array)