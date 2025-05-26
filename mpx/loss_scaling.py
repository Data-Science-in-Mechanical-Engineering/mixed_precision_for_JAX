"""
MIT License

Copyright (c) 2025 Alexander Gräfe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Tools for mixer precision training. Methods and general code architecture are from jmp https://github.com/google-deepmind/jmp. This can be seen as a port and extension of JMP tot equinox.
"""

"""Tools for automatic loss scaling (https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)."""


import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray, Bool
import optax 


def all_finite(tree: PyTree) -> Array:
    """
    Checks if all elements in a PyTree of arrays are finite.

    This function traverses the input PyTree, extracts all array leaves, and 
    verifies whether all elements in these arrays are finite (i.e., not NaN or Inf).

    Args:
        tree (PyTree): A PyTree containing arrays to be checked for finiteness.

    Returns:
        Array: A scalar ndarray of type bool indicating whether all elements 
        in the input PyTree are finite. Returns True if all elements are finite, 
        otherwise False.
    """
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(True)
    else:
        leaves = map(jnp.isfinite, leaves)
        leaves = map(jnp.all, leaves)
        return jnp.stack(list(leaves)).all()


def scaled(func: callable, scaling: 'DynamicLossScaling', has_aux: bool = False) -> callable:
    def wrapper(*_args, **_kwargs):
        if has_aux:
            value, aux = func(*_args, **_kwargs)
            value = scaling.scale(value)
            return value, aux
        else:
            value = func(*_args, **_kwargs)
            value = scaling.scale(value)
            return value
    return wrapper


class DynamicLossScaling(eqx.Module):
    """
    Implements dynamic loss scaling for mixed precision training in JAX.
    The basic structure is taken from jmp.
    This class automatically adjusts the loss scaling factor during training to prevent
    numerical underflow/overflow when using reduced precision (e.g., float16). The scaling
    factor is increased periodically if gradients are finite, and decreased if non-finite
    gradients are detected, within specified bounds.
    Attributes:
        loss_scaling (jnp.ndarray): Current loss scaling factor.
        min_loss_scaling (jnp.ndarray): Minimum allowed loss scaling factor.
        counter (jnp.ndarray): Counter for tracking update periods.
        factor (int): Multiplicative factor for adjusting loss scaling.
        period (int): Number of steps between potential increases of loss scaling.
    Methods:
        scale(tree):
            Scales all leaves of a pytree by the current loss scaling factor.
        unscale(tree):
            Unscales all leaves of a pytree by the inverse of the current loss scaling factor,
            casting the result to float32.
        adjust(grads_finite: jnp.ndarray) -> 'DynamicLossScaling':
            Returns a new DynamicLossScaling instance with updated loss scaling and counter,
            depending on whether the gradients are finite.
    """
    loss_scaling: jnp.ndarray
    min_loss_scaling: jnp.ndarray
    counter: jnp.ndarray
    factor: int
    period: int

    def __init__(self, loss_scaling: jnp.ndarray, min_loss_scaling: jnp.ndarray, factor: int = 2, period: int = 2000, counter=None):
        assert loss_scaling.ndim == 0, "Expected scalar loss scaling"
        assert min_loss_scaling.ndim == 0, "Expected scalar minimum loss scaling"
        self.loss_scaling = loss_scaling
        self.min_loss_scaling = min_loss_scaling
        self.factor = factor
        self.period = period
        if counter is None:
            self.counter = jnp.zeros((), dtype=jnp.int32)
        else:
            self.counter = counter

    def scale(self, tree):
        return jax.tree_util.tree_map(lambda x: x * self.loss_scaling.astype(jnp.float16), tree)

    def unscale(self, tree):
        inv_loss_scaling = 1 / self.loss_scaling
        inv_loss_scaling = inv_loss_scaling.astype(jnp.float32)   # cast to float32, so the result is float32 (otherwise the whole scaling point would be senseless)
        return jax.tree_util.tree_map(lambda x: x * inv_loss_scaling, tree)
    
    def adjust(self, grads_finite: jnp.ndarray) -> 'DynamicLossScaling':
        """Returns the next state dependent on whether grads are finite."""
        assert grads_finite.ndim == 0, "Expected boolean scalar"

        first_finite = lambda a, b: jax.lax.select(jnp.isfinite(a).all(), a, b)
        loss_scaling = jax.lax.select(
            grads_finite,

            # When grads are finite increase loss scaling periodically.
            jax.lax.select(
                self.counter == (self.period - 1),
                first_finite(self.loss_scaling * self.factor,
                            self.loss_scaling),
                self.loss_scaling),

            # If grads are non finite reduce loss scaling.
            jnp.maximum(self.min_loss_scaling, self.loss_scaling / self.factor))
        
        # clip to maximum float16 value.
        loss_scaling = jnp.clip(loss_scaling, min=self.min_loss_scaling, max=(2 - 2**(-10)) * 2**15)

        counter = ((self.counter + 1) % self.period) * grads_finite

        return DynamicLossScaling(
            loss_scaling=loss_scaling,
            counter=counter,
            period=self.period,
            factor=self.factor,
            min_loss_scaling=self.min_loss_scaling)
