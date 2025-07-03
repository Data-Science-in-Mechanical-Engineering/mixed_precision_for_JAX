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
    """
    Scales the output of a function using dynamic loss scaling.
    This decorator wraps a given function such that its output is scaled using the
    provided dynamic loss scaling object. If the wrapped function returns auxiliary
    data (indicated by has_aux=True), only the primary value is scaled; otherwise, the
    sole returned value is scaled.
    Parameters:
        func (callable): The original function whose output is to be scaled.
        scaling (DynamicLossScaling): An object providing a `scale` method for scaling
            the function's output.
        has_aux (bool, optional): Flag indicating whether the wrapped function returns
            a tuple (value, aux) where only the `value` should be scaled. Defaults to False.
    Returns:
        callable: A new function that wraps the original function's behavior by applying
        the dynamic loss scaling to its result.
    """

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
        """Scales each element in the input tree by the loss scaling factor.
        This method applies a multiplication operation to every leaf in the given pytree,
        using the loss scaling factor (converted to jnp.float16) stored in the instance.
        It returns a new pytree where each element has been scaled accordingly.

        Parameters:
            tree: A pytree (e.g., nested lists, tuples, dicts) containing numerical values
                  that represent the data to be scaled.
        Returns:
            A new pytree with each value multiplied by the loss scaling factor as a jnp.float16.
        """
        return jax.tree_util.tree_map(lambda x: x * self.loss_scaling.astype(jnp.float16), tree)

    def unscale(self, tree):
        """
        Unscales a pytree by multiplying each leaf element by the inverse of the loss scaling factor (in float32).
        Parameters:
            tree: A pytree (nested structure of arrays, lists, tuples, dicts, etc.) where each leaf is a numeric array.
                  These numerical values will be scaled by the computed inverse loss scaling factor.
        Returns:
            A new pytree with the same structure as the input, where each numeric leaf is multiplied by 1 / loss_scaling (as a float32).
        """

        inv_loss_scaling = 1 / self.loss_scaling
        inv_loss_scaling = inv_loss_scaling.astype(jnp.float32)   # cast to float32, so the result is float32 (otherwise the whole scaling point would be senseless)
        return jax.tree_util.tree_map(lambda x: x * inv_loss_scaling, tree)
    
    def adjust(self, grads_finite: jnp.ndarray) -> 'DynamicLossScaling':
        """
        Adjust the loss scaling based on the finiteness of gradients and update the internal counter.
        It follows https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html and is directly adopted form JMP https://github.com/google-deepmind/jmp .
        Parameters:
            grads_finite (jnp.ndarray):
                A boolean scalar (0-dimensional) indicating whether all gradients are finite.
                Must satisfy grads_finite.ndim == 0.
        Returns:
            DynamicLossScaling:
                A new instance of DynamicLossScaling. Use this and replace the current instance with it.
        """
        
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
