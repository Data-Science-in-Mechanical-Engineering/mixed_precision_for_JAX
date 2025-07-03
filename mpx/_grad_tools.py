"""Filtering tools for mixer precision training."""


import jax
import jax.numpy as jnp
import equinox as eqx

import optax

import mpx._cast as cast
import mpx._loss_scaling as loss_scaling

from jaxtyping import PyTree, Bool


def select_tree(pred: jnp.ndarray, a: PyTree, b: PyTree) -> PyTree:
    """
    Selects elements from one of two pytrees based on a scalar boolean predicate.

    This function traverses two input pytrees (`a` and `b`) and selects elements
    from either `a` or `b` based on the value of the scalar boolean `pred`. If
    `pred` is `True`, elements from `a` are selected; otherwise, elements from `b`
    are selected. Non-array elements in the pytrees are taken directly from `a`.

    Args:
        pred (jnp.ndarray): A scalar boolean array (`jnp.bool_`) that determines
            which pytree to select elements from.
        a (PyTree): The first pytree to select elements from.
        b (PyTree): The second pytree to select elements from.

    Returns:
        PyTree: A new pytree with elements selected from `a` or `b` based on `pred`.

    Raises:
        AssertionError: If `pred` is not a scalar boolean array (`jnp.bool_`).
    """
    """Selects a pytree based on the given predicate."""
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    def _select_leaf(x1, x2):
        if eqx.is_array(x1):
            return jax.lax.select(pred, x1, x2)
        else:
            return x1

    return jax.tree_util.tree_map(_select_leaf, a, b)


def filter_grad(func, scaling: loss_scaling.DynamicLossScaling, has_aux=False, use_mixed_precision=True) -> PyTree:
    """
    Filters the gradients of a function based on a predicate.

    This function computes the gradients of the given function `func` with respect
    to its arguments (`args` and `kwargs`). It then filters the gradients based on
    a predicate function that checks whether the gradients are finite. The filtered
    gradients are returned as a new pytree.

    Args:
        func (callable): The function to compute gradients for. This function must only use pytrees as parameters!
        has_aux (bool): If True, the function is expected to return auxiliary values along with the gradients.
        use_mixed_precision (bool, optional): If True, the function will be cast to half
            precision. Defaults to True.
    Returns:
        callable: A function that computes the filtered gradients of `func`. It returns the grad, the new loss scaling, and a boolean indicating whether the gradients are finite (and the aux-value if has_aux is true).
    """
    def wrapper(*args, **kwargs):
        if use_mixed_precision:
            args_cast = tuple([cast.cast_to_half_precision(x) for x in args])
            kwargs_cast = {k: cast.cast_to_half_precision(v) for k, v in kwargs.items()}

            func_scaled = loss_scaling.scaled(func, scaling, has_aux=has_aux)
        else:
            args_cast = args
            kwargs_cast = kwargs
            func_scaled = func

        dfunc_scaled = eqx.filter_grad(func_scaled, has_aux=has_aux)

        if has_aux:
            grad, aux = dfunc_scaled(*args_cast, **kwargs_cast)
            if use_mixed_precision:
                grads_finite = loss_scaling.all_finite(grad)
                loss_scaling_new = scaling.adjust(grads_finite)
                grad = loss_scaling_new.unscale(grad)
            else:
                grads_finite = jnp.bool_(True)
                loss_scaling_new = scaling
            return loss_scaling_new, grads_finite, grad, aux
        else:
            grad = dfunc_scaled(*args_cast, **kwargs_cast)
            if use_mixed_precision:
                grad = cast.cast_to_full_precision(grad)
                grads_finite = loss_scaling.all_finite(grad)
                loss_scaling_new = scaling.adjust(grads_finite)
                grad = loss_scaling_new.unscale(grad)
            else:
                grad = cast.cast_to_full_precision(grad)
                grads_finite = jnp.bool_(True)
                loss_scaling_new = scaling
            return loss_scaling_new, grads_finite, grad

    return wrapper


def calculate_scaled_grad(func, scaling: loss_scaling.DynamicLossScaling, has_aux=False, use_mixed_precision=True) -> PyTree: 
    def wrapper(*args, **kwargs):
        if use_mixed_precision:
                args_cast = tuple([cast.cast_to_half_precision(x) for x in args])
                kwargs_cast = {k: cast.cast_to_half_precision(v) for k, v in kwargs.items()}
                func_scaled = loss_scaling.scaled(func, scaling, has_aux=has_aux)
        else:
            args_cast = args
            kwargs_cast = kwargs
            func_scaled = func

        dfunc_scaled = eqx.filter_value_and_grad(func_scaled, has_aux=has_aux)
        return dfunc_scaled(*args_cast, **kwargs_cast)
    return wrapper


def filter_value_and_grad(func, scaling: loss_scaling.DynamicLossScaling, has_aux=False, use_mixed_precision=True) -> PyTree:
    """
    Wraps a function to compute its value and gradient with support for mixed precision
    and dynamic loss scaling.
    Args:
        func (Callable): The function for which the value and gradient are to be computed.
        scaling (loss_scaling.DynamicLossScaling): An instance of DynamicLossScaling to
            handle loss scaling and gradient unscaling. 
        has_aux (bool, optional): Indicates whether the function `func` returns auxiliary
            outputs along with the main value. Defaults to False.
        use_mixed_precision (bool, optional): If True, the function will be cast to half
            precision. Defaults to True.
    Returns:
        Callable: A wrapped function that computes the value, gradient, and additional
        information:
            - If `has_aux` is True:
                ((value, aux), loss_scaling_new, grads_finite, grad)
            - If `has_aux` is False:
                (value, loss_scaling_new, grads_finite, grad)
        Where:
            - `value`: The computed value of the function.
            - `aux`: Auxiliary outputs returned by the function (if `has_aux` is True).
            - `loss_scaling_new`: The updated loss scaling object.
            - `grads_finite`: A boolean indicating whether all gradients are finite.
            - `grad`: The computed gradients, unscaled.
    """

    def wrapper(*args, **kwargs):
        dfunc_scaled = calculate_scaled_grad(func, scaling=scaling, has_aux=has_aux, use_mixed_precision=use_mixed_precision)

        if has_aux:
            (value, aux), grad = dfunc_scaled(*args, **kwargs)
            if use_mixed_precision:
                grads_finite = loss_scaling.all_finite(grad)
                loss_scaling_new = scaling.adjust(grads_finite)
                grad = loss_scaling_new.unscale(grad)
                value = loss_scaling_new.unscale(value)
            else:
                grads_finite = jnp.bool_(True)
                loss_scaling_new = scaling
            return (value, aux), loss_scaling_new, grads_finite, grad
        else:
            value, grad = dfunc_scaled(*args, **kwargs)
            if use_mixed_precision:
                grads_finite = loss_scaling.all_finite(grad)
                loss_scaling_new = scaling.adjust(grads_finite)
                grad = loss_scaling_new.unscale(grad)
                value = loss_scaling_new.unscale(value)
            else:
                grads_finite = jnp.bool_(True)
                loss_scaling_new = scaling
            return value, loss_scaling_new, grads_finite, grad

    return wrapper


def optimizer_update(model: PyTree, optimizer: optax.GradientTransformation, optimizer_state: PyTree, grads: PyTree, grads_finite: Bool):
    
    # optimizer step
    updates, new_optimizer_state = optimizer.update(
        grads, optimizer_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)

    # only apply updates to the model and optimizer state if gradients are finite
    new_model = select_tree(grads_finite, new_model, model)
    optimizer_state = select_tree(grads_finite, new_optimizer_state, optimizer_state)

    return new_model, optimizer_state
