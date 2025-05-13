import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from mpfj.grad_tools import select_tree, filter_grad, filter_value_and_grad, optimizer_update
from mpfj.loss_scaling import DynamicLossScaling


# Create a simple model for testing
class SimpleModel(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    
    def __init__(self):
        self.weight = jnp.array([1.0, 2.0], dtype=jnp.float32)
        self.bias = jnp.array([0.5], dtype=jnp.float32)
    
    def __call__(self, x):
        return jnp.dot(x, self.weight) + self.bias


class TestGradTools(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.loss_scaling = DynamicLossScaling(
            loss_scaling=jnp.array([2**10], dtype=jnp.float32),
            min_loss_scaling=jnp.array([2**-14], dtype=jnp.float32),
            factor=2,
            period=2000
        )
        
        self.model = SimpleModel()
        self.optimizer = optax.sgd(learning_rate=0.1)
        self.optimizer_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

    def test_select_tree(self):
        """Test the select_tree function"""
        tree_a = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0, 4.0])}
        tree_b = {'a': jnp.array([5.0, 6.0]), 'b': jnp.array([7.0, 8.0])}
        
        # Test with True predicate
        result = select_tree(jnp.array(True), tree_a, tree_b)
        self.assertTrue(jnp.allclose(result['a'], tree_a['a']))
        self.assertTrue(jnp.allclose(result['b'], tree_a['b']))
        
        # Test with False predicate
        result = select_tree(jnp.array(False), tree_a, tree_b)
        self.assertTrue(jnp.allclose(result['a'], tree_b['a']))
        self.assertTrue(jnp.allclose(result['b'], tree_b['b']))

    def test_filter_grad(self):
        """Test the filter_grad function"""
        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)
        
        x = jnp.array([1.0, 2.0])
        y = jnp.array([2.0])
        
        # Test with finite gradients
        grad_fn = filter_grad(loss_fn, self.loss_scaling)
        loss_scaling_new, grads_finite, grad = grad_fn(self.model, x, y)
        
        self.assertTrue(bool(grads_finite))
        self.assertIsInstance(grad, SimpleModel)
        
        # Test with infinite gradients
        def bad_loss_fn(model, x, y):
            pred = model(x)
            return jnp.inf * jnp.mean((pred - y) ** 2)
        
        grad_fn = filter_grad(bad_loss_fn, self.loss_scaling)
        loss_scaling_new, grads_finite, grad = grad_fn(self.model, x, y)
        self.assertFalse(bool(grads_finite))

    def test_filter_value_and_grad(self):
        """Test the filter_value_and_grad function"""
        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)
        
        x = jnp.array([1.0, 2.0])
        y = jnp.array([2.0])
        
        # Test without aux
        value_grad_fn = filter_value_and_grad(loss_fn, self.loss_scaling)
        value, loss_scaling_new, grads_finite, grad = value_grad_fn(self.model, x, y)
        
        self.assertIsInstance(value, jnp.ndarray)
        self.assertTrue(grads_finite)
        self.assertIsInstance(grad, SimpleModel)
        
        # Test with aux
        def loss_fn_with_aux(model, x, y):
            pred = model(x)
            loss = jnp.mean((pred - y) ** 2)
            return loss, {'pred': pred}
        
        value_grad_fn = filter_value_and_grad(loss_fn_with_aux, self.loss_scaling, has_aux=True)
        (value, aux), loss_scaling_new, grads_finite, grad = value_grad_fn(self.model, x, y)
        
        self.assertIsInstance(value, jnp.ndarray)
        self.assertIsInstance(aux, dict)
        self.assertIn('pred', aux)
        self.assertTrue(grads_finite)
        self.assertIsInstance(grad, SimpleModel)

    def test_optimizer_update(self):
        """Test the optimizer_update function"""
        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)
        
        x = jnp.array([1.0, 2.0])
        y = jnp.array([2.0])
        
        # Calculate gradients using filter_grad
        grad_fn = filter_grad(loss_fn, self.loss_scaling)
        loss_scaling_new, grads_finite, grads = grad_fn(self.model, x, y)
        
        # Test with finite gradients
        new_model, new_optimizer_state = optimizer_update(
            self.model, self.optimizer, self.optimizer_state, grads, grads_finite
        )
        
        self.assertIsInstance(new_model, type(self.model))
        self.assertIsInstance(new_optimizer_state, tuple)
        
        # Test with infinite gradients
        def bad_loss_fn(model, x, y):
            pred = model(x)
            return jnp.inf * jnp.mean((pred - y) ** 2)
        
        grad_fn = filter_grad(bad_loss_fn, self.loss_scaling)
        loss_scaling_new, grads_finite, grads = grad_fn(self.model, x, y)
        
        new_model, new_optimizer_state = optimizer_update(
            self.model, self.optimizer, self.optimizer_state, grads, grads_finite
        )
        
        # Model should remain unchanged
        self.assertTrue(jnp.allclose(new_model.weight, self.model.weight))
        self.assertTrue(jnp.allclose(new_model.bias, self.model.bias))

if __name__ == '__main__':
    unittest.main() 