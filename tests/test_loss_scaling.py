import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
from mpx.loss_scaling import DynamicLossScaling, all_finite, scaled

class TestLossScaling(unittest.TestCase):
    def setUp(self):
        self.loss_scaling = DynamicLossScaling(
            loss_scaling=jnp.array([2**14], dtype=jnp.float32),
            min_loss_scaling=jnp.array([2**-14], dtype=jnp.float32),
            factor=2,
            period=2000
        )
        
        # Create some test data
        self.finite_tree = {
            'a': jnp.array([1.0, 2.0, 3.0]),
            'b': {'c': jnp.array([4.0, 5.0])}
        }
        
        self.infinite_tree = {
            'a': jnp.array([1.0, jnp.inf, 3.0]),
            'b': {'c': jnp.array([4.0, jnp.nan])}
        }

    def test_all_finite(self):
        """Test the all_finite function with finite and infinite values"""
        self.assertTrue(all_finite(self.finite_tree))
        self.assertFalse(all_finite(self.infinite_tree))
        self.assertTrue(all_finite({}))  # Empty tree should be considered finite

    def test_scale_unscale(self):
        """Test scaling and unscaling of values"""
        # Test scaling
        scaled_tree = self.loss_scaling.scale(self.finite_tree)
        self.assertEqual(scaled_tree['a'].dtype, jnp.float32)
        self.assertTrue(jnp.allclose(scaled_tree['a'], self.finite_tree['a'] * 2**14))
        
        # Test unscaling
        unscaled_tree = self.loss_scaling.unscale(scaled_tree)
        self.assertEqual(unscaled_tree['a'].dtype, jnp.float32)
        self.assertTrue(jnp.allclose(unscaled_tree['a'], self.finite_tree['a']))

    def test_adjust_finite_grads(self):
        """Test loss scaling adjustment with finite gradients"""
        # Test multiple periods to ensure scaling increases
        scaling = self.loss_scaling
        for _ in range(2000):  # One full period
            scaling = scaling.adjust(jnp.array(True))
        
        # After one period, scaling should have doubled
        self.assertAlmostEqual(scaling.loss_scaling[0], 2**15)
        self.assertEqual(scaling.counter[0], 0)  # Counter should reset

        for _ in range(2000):  # One full period
            scaling = scaling.adjust(jnp.array(True))

        # After two periods, scaling should not have doubled, but clipped to maximum of float16
        self.assertAlmostEqual(scaling.loss_scaling[0], (2 - 2**(-10)) * 2**15)
        self.assertEqual(scaling.counter[0], 0)  # Counter should reset

    def test_adjust_infinite_grads(self):
        """Test loss scaling adjustment with infinite gradients"""
        # Test with infinite gradients
        scaling = self.loss_scaling
        scaling = scaling.adjust(jnp.array(False))
        
        # Scaling should be halved
        self.assertEqual(scaling.loss_scaling[0], 2**13)
        self.assertEqual(scaling.counter[0], 0)  # Counter should reset

    def test_scaling_clipping(self):
        """Test that scaling is properly clipped"""
        # Test maximum clipping
        scaling = DynamicLossScaling(
            loss_scaling=jnp.array([2**16], dtype=jnp.float32),  # Above max
            min_loss_scaling=jnp.array([2**-14], dtype=jnp.float32),
            factor=2,
            period=2000
        )
        scaling = scaling.adjust(jnp.array(True))
        self.assertLessEqual(scaling.loss_scaling[0], (2 - 2**(-10)) * 2**15)
        
        # Test minimum clipping
        scaling = DynamicLossScaling(
            loss_scaling=jnp.array([2**-15], dtype=jnp.float32),  # Below min
            min_loss_scaling=jnp.array([2**-14], dtype=jnp.float32),
            factor=2,
            period=2000
        )
        scaling = scaling.adjust(jnp.array(False))
        self.assertGreaterEqual(scaling.loss_scaling[0], 2**-14)

    def test_scaled_decorator(self):
        """Test the scaled decorator"""
        def test_func(x):
            return x
        
        test_func = scaled(test_func, self.loss_scaling)
        
        result = test_func(self.finite_tree)
        self.assertTrue(jnp.allclose(result['a'], self.finite_tree['a'] * 2**14))

if __name__ == '__main__':
    unittest.main() 