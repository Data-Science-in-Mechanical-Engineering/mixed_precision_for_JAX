import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree
import numpy as np

from mpx import set_forward_backward_precision

from mpx.experimental import cast_function_fwd_bwd


class MLP(eqx.Module):
    a: Array
    b: Array

    def __init__(self):
        self.a = jnp.ones((10, 10), dtype=jnp.float32)
        self.b = jnp.ones(10, dtype=jnp.float32)

    def __call__(self, x):
        return jax.nn.relu(self.a @ x + self.b)


class TestFP8(unittest.TestCase):
    def setUp(self):
        # Create some test data
        self.array_float32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    
    def test_cast_function_fwd_bwd(self):
        # Create test module
        module = MLP()
        for bwd_dtype in [jnp.float32]:
            set_forward_backward_precision(jnp.float8_e5m2, bwd_dtype)

            def loss_fn(mdl, inp):
                out = mdl(inp)
                return jnp.sum(out)
            loss_fn_fp8 = cast_function_fwd_bwd(loss_fn)
            
            x = jnp.ones((10,1), dtype=jnp.float32)

            # test forward pass
            # the output should be the backward datatype as we only cast multiplications to fwd
            output = loss_fn_fp8(module, x)
            output_original = loss_fn(module, x)
            print(output)
            print(output_original)
            self.assertTrue(np.allclose(output, output_original, atol=1e-4))
            self.assertEqual(output.dtype, bwd_dtype)
            
            # test backward pass
            grad_fn_fp8 = jax.grad(loss_fn_fp8)
            grad_fn = jax.grad(loss_fn)
            grads_fp8 = grad_fn_fp8(module, x)
            grads = grad_fn(module, x)

            # as MLP and x all have the same values, the gradients should be the same
            # (for other values, the gradients will differ slightly due to quantization errors)
            self.assertTrue(np.allclose(grads_fp8.a, grads.a, atol=1e-4))
            self.assertTrue(np.allclose(grads_fp8.b, grads.b, atol=1e-4))

            self.assertEqual(grads_fp8.a.dtype, bwd_dtype)
            self.assertEqual(grads_fp8.b.dtype, bwd_dtype)

            # test now with values where quantization errors are larger
            x = jnp.arange(10, dtype=bwd_dtype).reshape((10,1)) + 1.0
            output = loss_fn_fp8(module, x)
            output_original = loss_fn(module, x)
            grads_fp8 = grad_fn_fp8(module, x)
            grads = grad_fn(module, x)

            self.assertFalse(np.allclose(output, output_original, atol=1e-4))
            self.assertFalse(np.allclose(grads_fp8.a, grads.a, atol=1e-4))
            # bias is in fp32, so it should be close
            self.assertTrue(np.allclose(grads_fp8.b, grads.b, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
