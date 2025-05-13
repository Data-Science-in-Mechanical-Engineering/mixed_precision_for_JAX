import unittest
import jax.numpy as jnp
from mpfj.dtypes import half_precision_datatype, set_half_precision_datatype

class TestDtypes(unittest.TestCase):
    def test_default_half_precision(self):
        """Test that the default half precision datatype is float16"""
        self.assertEqual(half_precision_datatype(), jnp.float16)

    def test_set_half_precision_datatype(self):
        """Test setting half precision datatype to bfloat16"""
        set_half_precision_datatype(jnp.bfloat16)
        self.assertEqual(half_precision_datatype(), jnp.bfloat16)
        
        # Reset to default
        set_half_precision_datatype(jnp.float16)
        self.assertEqual(half_precision_datatype(), jnp.float16)

if __name__ == '__main__':
    unittest.main() 