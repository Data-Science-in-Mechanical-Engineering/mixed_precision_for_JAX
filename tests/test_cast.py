import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree
import numpy as np

from mpfj.cast import (
    cast_tree,
    cast_to_float32,
    cast_to_float16,
    cast_to_bfloat16,
    cast_to_full_precision,
    cast_to_half_precision,
    force_full_precision,
)
from mpfj.dtypes import HALF_PRECISION_DATATYPE


class EQXModuleBase(eqx.Module):
    a: Array
    b: Array

    def __init__(self):
        self.a = jnp.ones(10, dtype=jnp.float32)
        self.b = jnp.ones(10, dtype=jnp.float32)

class LeafClass:
    """If implemented correctly, this class should not be casted"""
    a: Array
    b: Array

    def __init__(self):
        self.a = jnp.ones(10, dtype=jnp.float32)
        self.b = jnp.ones(10, dtype=jnp.float32)

class EQXModule1(eqx.Module):
    a: list[EQXModuleBase]
    b: Array
    c: LeafClass

    def __init__(self):
        self.a = [EQXModuleBase() for _ in range(10)]
        self.b = jnp.ones(10, dtype=jnp.float32)
        self.c = LeafClass()


class TestCastFunctions(unittest.TestCase):
    def setUp(self):
        # Create some test data
        self.array_float32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        self.array_float16 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float16)
        self.array_bfloat16 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.bfloat16)
        self.nested_dict = {
            'a': self.array_float32,
            'b': {'c': self.array_float16, 'd': self.array_bfloat16}
        }
        self.mixed_tree = {
            'array': self.array_float32,
            'scalar': 42,
            'nested': {
                'array': self.array_float16,
                'none': None
            }
        }
    
    def test_cast_eqx_module(self):
        # Create test module
        module = EQXModule1()
        
        # Test casting to float16
        result = cast_tree(module, jnp.float16)
        # Check that arrays in nested EQXModuleBase instances are cast
        for base_module in result.a:
            self.assertEqual(base_module.a.dtype, jnp.float16)
            self.assertEqual(base_module.b.dtype, jnp.float16)
        # Check direct array is cast
        self.assertEqual(result.b.dtype, jnp.float16)
        # Check that LeafClass arrays are NOT cast since it's not an eqx.Module
        self.assertEqual(result.c.a.dtype, jnp.float32)
        self.assertEqual(result.c.b.dtype, jnp.float32)

        # Test casting to bfloat16 
        result = cast_tree(module, jnp.bfloat16)
        # Check nested modules
        for base_module in result.a:
            self.assertEqual(base_module.a.dtype, jnp.bfloat16)
            self.assertEqual(base_module.b.dtype, jnp.bfloat16)
        self.assertEqual(result.b.dtype, jnp.bfloat16)
        # LeafClass should remain unchanged
        self.assertEqual(result.c.a.dtype, jnp.float32)
        self.assertEqual(result.c.b.dtype, jnp.float32)

        # Test casting back to float32
        result = cast_tree(module, jnp.float32)
        for base_module in result.a:
            self.assertEqual(base_module.a.dtype, jnp.float32)
            self.assertEqual(base_module.b.dtype, jnp.float32)
        self.assertEqual(result.b.dtype, jnp.float32)
        self.assertEqual(result.c.a.dtype, jnp.float32)
        self.assertEqual(result.c.b.dtype, jnp.float32)

    def test_cast_tree(self):
        # Test casting to float32
        result = cast_tree(self.array_float16, jnp.float32)
        self.assertEqual(result.dtype, jnp.float32)
        
        # Test casting nested structure
        result = cast_tree(self.nested_dict, jnp.float32)
        self.assertEqual(result['a'].dtype, jnp.float32)
        self.assertEqual(result['b']['c'].dtype, jnp.float32)
        self.assertEqual(result['b']['d'].dtype, jnp.float32)

    def test_cast_to_float32(self):
        result = cast_to_float32(self.array_float16)
        self.assertEqual(result.dtype, jnp.float32)
        
        result = cast_to_float32(self.nested_dict)
        self.assertEqual(result['a'].dtype, jnp.float32)
        self.assertEqual(result['b']['c'].dtype, jnp.float32)
        self.assertEqual(result['b']['d'].dtype, jnp.float32)

    def test_cast_to_float16(self):
        result = cast_to_float16(self.array_float32)
        self.assertEqual(result.dtype, jnp.float16)
        
        result = cast_to_float16(self.nested_dict)
        self.assertEqual(result['a'].dtype, jnp.float16)
        self.assertEqual(result['b']['c'].dtype, jnp.float16)
        self.assertEqual(result['b']['d'].dtype, jnp.float16)

    def test_cast_to_bfloat16(self):
        result = cast_to_bfloat16(self.array_float32)
        self.assertEqual(result.dtype, jnp.bfloat16)
        
        result = cast_to_bfloat16(self.nested_dict)
        self.assertEqual(result['a'].dtype, jnp.bfloat16)
        self.assertEqual(result['b']['c'].dtype, jnp.bfloat16)
        self.assertEqual(result['b']['d'].dtype, jnp.bfloat16)

    def test_cast_to_full_precision(self):
        result = cast_to_full_precision(self.array_float16)
        self.assertEqual(result.dtype, jnp.float32)
        
        result = cast_to_full_precision(self.nested_dict)
        self.assertEqual(result['a'].dtype, jnp.float32)
        self.assertEqual(result['b']['c'].dtype, jnp.float32)
        self.assertEqual(result['b']['d'].dtype, jnp.float32)

    def test_cast_to_half_precision(self):
        result = cast_to_half_precision(self.array_float32)
        self.assertEqual(result.dtype, HALF_PRECISION_DATATYPE)
        
        result = cast_to_half_precision(self.nested_dict)
        self.assertEqual(result['a'].dtype, HALF_PRECISION_DATATYPE)
        self.assertEqual(result['b']['c'].dtype, HALF_PRECISION_DATATYPE)
        self.assertEqual(result['b']['d'].dtype, HALF_PRECISION_DATATYPE)

    def test_force_full_precision_decorator(self):
        @force_full_precision
        def test_func(x, y):
            return x + y, x * y

        # Test with float16 inputs
        x = jnp.array([1.0, 2.0], dtype=jnp.float16)
        y = jnp.array([3.0, 4.0], dtype=jnp.float16)
        
        result1, result2 = test_func(x, y)
        
        # Check that inputs were converted to float32 during computation
        self.assertEqual(result1.dtype, jnp.float16)  # Output is cast back to float16
        self.assertEqual(result2.dtype, jnp.float16)  # Output is cast back to float16

    def test_mixed_tree_handling(self):
        # Test that non-array elements are preserved
        result = cast_to_float32(self.mixed_tree)
        self.assertEqual(result['array'].dtype, jnp.float32)
        self.assertEqual(result['scalar'], 42)
        self.assertEqual(result['nested']['none'], None)
        self.assertEqual(result['nested']['array'].dtype, jnp.float32)

    def test_empty_structures(self):
        # Test with empty structures
        empty_dict = {}
        result = cast_to_float32(empty_dict)
        self.assertEqual(result, {})

        empty_list = []
        result = cast_to_float32(empty_list)
        self.assertEqual(result, [])

    def test_non_float_arrays(self):
        # Test that non-float arrays are not cast
        int32_array = jnp.array([1, 2, 3], dtype=jnp.int32)
        bool_array = jnp.array([True, False], dtype=jnp.bool_)
        
        # Test with single arrays
        result = cast_to_float32(int32_array)
        self.assertEqual(result.dtype, jnp.int32)
        
        result = cast_to_float32(bool_array)
        self.assertEqual(result.dtype, jnp.bool_)
        
        # Test with nested structure containing mixed dtypes
        mixed_tree = {
            'float32': jnp.array([1.0, 2.0], dtype=jnp.float32),
            'int32': int32_array,
            'bool': bool_array,
            'nested': {
                'float16': jnp.array([1.0, 2.0], dtype=jnp.float16),
                'int32': int32_array
            }
        }
        
        result = cast_to_float32(mixed_tree)
        # Check that float arrays are cast
        self.assertEqual(result['float32'].dtype, jnp.float32)
        self.assertEqual(result['nested']['float16'].dtype, jnp.float32)
        # Check that non-float arrays remain unchanged
        self.assertEqual(result['int32'].dtype, jnp.int32)
        self.assertEqual(result['bool'].dtype, jnp.bool_)
        self.assertEqual(result['nested']['int32'].dtype, jnp.int32)

if __name__ == '__main__':
    unittest.main()
