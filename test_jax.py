import unittest

import jax
import jax.numpy as jnp

class TestJaxGPU(unittest.TestCase):
    def test_backend(self):
        # Checks if jax is using the gpu backend
        self.assertEqual(jax.default_backend(), 'gpu', "JAX is not using GPU backend, something went wrong during the environment setup")

    def test_device_allocation(self):
        # Creates an array and checks where it is allocated
        x = jnp.array([1.0, 2.0, 3.0])
        self.assertEqual(x.device.platform, 'gpu', "Array is not allocated on GPU")

    def test_computation(self):
        # Tests computation
        key = jax.random.key(42)
        x = jax.random.normal(key, (1000, 1000))
        y = jnp.dot(x, x.T).block_until_ready()
        self.assertEqual(y.device.platform, 'gpu', "Computation did not occur on the gpu")


if __name__ == "__main__":
    unittest.main()
