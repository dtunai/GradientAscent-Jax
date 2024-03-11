import jax
import jax.numpy as jnp

from flax import linen as nn
from gradient_ascent_jax.gradient_ascent import GradientAscent

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(name="fc", features=1)(x)


# Initialize model and optimizer
rng = jax.random.PRNGKey(0)
key, subkey = jax.random.split(rng)
params = Model().init(subkey, jnp.ones((1, 1)))

# Use custom GA optimizer
solver = GradientAscent(parameters=params, lr=0.01, momentum=0.9, beta=0.999, eps=1e-8)
