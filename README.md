# Gradient Ascent Jax

This repository provides a custom implementation of the Gradient Ascent solver with momentum and adaptive learning rates for JAX and Flax models.

**Optimizer Built-In Features:**

- Gradient Ascent optimization algorithm
- Momentum for smoother convergence
- Adaptive learning rate based on parameter gradients
- Optional gradient clipping to prevent exploding gradients
- Support for learning rate decay
- Logging of learning rate and parameter norm during training


## **Installation**

You can install the package using `pip3 install -e .`:

```bash
git clone https://github.com/simudt/GradientAscent-Jax
cd GradientAscent-Jax
pip3 install -e .
```

## **Example**

You can easily initialize the solver as follows:

```python
import jax
import jax.numpy as jnp


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
```