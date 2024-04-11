# Gradient Ascent Jax

This repository provides a custom packaged implementation of the Gradient Ascent solver with advanced momentum and adaptive LR features for JAX and Flax models.

**Optimizer Built-In Features:**

- Gradient Ascent optimization strategy, tailored for maximizing objective functions.
- Momentum to facilitate smoother and more stable convergence by incorporating previous gradient steps.
- Adaptive LR mechanism that adjusts according to the magnitude of parameter gradients, promoting efficient progress towards the optimization peak.
- Optional clipping, a crucial safeguard against the detrimental effects of exploding gradients, thereby ensuring stable training dynamics.
- LR Decay, a methodical reduction of the learning rate over time, which helps fine-tune the model's performance as it approaches the optimal solution.
- Logging functionalities, capturing and reporting the evolution of the learning rate and the norm of parameters throughout the training process.

## **Installation**

**Requirements**

```bash
jax==0.4.25
jaxlib==0.4.25
```

You can install the package using `pip3 install -e .`:

```bash
git clone https://github.com/attophyd/GradientAscent-Jax
cd GradientAscent-Jax
pip3 install -e .
```

## **Example**

You can easily initialize the solver as follows:

```python
import jax
import jax.numpy as jnp
from gradient_ascent_jax.gradient_ascent import GradientAscent


def f(x):
    return -(x**2)


def grad_f(x):
    return -2 * x


x_init = jnp.array(10.0)
parameters = {"x": x_init}

solver = GradientAscent(
    parameters=parameters, lr=0.1, momentum=0.9, warmup_steps=0, logging_interval=1
)

for step in range(50):
    grads = {"x": grad_f(solver.parameters["x"])}
    parameters = solver.step(grads)
    if step % 10 == 0:
        print(f"Step {step}, x: {parameters['x']}")
```