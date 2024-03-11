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
