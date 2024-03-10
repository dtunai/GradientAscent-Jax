import jax
import jax.numpy as jnp


class GradientAscent:
    """
    Custom gradient ascent solver (optimizer) for JAX/Flax models.

    Args:

    parameters: flax.linen.Module
        Model parameters to be updated

    lr: float
        Learning rate

    momentum: float
        Momentum factor

    beta: float
        Exponential decay rate for the second moment estimates

    eps: float
        Small constant to avoid division by zero

    nesterov: bool
        Whether to use Nesterov momentum

    clip_value: float
        Maximum allowed value for the gradients

    lr_decay: float
        Learning rate decay factor

    warmup_steps: int
        Number of warmup steps

    logging_interval: int
        Logging interval

    Usage:
        grad_ascent_solver = GradientAscent(parameters=model_params, lr=0.01, momentum=0.9, beta=0.999, eps=1e-8)

    """

    def __init__(
        self,
        parameters,
        lr=0.01,
        momentum=0.9,
        beta=0.999,
        eps=1e-8,
        nesterov=False,
        clip_value=None,
        lr_decay=None,
        warmup_steps=0,
        logging_interval=10,
    ):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.eps = eps
        self.nesterov = nesterov
        self.clip_value = clip_value
        self.lr_decay = lr_decay
        self.warmup_steps = warmup_steps
        self.logging_interval = logging_interval

        self.step_count = 0
        self.v = jax.tree_map(lambda p: jnp.zeros_like(p), parameters)
        self.m = jax.tree_map(lambda p: jnp.zeros_like(p), parameters)

    def step(self):
        self.step_count += 1
        for param, grad in zip(self.parameters, jax.tree_leaves(self.parameters)):
            if grad is not None:
                if self.clip_value:
                    grad = jnp.clip(grad, -self.clip_value, self.clip_value)

                if self.nesterov:
                    grad = grad + self.momentum * self.v[param]
                else:
                    grad = grad

                self.v[param] = self.momentum * self.v[param] + grad

                self.m[param] = self.beta * self.m[param] + (1 - self.beta) * grad**2
                adapted_lr = self.lr / (jnp.sqrt(self.m[param]) + self.eps)

                if self.lr_decay:
                    adapted_lr *= self.lr_decay

                self.parameters = jax.tree_multimap(
                    lambda p, v: p - adapted_lr * v, self.parameters, self.v
                )

        if self.step_count % self.logging_interval == 0:
            updated_param_norm = jnp.linalg.norm(
                jax.tree_multimap(lambda p, v: p - adapted_lr * v, self.parameters, self.v)
            )
            print(
                f"Step: {self.step_count}, Learning Rate: {self.lr}, Parameter Norm: {updated_param_norm}"
            )
