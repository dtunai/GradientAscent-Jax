import jax
import jax.numpy as jnp


class GradientAscent:
    """
    Custom gradient ascent optimizer for JAX/Flax models, featuring momentum and Nesterov momentum options,
    gradient clipping, learning rate decay, and warmup functionality.

    Attr:
        parameters (dict): A dictionary of model parameters to be optimized
        lr (float): Initial learning rate for the optimizer. Default is 0.01
        momentum (float): Momentum factor to accelerate optimization in the direction of consistent gradient. Default is 0.9
        clip_value (float, optional): The maximum gradient norm if gradient clipping is applied. None means no clipping
        lr_decay (float): Learning rate decay factor applied each decay step. Default is 1.0 (no decay)
        warmup_steps (int): Number of steps to linearly scale the learning rate from 0 to the initial learning rate. Default is 0 (no warmup)
        logging_interval (int): Interval in steps between logging of optimization information. Default is 10
        nesterov (bool): If True, uses Nesterov momentum instead of standard momentum. Default is False
        decay_step (int): Number of steps between each decay of the learning rate. Default is 10
        decay_rate (float): Factor by which the learning rate decays at each decay step. Default is 0.9

    Methods:
        clip_grads(grads): Clips the gradients to a maximum norm specified by clip_value
        step(grads): Performs a single optimization step using the provided gradients
        adjust_learning_rate(): Adjusts the learning rate based on the current step count, applying warmup and decay
    """

    def __init__(
        self,
        parameters,
        lr=0.01,
        momentum=0.9,
        clip_value=None,
        lr_decay=1.0,
        warmup_steps=0,
        logging_interval=10,
        nesterov=False,
        decay_step=10,
        decay_rate=0.9,
    ):
        """
        Init the optimizer class with the provided parameters and configuration
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.clip_value = clip_value
        self.lr_decay = lr_decay
        self.warmup_steps = warmup_steps
        self.logging_interval = logging_interval
        self.nesterov = nesterov
        self.step_count = 0

        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.v = jax.tree_map(jnp.zeros_like, parameters)

    def clip_grads(self, grads):
        """
        Clips gradients to a maximum norm
        """
        if self.clip_value is None:
            return grads

        norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_leaves(grads)]))
        clip_coef = jnp.minimum(1, self.clip_value / (norm + 1e-6))
        return jax.tree_map(lambda g: g * clip_coef, grads)

    def step(self, grads):
        """
        Single optimization step using the provided gradients
        """
        self.step_count += 1
        grads = self.clip_grads(grads)

        lr = self.adjust_learning_rate()

        def update_param(param, grad, v):
            if self.nesterov:
                v_prev = v
                v = self.momentum * v + grad
                update = lr * ((1 + self.momentum) * v - self.momentum * v_prev)
            else:
                v = self.momentum * v + grad
                update = lr * v

            return param + update, v

        updated_params_and_v = jax.tree_map(
            update_param, self.parameters, grads, self.v, is_leaf=lambda x: isinstance(x, tuple)
        )

        for key in self.parameters:
            self.parameters[key], self.v[key] = updated_params_and_v[key]

        if self.step_count % self.logging_interval == 0:
            print(f"Step: {self.step_count}, Learning Rate: {lr}")

        return self.parameters

    def adjust_learning_rate(self):
        if self.step_count <= self.warmup_steps:
            lr = self.lr * (self.step_count / self.warmup_steps)
        else:
            decay_factor = self.decay_rate ** (self.step_count // self.decay_step)
            lr = self.lr * decay_factor
        return lr
