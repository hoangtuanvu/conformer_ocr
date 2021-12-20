import torch.optim as optim


class NaiveScheduler:
    """A simple wrapper class for learning rate scheduling followed by scheduler in Vanilla Transformer.
    Args:
        optimizer (optim.Optimizer):
        lr_mul (float):
        d_model (int):
        n_warmup_steps (int):
    """

    def __init__(self, optimizer: optim.Optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        """Step with the inner optimizer."""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer."""
        self._optimizer.zero_grad()

    def get_optimizer(self):
        return self._optimizer

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """Learning rate scheduling per step."""

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def test():
    lr_mul = 5.0
    steps = 200000
    d_model = 256
    n_warmup_steps = 4000
    for i in range(1, steps):
        lr = lr_mul * (d_model ** -0.5) * min(i ** (-0.5), i * n_warmup_steps ** (-1.5))

        if i % 10 == 0:
            print(i, lr)


if __name__ == "__main__":
    test()

