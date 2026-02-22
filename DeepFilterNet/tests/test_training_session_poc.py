"""POC: Validate nn.value_and_grad with bound methods and mutable self attributes.

This is the Phase 3 gate test. If it fails, the TrainingSession class design
needs rework (recreate value_and_grad each step instead of caching once).
"""

import mlx.core as mx
import mlx.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def __call__(self, x):
        return self.linear(x)


class LossHolder:
    """Simulate TrainingSession holding mutable training state."""

    def __init__(self, scale: float):
        self.scale = scale

    def loss_fn(self, model, x, y):
        pred = model(x)
        return self.scale * mx.mean((pred - y) ** 2)


def test_value_and_grad_with_bound_method():
    """nn.value_and_grad works with cached bound methods and reflects self mutations."""
    model = SimpleModel()
    holder = LossHolder(scale=1.0)

    # Cache value_and_grad once (simulate __init__ time caching)
    vg = nn.value_and_grad(model, holder.loss_fn)

    x = mx.random.normal((8, 4))
    y = mx.random.normal((8, 1))

    loss1, grads1 = vg(model, x, y)
    mx.eval(loss1, grads1)

    # Mutate self attribute (simulate GAN activation changing training state)
    holder.scale = 2.0
    loss2, grads2 = vg(model, x, y)
    mx.eval(loss2, grads2)

    # loss2 should be ~2x loss1 if self.scale mutation is visible
    ratio = loss2.item() / max(loss1.item(), 1e-12)
    assert abs(ratio - 2.0) < 0.01, (
        f"Bound method did not reflect mutated self.scale: "
        f"loss1={loss1.item():.6f}, loss2={loss2.item():.6f}, ratio={ratio:.4f}"
    )


def test_value_and_grad_with_dict_attribute():
    """nn.value_and_grad reflects mutations to mutable container attributes."""
    model = SimpleModel()

    class ConfigHolder:
        def __init__(self):
            self.weights = {"spectral": 1.0, "mrstft": 0.0}

        def loss_fn(self, model, x, y):
            pred = model(x)
            base = mx.mean((pred - y) ** 2)
            return self.weights["spectral"] * base + self.weights["mrstft"] * base

    holder = ConfigHolder()
    vg = nn.value_and_grad(model, holder.loss_fn)

    x = mx.random.normal((8, 4))
    y = mx.random.normal((8, 1))

    loss1, _ = vg(model, x, y)
    mx.eval(loss1)

    # Enable mrstft component
    holder.weights["mrstft"] = 1.0
    loss2, _ = vg(model, x, y)
    mx.eval(loss2)

    # loss2 should be ~2x loss1
    ratio = loss2.item() / max(loss1.item(), 1e-12)
    assert abs(ratio - 2.0) < 0.01, (
        f"Dict attribute mutation not reflected: "
        f"loss1={loss1.item():.6f}, loss2={loss2.item():.6f}, ratio={ratio:.4f}"
    )


def test_value_and_grad_with_boolean_flag():
    """nn.value_and_grad reflects boolean flag changes (GAN active toggle)."""
    model = SimpleModel()

    class FlagHolder:
        def __init__(self):
            self.gan_active = False

        def loss_fn(self, model, x, y):
            pred = model(x)
            base = mx.mean((pred - y) ** 2)
            if self.gan_active:
                return base * 3.0  # Simulate GAN adding extra loss
            return base

    holder = FlagHolder()
    vg = nn.value_and_grad(model, holder.loss_fn)

    x = mx.random.normal((8, 4))
    y = mx.random.normal((8, 1))

    loss1, _ = vg(model, x, y)
    mx.eval(loss1)

    holder.gan_active = True
    loss2, _ = vg(model, x, y)
    mx.eval(loss2)

    ratio = loss2.item() / max(loss1.item(), 1e-12)
    assert abs(ratio - 3.0) < 0.01, (
        f"Boolean flag mutation not reflected: "
        f"loss1={loss1.item():.6f}, loss2={loss2.item():.6f}, ratio={ratio:.4f}"
    )
