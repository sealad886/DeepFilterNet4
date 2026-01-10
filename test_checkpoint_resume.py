#!/usr/bin/env python3
"""Test checkpoint save and resume functionality."""

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent))

from mlx.utils import tree_flatten  # noqa: E402

from scripts.train_dfnetmf_wall import load_checkpoint, save_checkpoint  # noqa: E402


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def test_checkpoint_save_load():
    """Test that checkpoint save and load work correctly."""
    print("Testing checkpoint save/load...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Create model and optimizer
        model = SimpleModel()
        optimizer = optim.AdamW(learning_rate=0.001)

        # Get initial weights
        initial_weights = dict(tree_flatten(model.parameters()))
        print(f"  Initial model: {list(initial_weights.keys())}")

        # Save checkpoint
        save_checkpoint(
            model,
            optimizer,
            step=100,
            epoch=5,
            metrics={"loss": 0.5, "accuracy": 0.95},
            checkpoint_dir=checkpoint_dir,
        )

        # Verify files exist
        state_file = checkpoint_dir / "step_000100_state.json"
        weights_file = checkpoint_dir / "step_000100.safetensors"

        assert state_file.exists(), f"State file not created: {state_file}"
        assert weights_file.exists(), f"Weights file not created: {weights_file}"
        print("  ✅ Checkpoint files created")

        # Verify state file content
        with open(state_file) as f:
            state = json.load(f)

        assert state["step"] == 100
        assert state["epoch"] == 5
        print(f"  ✅ State file correct: step={state['step']}, epoch={state['epoch']}")

        # Modify model weights to simulate different state by recreating model
        print("  Creating new model instance...")
        model2 = SimpleModel()

        # Verify weights are different
        initial_flat = dict(tree_flatten(model.parameters()))
        model2_flat = dict(tree_flatten(model2.parameters()))

        for key in initial_flat:
            if mx.allclose(initial_flat[key], model2_flat[key], rtol=1e-5, atol=1e-5):
                print("  ⚠️  Model 2 has same weights as Model 1 (unexpected)")
            else:
                print("  ✅ Model 2 has different weights (as expected)")
                break

        # Load checkpoint into model2
        loaded_step, loaded_epoch = load_checkpoint(checkpoint_dir, model2)

        assert loaded_step == 100, f"Expected step 100, got {loaded_step}"
        assert loaded_epoch == 5, f"Expected epoch 5, got {loaded_epoch}"
        print(f"  ✅ Checkpoint loaded into new model: step={loaded_step}, epoch={loaded_epoch}")

        # Verify weights were restored to match original model
        model2_restored = dict(tree_flatten(model2.parameters()))

        mismatches = 0
        for key in initial_flat:
            loaded = model2_restored[key]
            initial = initial_flat[key]
            match = mx.allclose(initial, loaded, rtol=1e-5, atol=1e-5)
            if not match:
                print(f"  ⚠️  Weight mismatch for {key}")
                mismatches += 1
            else:
                print(f"  ✅ Weight restored correctly: {key}")

        if mismatches == 0:
            print("\n✅ All checkpoint tests passed!")
        else:
            print(f"\n❌ {mismatches} weight mismatches found!")
            sys.exit(1)


if __name__ == "__main__":
    test_checkpoint_save_load()
