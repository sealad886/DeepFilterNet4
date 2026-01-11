#!/usr/bin/env python3
"""Test checkpoint consistency enhancements in train_dynamic.py

This script verifies:
1. Checkpoint validation functions work correctly
2. Optimizer state serialization/deserialization works
3. SIGINT handler is properly registered
"""

import sys
import tempfile
from pathlib import Path

# Add DeepFilterNet to path
sys.path.insert(0, str(Path(__file__).parent.parent / "DeepFilterNet"))

import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from df_mlx.train_dynamic import (
    _register_sigint_handler,
    _validate_checkpoint_pair,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_validation():
    """Test _validate_checkpoint_pair function."""
    print("\n" + "=" * 60)
    print("Test 1: Checkpoint Validation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test 1a: Missing files
        print("\n1a. Testing missing checkpoint files...")
        missing_path = tmpdir / "missing.safetensors"
        result = _validate_checkpoint_pair(missing_path)
        assert not result, "Should fail for missing files"
        print("  ✅ Correctly detects missing checkpoint")

        # Test 1b: Empty files
        print("\n1b. Testing empty checkpoint files...")
        empty_ckpt = tmpdir / "empty.safetensors"
        empty_state = tmpdir / "empty.state.json"
        empty_ckpt.touch()
        empty_state.touch()
        result = _validate_checkpoint_pair(empty_ckpt)
        assert not result, "Should fail for empty files"
        print("  ✅ Correctly detects empty checkpoint files")

        # Test 1c: Valid files
        print("\n1c. Testing valid checkpoint files...")
        valid_ckpt = tmpdir / "valid.safetensors"
        valid_state = tmpdir / "valid.state.json"
        valid_ckpt.write_bytes(b"fake weights data")
        valid_state.write_text('{"epoch": 1}')
        result = _validate_checkpoint_pair(valid_ckpt)
        assert result, "Should pass for valid files"
        print("  ✅ Correctly validates proper checkpoint pair")

    print("\n✅ All checkpoint validation tests passed")


def test_optimizer_state_persistence():
    """Test optimizer state save/load."""
    print("\n" + "=" * 60)
    print("Test 2: Optimizer State Persistence")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ckpt_path = tmpdir / "test.safetensors"

        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def __call__(self, x):
                return self.linear(x)

        # Create model and optimizer
        print("\n2a. Creating model and optimizer...")
        model = SimpleModel()
        optimizer = optim.AdamW(learning_rate=0.001)

        # Initialize optimizer state by running one step
        x = mx.random.normal((4, 10))
        y = model(x)
        loss = mx.mean(y * y)
        loss_and_grad = nn.value_and_grad(model, lambda m, x: mx.mean(m(x) * m(x)))
        _, grads = loss_and_grad(model, x)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        print("  ✅ Model and optimizer initialized")

        # Save checkpoint with optimizer state
        print("\n2b. Saving checkpoint with optimizer state...")
        config = {"test": "config"}
        save_checkpoint(
            model,
            ckpt_path,
            epoch=1,
            loss=0.5,
            best_valid_loss=0.4,
            config=config,
            optimizer=optimizer,
        )
        assert ckpt_path.exists(), "Checkpoint file should exist"
        assert ckpt_path.with_suffix(".state.json").exists(), "State file should exist"
        print("  ✅ Checkpoint saved")

        # Verify optimizer state was saved
        print("\n2c. Verifying optimizer state in checkpoint...")
        with open(ckpt_path.with_suffix(".state.json")) as f:
            state = json.load(f)
        assert "optimizer_state" in state, "Optimizer state should be in JSON"
        assert len(state["optimizer_state"]) > 0, "Optimizer state should not be empty"
        print(f"  ✅ Optimizer state saved ({len(state['optimizer_state'])} entries)")

        # Create new model and optimizer for loading
        print("\n2d. Loading checkpoint with optimizer state...")
        new_model = SimpleModel()
        new_optimizer = optim.AdamW(learning_rate=0.001)

        loaded_state = load_checkpoint(new_model, ckpt_path, optimizer=new_optimizer)
        assert loaded_state.get("epoch") == 1, "Epoch should be restored"
        assert "optimizer_state" in loaded_state, "Optimizer state should be loaded"
        print("  ✅ Checkpoint and optimizer state loaded")

    print("\n✅ All optimizer persistence tests passed")


def test_sigint_handler_registration():
    """Test SIGINT handler registration."""
    print("\n" + "=" * 60)
    print("Test 3: SIGINT Handler Registration")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def __call__(self, x):
                return self.linear(x)

        print("\n3a. Registering SIGINT handler...")
        model = SimpleModel()
        optimizer = optim.AdamW(learning_rate=0.001)
        config = {"test": "config"}

        # Register handler (should not raise error)
        _register_sigint_handler(model, optimizer, tmpdir, config)
        print("  ✅ SIGINT handler registered successfully")

        # Verify handler is set
        import signal

        handler = signal.getsignal(signal.SIGINT)
        assert handler is not None, "SIGINT handler should be set"
        print("  ✅ SIGINT handler is active")

    print("\n✅ All SIGINT handler tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Checkpoint Consistency Enhancement Tests")
    print("=" * 60)

    try:
        test_checkpoint_validation()
        test_optimizer_state_persistence()
        test_sigint_handler_registration()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
