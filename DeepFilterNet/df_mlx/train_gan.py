"""GAN training script for MLX DeepFilterNet4.

This module provides GAN-based adversarial training for speech enhancement,
integrating:
- Generator (DfNet4 model)
- Multi-Period and Multi-Scale Discriminators
- Feature matching loss
- Progressive adversarial training with warmup

Based on the PyTorch implementation in df/train.py with adaptations for MLX.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from loguru import logger

from .checkpoint import CheckpointManager, PatienceState, check_patience
from .config import LossConfig
from .discriminator import CombinedDiscriminator
from .evaluation import ValidationMetrics
from .loss import CombinedLoss, FeatureMatchingLoss, discriminator_loss, generator_loss
from .lr import CosineScheduler
from .model import DfNet4, count_parameters
from .train import MultiResolutionSTFTLoss


@dataclass
class GANConfig:
    """Configuration for GAN training."""

    # Generator training
    generator_lr: float = 5e-4
    generator_weight_decay: float = 0.05
    generator_grad_clip: float = 1.0

    # Discriminator training
    discriminator_lr: float = 1e-4
    discriminator_weight_decay: float = 0.05
    discriminator_grad_clip: float = 1.0

    # GAN schedule
    gan_warmup_epochs: int = 50
    gan_schedule_start: int = 100
    gan_schedule_end: int = 200

    # Loss weights
    lambda_adv: float = 0.1
    lambda_fm: float = 2.0
    lambda_spec: float = 1.0
    lambda_sisdr: float = 1.0

    # Training parameters
    batch_size: int = 4
    max_epochs: int = 300
    steps_per_epoch: int = 1000

    # LR scheduling
    warmup_epochs: int = 5
    lr_decay_mul: float = 0.2
    lr_min: float = 1e-6
    cycle_length: int = 100

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 5
    patience: int = 10
    patience_window: int = 5

    # Logging
    log_every_steps: int = 100
    eval_every_epochs: int = 1

    # Discriminator settings
    mpd_periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    msd_scales: int = 3

    # Sample rate
    sample_rate: int = 48000


class GANTrainer:
    """GAN training manager for DeepFilterNet4.

    Implements adversarial training with:
    - Multi-Period and Multi-Scale Discriminators
    - Feature matching loss
    - Progressive adversarial weight scheduling
    - Patience-based early stopping
    """

    def __init__(
        self,
        generator: DfNet4,
        config: GANConfig,
        loss_config: Optional[LossConfig] = None,
        resume_from: Optional[str] = None,
    ):
        self.generator = generator
        self.config = config
        self.loss_config = loss_config or LossConfig()

        # Initialize discriminator
        self.discriminator = CombinedDiscriminator(
            mpd_periods=config.mpd_periods,
            msd_scales=config.msd_scales,
        )

        # Optimizers
        self.gen_optimizer = optim.AdamW(
            learning_rate=config.generator_lr,
            weight_decay=config.generator_weight_decay,
        )
        self.disc_optimizer = optim.AdamW(
            learning_rate=config.discriminator_lr,
            weight_decay=config.discriminator_weight_decay,
        )

        # LR Schedulers
        self.gen_lr_schedule = CosineScheduler(
            base_lr=config.generator_lr,
            min_lr=config.lr_min,
            epochs=config.max_epochs,
            steps_per_epoch=config.steps_per_epoch,
            warmup_epochs=config.warmup_epochs,
        )
        self.disc_lr_schedule = CosineScheduler(
            base_lr=config.discriminator_lr,
            min_lr=config.lr_min,
            epochs=config.max_epochs,
            steps_per_epoch=config.steps_per_epoch,
            warmup_epochs=config.warmup_epochs,
        )

        # Loss functions
        self.spectral_loss = MultiResolutionSTFTLoss.from_config(self.loss_config)
        self.feature_matching_loss = FeatureMatchingLoss()
        self.combined_loss = CombinedLoss(self.loss_config)

        # Checkpoint manager
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            max_to_keep=5,
        )

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.patience_state = PatienceState(
            max_patience=config.patience,
        )

        # Resume if provided
        if resume_from:
            self.load_checkpoint(resume_from)

    def get_gan_weight(self, epoch: int) -> float:
        """Get current adversarial loss weight based on schedule.

        Returns 0 during warmup, then linearly increases to lambda_adv.
        """
        if epoch < self.config.gan_warmup_epochs:
            return 0.0

        start = self.config.gan_schedule_start
        end = self.config.gan_schedule_end

        if epoch < start:
            return 0.0
        elif epoch >= end:
            return self.config.lambda_adv
        else:
            # Linear ramp
            progress = (epoch - start) / (end - start)
            return self.config.lambda_adv * progress

    def generator_loss_fn(
        self,
        generator: DfNet4,
        noisy_spec: Tuple[mx.array, mx.array],
        feat_erb: mx.array,
        feat_spec: mx.array,
        clean_spec: Tuple[mx.array, mx.array],
        clean_audio: mx.array,
        gan_weight: float,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Compute generator loss.

        Args:
            generator: Generator model
            noisy_spec: Noisy spectrum (real, imag)
            feat_erb: ERB features
            feat_spec: DF features
            clean_spec: Clean spectrum (real, imag)
            clean_audio: Clean audio waveform
            gan_weight: Weight for adversarial loss

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Generate enhanced spectrum
        pred_spec = generator(noisy_spec, feat_erb, feat_spec)
        pred_real, pred_imag = pred_spec

        # Spectral loss
        spec_loss = self.spectral_loss(
            self._spec_to_audio(pred_spec),
            clean_audio,
        )

        loss_dict = {"spec_loss": spec_loss}
        total_loss = spec_loss * self.config.lambda_spec

        # Adversarial loss (if enabled)
        if gan_weight > 0:
            # Get enhanced audio
            pred_audio = self._spec_to_audio(pred_spec)

            # Get discriminator outputs
            disc_fake, disc_fake_feats = self.discriminator(pred_audio)
            disc_real, disc_real_feats = self.discriminator(clean_audio)

            # Generator adversarial loss
            gen_adv_loss = generator_loss(disc_fake)
            loss_dict["gen_adv_loss"] = gen_adv_loss
            total_loss = total_loss + gan_weight * gen_adv_loss

            # Feature matching loss
            if self.config.lambda_fm > 0:
                fm_loss = self.feature_matching_loss(disc_fake_feats, disc_real_feats)
                loss_dict["fm_loss"] = fm_loss
                total_loss = total_loss + self.config.lambda_fm * fm_loss

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict

    def discriminator_loss_fn(
        self,
        discriminator: CombinedDiscriminator,
        pred_audio: mx.array,
        clean_audio: mx.array,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Compute discriminator loss.

        Args:
            discriminator: Discriminator model
            pred_audio: Generated audio
            clean_audio: Real clean audio

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Stop gradient on generated audio
        pred_audio_sg = mx.stop_gradient(pred_audio)

        # Get discriminator outputs
        disc_fake, _ = discriminator(pred_audio_sg)
        disc_real, _ = discriminator(clean_audio)

        # Discriminator loss (returns total, real_loss, fake_loss)
        total_loss, real_loss, fake_loss = discriminator_loss(disc_real, disc_fake)

        loss_dict = {
            "disc_loss": total_loss,
            "disc_real_loss": real_loss,
            "disc_fake_loss": fake_loss,
        }

        return total_loss, loss_dict

    def _spec_to_audio(self, spec: Tuple[mx.array, mx.array]) -> mx.array:
        """Convert spectrum to audio using ISTFT.

        This is a placeholder - in practice, use the DF state for synthesis.
        """
        from .ops import istft

        real, imag = spec
        audio = istft(real, imag)
        return audio

    def train_step_generator(
        self,
        batch: Dict[str, mx.array],
        gan_weight: float,
    ) -> Dict[str, float]:
        """Execute generator training step.

        Args:
            batch: Training batch
            gan_weight: Current adversarial loss weight

        Returns:
            Dictionary of loss values
        """
        # Update LR
        gen_lr = self.gen_lr_schedule.step()
        self.gen_optimizer.learning_rate = gen_lr

        # Unpack batch
        noisy_spec = (batch["noisy_real"], batch["noisy_imag"])
        feat_erb = batch["feat_erb"]
        feat_spec = batch["feat_spec"]
        clean_spec = (batch["clean_real"], batch["clean_imag"])
        clean_audio = batch["clean_audio"]

        # Compute loss and gradients
        def loss_fn(generator):
            loss, loss_dict = self.generator_loss_fn(
                generator,
                noisy_spec,
                feat_erb,
                feat_spec,
                clean_spec,
                clean_audio,
                gan_weight,
            )
            return loss

        loss, grads = nn.value_and_grad(self.generator, loss_fn)(self.generator)

        # Gradient clipping
        if self.config.generator_grad_clip > 0:
            grads, grad_norm = optim.clip_grad_norm(grads, self.config.generator_grad_clip)
        else:
            grad_norm = mx.array(0.0)

        # Update parameters
        self.gen_optimizer.update(self.generator, grads)
        mx.eval(loss)

        return {
            "gen_loss": float(loss),
            "gen_grad_norm": float(grad_norm),
            "gen_lr": gen_lr,
        }

    def train_step_discriminator(
        self,
        batch: Dict[str, mx.array],
    ) -> Dict[str, float]:
        """Execute discriminator training step.

        Args:
            batch: Training batch

        Returns:
            Dictionary of loss values
        """
        # Update LR
        disc_lr = self.disc_lr_schedule.step()
        self.disc_optimizer.learning_rate = disc_lr

        # Get clean and enhanced audio
        clean_audio = batch["clean_audio"]

        # Generate enhanced audio (no gradient)
        noisy_spec = (batch["noisy_real"], batch["noisy_imag"])
        feat_erb = batch["feat_erb"]
        feat_spec = batch["feat_spec"]

        pred_spec = self.generator(noisy_spec, feat_erb, feat_spec)
        pred_audio = self._spec_to_audio(pred_spec)

        # Compute loss and gradients
        def loss_fn(discriminator):
            loss, _ = self.discriminator_loss_fn(discriminator, pred_audio, clean_audio)
            return loss

        loss, grads = nn.value_and_grad(self.discriminator, loss_fn)(self.discriminator)

        # Gradient clipping
        if self.config.discriminator_grad_clip > 0:
            grads, grad_norm = optim.clip_grad_norm(grads, self.config.discriminator_grad_clip)
        else:
            grad_norm = mx.array(0.0)

        # Update parameters
        self.disc_optimizer.update(self.discriminator, grads)
        mx.eval(loss)

        return {
            "disc_loss": float(loss),
            "disc_grad_norm": float(grad_norm),
            "disc_lr": disc_lr,
        }

    def train_epoch(
        self,
        train_loader: Iterator,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data iterator
            epoch: Current epoch number

        Returns:
            Dictionary of average losses
        """
        gan_weight = self.get_gan_weight(epoch)
        use_gan = gan_weight > 0

        # Accumulators
        gen_losses = []
        disc_losses = []
        step_times = []

        for step, batch in enumerate(train_loader):
            if step >= self.config.steps_per_epoch:
                break

            start_time = time.time()

            # Generator step
            gen_metrics = self.train_step_generator(batch, gan_weight)
            gen_losses.append(gen_metrics["gen_loss"])

            # Discriminator step (if GAN training active)
            if use_gan:
                disc_metrics = self.train_step_discriminator(batch)
                disc_losses.append(disc_metrics["disc_loss"])
            else:
                disc_metrics = {"disc_loss": 0.0}

            step_time = time.time() - start_time
            step_times.append(step_time)

            self.global_step += 1

            # Logging
            if step % self.config.log_every_steps == 0:
                avg_gen = np.mean(gen_losses[-self.config.log_every_steps :])
                avg_disc = np.mean(disc_losses[-self.config.log_every_steps :]) if disc_losses else 0.0
                avg_time = np.mean(step_times[-self.config.log_every_steps :])

                logger.info(
                    f"Epoch {epoch} Step {step}/{self.config.steps_per_epoch} | "
                    f"Gen: {avg_gen:.4f} | Disc: {avg_disc:.4f} | "
                    f"GAN wt: {gan_weight:.3f} | {1 / avg_time:.1f} steps/s"
                )

        return {
            "gen_loss": float(np.mean(gen_losses)),
            "disc_loss": float(np.mean(disc_losses)) if disc_losses else 0.0,
            "gan_weight": gan_weight,
        }

    def validate(
        self,
        val_loader: Iterator,
        num_batches: int = 50,
    ) -> Dict[str, float]:
        """Run validation.

        Args:
            val_loader: Validation data iterator
            num_batches: Number of batches to evaluate

        Returns:
            Dictionary of validation metrics
        """
        metrics = ValidationMetrics(self.config.sample_rate, ["si-sdr", "snr"])

        total_loss = 0.0
        count = 0

        for batch in val_loader:
            if count >= num_batches:
                break

            noisy_spec = (batch["noisy_real"], batch["noisy_imag"])
            feat_erb = batch["feat_erb"]
            feat_spec = batch["feat_spec"]
            clean_audio = batch["clean_audio"]
            noisy_audio = batch.get("noisy_audio")

            # Forward pass
            pred_spec = self.generator(noisy_spec, feat_erb, feat_spec)
            pred_audio = self._spec_to_audio(pred_spec)

            # Spectral loss
            loss = self.spectral_loss(pred_audio, clean_audio)
            mx.eval(loss)
            total_loss += float(loss)

            # Update metrics
            metrics.update(clean_audio, pred_audio, noisy_audio)

            count += 1

        val_metrics = metrics.compute()
        val_metrics["loss"] = total_loss / max(count, 1)

        return val_metrics

    def train(
        self,
        train_loader: Iterator,
        val_loader: Optional[Iterator] = None,
    ) -> Dict[str, Any]:
        """Run full training loop.

        Args:
            train_loader: Training data iterator (yields batches)
            val_loader: Optional validation data iterator

        Returns:
            Training history
        """
        logger.info(f"Starting GAN training for {self.config.max_epochs} epochs")
        logger.info(f"Generator parameters: {count_parameters(self.generator):,}")
        logger.info(f"Discriminator parameters: {count_parameters(self.discriminator):,}")

        history = {
            "train_gen_loss": [],
            "train_disc_loss": [],
            "val_loss": [],
            "val_sisdr": [],
        }

        start_time = time.time()

        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train_gen_loss"].append(train_metrics["gen_loss"])
            history["train_disc_loss"].append(train_metrics["disc_loss"])

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch} complete | "
                f"Gen Loss: {train_metrics['gen_loss']:.4f} | "
                f"Disc Loss: {train_metrics['disc_loss']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Validation
            if val_loader and epoch % self.config.eval_every_epochs == 0:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_sisdr"].append(val_metrics.get("si-sdr", 0.0))

                logger.info(
                    f"Validation | Loss: {val_metrics['loss']:.4f} | "
                    f"SI-SDR: {val_metrics.get('si-sdr', 0.0):.2f} dB"
                )

                # Check patience - returns (is_best, should_stop, new_state)
                improved, should_stop, self.patience_state = check_patience(
                    val_metrics["loss"],
                    self.patience_state,
                )

                if improved:
                    self.best_loss = val_metrics["loss"]
                    self.save_checkpoint("best")
                    logger.info(f"New best model! Loss: {self.best_loss:.4f}")

                if should_stop:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Periodic checkpoint
            if epoch % self.config.save_every_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}")

        # Final checkpoint
        self.save_checkpoint("final")

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_loss:.4f}")

        return history

    def save_checkpoint(self, name: str):
        """Save training checkpoint.

        Args:
            name: Checkpoint name (without extension)
        """
        from mlx.utils import tree_flatten

        checkpoint_path = self.checkpoint_dir / f"{name}.safetensors"

        # Save generator
        gen_params = self.generator.parameters()
        gen_flat = tree_flatten(gen_params)
        gen_weights = {f"generator.{k}": v for k, v in gen_flat}

        # Save discriminator
        disc_params = self.discriminator.parameters()
        disc_flat = tree_flatten(disc_params)
        disc_weights = {f"discriminator.{k}": v for k, v in disc_flat}

        # Combine and save
        all_weights = {**gen_weights, **disc_weights}
        mx.save_safetensors(str(checkpoint_path), all_weights)

        # Save training state
        state_path = checkpoint_path.with_suffix(".json")
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "patience_state": self.patience_state.to_dict(),
            "gen_lr_state": self.gen_lr_schedule.state_dict(),
            "disc_lr_state": self.disc_lr_schedule.state_dict(),
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, path_str: str):
        """Load training checkpoint.

        Args:
            path_str: Path to checkpoint file
        """
        path = Path(path_str)

        # Load weights
        weights: Dict[str, mx.array] = mx.load(str(path))  # type: ignore

        # Separate generator and discriminator weights
        gen_weights = {k.replace("generator.", ""): v for k, v in weights.items() if k.startswith("generator.")}
        disc_weights = {
            k.replace("discriminator.", ""): v for k, v in weights.items() if k.startswith("discriminator.")
        }

        self.generator.load_weights(list(gen_weights.items()))
        self.discriminator.load_weights(list(disc_weights.items()))

        # Load training state
        state_path = path.with_suffix(".json")
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)

            self.epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
            self.best_loss = state.get("best_loss", float("inf"))

            # Restore patience state
            if "patience_state" in state:
                self.patience_state = PatienceState.from_dict(state["patience_state"])

            # Restore LR scheduler state
            if "gen_lr_state" in state:
                self.gen_lr_schedule.load_state_dict(state["gen_lr_state"])
            if "disc_lr_state" in state:
                self.disc_lr_schedule.load_state_dict(state["disc_lr_state"])

        logger.info(f"Loaded checkpoint: {path} (epoch {self.epoch})")


# ============================================================================
# Convenience Functions
# ============================================================================


def train_gan(
    generator: DfNet4,
    train_loader: Iterator,
    val_loader: Optional[Iterator] = None,
    config: Optional[GANConfig] = None,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for GAN training.

    Args:
        generator: Generator model (DfNet4)
        train_loader: Training data iterator
        val_loader: Optional validation iterator
        config: GAN training configuration
        resume_from: Optional checkpoint to resume from

    Returns:
        Training history
    """
    if config is None:
        config = GANConfig()

    trainer = GANTrainer(generator, config, resume_from=resume_from)
    return trainer.train(train_loader, val_loader)
