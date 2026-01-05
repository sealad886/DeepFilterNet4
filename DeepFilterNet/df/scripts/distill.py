#!/usr/bin/env python3
"""Knowledge Distillation Training Script for DeepFilterNet4.

This script trains a smaller "student" model (DfNet4Lite) to mimic
a larger "teacher" model (DfNet4) using knowledge distillation.

The distillation process uses:
1. Hard labels: Standard loss from ground truth clean speech
2. Soft labels: KL divergence loss between teacher and student outputs
3. Feature matching: MSE loss between intermediate representations

Usage:
    python -m df.scripts.distill \\
        --teacher-checkpoint /path/to/teacher/model.ckpt \\
        --output-dir /path/to/output \\
        --config /path/to/config.ini \\
        --temperature 4.0 \\
        --alpha 0.7

Arguments:
    --teacher-checkpoint: Path to pre-trained teacher model checkpoint
    --output-dir: Directory to save student model checkpoints
    --config: Path to training config file
    --temperature: Distillation temperature (higher = softer targets)
    --alpha: Weight for distillation loss vs hard label loss (0-1)
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from df.config import config
from df.checkpoint import read_cp, write_cp
from df.loss import Loss


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard and soft targets.
    
    The total loss is:
        L = alpha * L_soft + (1 - alpha) * L_hard
    
    Where:
        L_soft: KL divergence between teacher and student outputs (soft targets)
        L_hard: Standard reconstruction loss from ground truth
        
    Args:
        temperature: Softmax temperature for soft targets
        alpha: Weight for soft target loss (1-alpha for hard targets)
        feature_weight: Weight for intermediate feature matching loss
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        feature_weight: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        
    def forward(
        self,
        student_output: Tuple[Tensor, ...],
        teacher_output: Tuple[Tensor, ...],
        target: Tensor,
        hard_loss: Tensor,
        student_features: Optional[List[Tensor]] = None,
        teacher_features: Optional[List[Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Compute distillation loss.
        
        Args:
            student_output: Student model outputs (spec, mask, lsnr, alpha)
            teacher_output: Teacher model outputs (same format)
            target: Ground truth clean spectrogram
            hard_loss: Pre-computed hard label loss
            student_features: Optional intermediate student features
            teacher_features: Optional intermediate teacher features
            
        Returns:
            Dictionary with loss components and total loss
        """
        # Soft target loss (KL divergence on outputs)
        # Use spectral output for distillation
        student_spec = student_output[0]
        teacher_spec = teacher_output[0]
        
        # Apply temperature scaling
        student_soft = F.log_softmax(student_spec.flatten(2) / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_spec.flatten(2) / self.temperature, dim=-1)
        
        # KL divergence (multiply by T^2 as per Hinton et al.)
        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean",
        ) * (self.temperature ** 2)
        
        # Mask output distillation
        student_mask = student_output[1]
        teacher_mask = teacher_output[1]
        mask_loss = F.mse_loss(student_mask, teacher_mask.detach())
        
        # Feature matching loss
        feature_loss = torch.tensor(0.0, device=student_spec.device)
        if student_features is not None and teacher_features is not None:
            for s_feat, t_feat in zip(student_features, teacher_features):
                # Align dimensions if needed
                if s_feat.shape != t_feat.shape:
                    # Project student features to teacher dimension
                    if s_feat.shape[-1] != t_feat.shape[-1]:
                        continue  # Skip mismatched features
                feature_loss = feature_loss + F.mse_loss(s_feat, t_feat.detach())
            if len(student_features) > 0:
                feature_loss = feature_loss / len(student_features)
        
        # Combine losses
        soft_total = soft_loss + mask_loss
        total_loss = (
            self.alpha * soft_total +
            (1 - self.alpha) * hard_loss +
            self.feature_weight * feature_loss
        )
        
        return {
            "total": total_loss,
            "soft": soft_loss,
            "hard": hard_loss,
            "mask": mask_loss,
            "feature": feature_loss,
        }


class DistillationTrainer:
    """Trainer for knowledge distillation.
    
    Handles the complete distillation training loop including:
    - Teacher model inference (frozen)
    - Student model training
    - Loss computation and optimization
    - Checkpoint saving and logging
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: Loss,
        distill_loss: DistillationLoss,
        device: torch.device,
        checkpoint_dir: str,
    ):
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.distill_loss = distill_loss
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Move to device
        self.teacher = self.teacher.to(device)
        self.student = self.student.to(device)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.student.train()
        
        total_loss = 0.0
        total_soft = 0.0
        total_hard = 0.0
        total_mask = 0.0
        total_feature = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Unpack batch (format depends on dataloader)
            feat_erb = batch["feat_erb"].to(self.device)
            feat_spec = batch["feat_spec"].to(self.device)
            target_spec = batch.get("target_spec", batch.get("clean_spec"))
            if target_spec is not None:
                target_spec = target_spec.to(self.device)
            
            # Teacher inference (no grad)
            with torch.no_grad():
                teacher_output = self.teacher(feat_erb, feat_spec)
                
            # Student forward
            student_output = self.student(feat_erb, feat_spec)
            
            # Hard label loss
            hard_loss = self.loss_fn(
                student_output[0],  # enhanced spec
                target_spec,
            ) if target_spec is not None else torch.tensor(0.0, device=self.device)
            
            # Distillation loss
            losses = self.distill_loss(
                student_output,
                teacher_output,
                target_spec,
                hard_loss,
            )
            
            # Backward
            self.optimizer.zero_grad()
            losses["total"].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += losses["total"].item()
            total_soft += losses["soft"].item()
            total_hard += losses["hard"].item()
            total_mask += losses["mask"].item()
            total_feature += losses["feature"].item()
            num_batches += 1
            
        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
            
        return {
            "loss": total_loss / num_batches,
            "soft_loss": total_soft / num_batches,
            "hard_loss": total_hard / num_batches,
            "mask_loss": total_mask / num_batches,
            "feature_loss": total_feature / num_batches,
        }
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.valid_loader is None:
            return {}
            
        self.student.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.valid_loader:
            feat_erb = batch["feat_erb"].to(self.device)
            feat_spec = batch["feat_spec"].to(self.device)
            target_spec = batch.get("target_spec", batch.get("clean_spec"))
            if target_spec is not None:
                target_spec = target_spec.to(self.device)
            
            # Teacher inference
            teacher_output = self.teacher(feat_erb, feat_spec)
            
            # Student forward
            student_output = self.student(feat_erb, feat_spec)
            
            # Hard label loss only for validation
            hard_loss = self.loss_fn(
                student_output[0],
                target_spec,
            ) if target_spec is not None else torch.tensor(0.0, device=self.device)
            
            total_loss += hard_loss.item()
            num_batches += 1
            
        return {
            "val_loss": total_loss / num_batches if num_batches > 0 else 0.0,
        }
        
    def train(
        self,
        num_epochs: int,
        save_every: int = 1,
        log_every: int = 10,
    ) -> Dict[str, List[float]]:
        """Run full training loop.
        
        Args:
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            log_every: Log metrics every N batches
            
        Returns:
            Dictionary of metric histories
        """
        history = {
            "train_loss": [],
            "val_loss": [],
        }
        
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            history["train_loss"].append(train_metrics["loss"])
            
            # Validate
            val_metrics = self.validate()
            if "val_loss" in val_metrics:
                history["val_loss"].append(val_metrics["val_loss"])
                
            # Log
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Soft: {train_metrics['soft_loss']:.4f} | "
                f"Hard: {train_metrics['hard_loss']:.4f} | "
                f"Val Loss: {val_metrics.get('val_loss', 0):.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                write_cp(
                    self.student,
                    "student",
                    self.checkpoint_dir,
                    epoch + 1,
                )
                write_cp(
                    self.optimizer,
                    "opt",
                    self.checkpoint_dir,
                    epoch + 1,
                )
                
            # Save best
            if val_metrics.get("val_loss", float("inf")) < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                write_cp(
                    self.student,
                    "student",
                    self.checkpoint_dir,
                    epoch + 1,
                    metric=best_val_loss,
                    cmp="min",
                )
                logger.info(f"New best model saved with val_loss={best_val_loss:.4f}")
                
        return history


def compare_models(
    teacher: nn.Module,
    student: nn.Module,
    sample_input: Tuple[Tensor, Tensor],
) -> Dict[str, float]:
    """Compare teacher and student model statistics.
    
    Args:
        teacher: Teacher model
        student: Student model
        sample_input: Sample input (feat_erb, feat_spec)
        
    Returns:
        Dictionary with comparison metrics
    """
    teacher.eval()
    student.eval()
    
    # Parameter counts
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    # Model sizes
    teacher_size = sum(p.numel() * p.element_size() for p in teacher.parameters()) / 1024 / 1024
    student_size = sum(p.numel() * p.element_size() for p in student.parameters()) / 1024 / 1024
    
    # Inference time
    device = next(teacher.parameters()).device
    feat_erb, feat_spec = sample_input
    feat_erb = feat_erb.to(device)
    feat_spec = feat_spec.to(device)
    
    # Warmup
    with torch.no_grad():
        teacher(feat_erb, feat_spec)
        student(feat_erb, feat_spec)
        
    # Time teacher
    import time
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            teacher(feat_erb, feat_spec)
    teacher_time = (time.time() - start) / 10
    
    # Time student
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            student(feat_erb, feat_spec)
    student_time = (time.time() - start) / 10
    
    return {
        "teacher_params": teacher_params,
        "student_params": student_params,
        "param_reduction": 1 - student_params / teacher_params,
        "teacher_size_mb": teacher_size,
        "student_size_mb": student_size,
        "size_reduction": 1 - student_size / teacher_size,
        "teacher_time_ms": teacher_time * 1000,
        "student_time_ms": student_time * 1000,
        "speedup": teacher_time / student_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for DFNet4")
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained teacher model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save student model checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Distillation temperature (default: 4.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for distillation loss (default: 0.7)",
    )
    parser.add_argument(
        "--feature-weight",
        type=float,
        default=0.1,
        help="Weight for feature matching loss (default: 0.1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare teacher/student models, don't train",
    )
    
    args = parser.parse_args()
    
    # Load config
    config.load(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load teacher model
    logger.info(f"Loading teacher model from {args.teacher_checkpoint}")
    from df.model import init_model
    from df.deepfilternet4 import DfNet4, DfNet4Lite, ModelParams4
    from df.modules import erb_fb
    from libdf import DF
    
    p = ModelParams4()
    df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    
    # Initialize teacher (full model)
    teacher = DfNet4(erb, erb_inverse, run_df=True, train_mask=True)
    teacher_state = torch.load(args.teacher_checkpoint, map_location="cpu")
    if "model" in teacher_state:
        teacher_state = teacher_state["model"]
    teacher.load_state_dict(teacher_state)
    
    # Initialize student (lite model)
    student = DfNet4Lite(erb, erb_inverse, run_df=True, train_mask=True)
    
    device = torch.device(args.device)
    teacher = teacher.to(device)
    student = student.to(device)
    
    # Sample input for comparison
    sample_input = (
        torch.randn(1, 1, 100, p.nb_erb),
        torch.randn(1, 1, 100, p.nb_df, 2),
    )
    
    # Compare models
    logger.info("Comparing teacher and student models...")
    comparison = compare_models(teacher, student, sample_input)
    
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"Teacher parameters: {comparison['teacher_params']:,}")
    print(f"Student parameters: {comparison['student_params']:,}")
    print(f"Parameter reduction: {comparison['param_reduction']*100:.1f}%")
    print(f"Teacher size: {comparison['teacher_size_mb']:.2f} MB")
    print(f"Student size: {comparison['student_size_mb']:.2f} MB")
    print(f"Size reduction: {comparison['size_reduction']*100:.1f}%")
    print(f"Teacher inference: {comparison['teacher_time_ms']:.2f} ms")
    print(f"Student inference: {comparison['student_time_ms']:.2f} ms")
    print(f"Speedup: {comparison['speedup']:.2f}x")
    print("=" * 60 + "\n")
    
    if args.compare_only:
        return
        
    # Setup training
    logger.info("Setting up distillation training...")
    
    # Initialize dataloader (placeholder - would use actual dataset)
    logger.warning("Dataloader setup requires actual dataset configuration")
    logger.info(
        "To run full distillation training, configure your dataset in the config file "
        "and update this script's dataloader initialization."
    )
    
    # For demonstration, show what the training loop would look like
    print("\nDistillation training would proceed with:")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Alpha (soft/hard weight): {args.alpha}")
    print(f"  - Feature weight: {args.feature_weight}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Device: {args.device}")
    print(f"  - Output: {args.output_dir}")
    
    logger.info("Distillation training script ready")


if __name__ == "__main__":
    main()
