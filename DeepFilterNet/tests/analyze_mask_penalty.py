#!/usr/bin/env python3
"""Analyze mask saturation penalty logic."""

import mlx.core as mx

print("Mask Saturation Penalty Analysis")
print("=" * 60)

confident_mask = mx.array([[0.05, 0.95, 0.02, 0.98]])
uncertain_mask = mx.array([[0.45, 0.55, 0.48, 0.52]])


def current_saturation_penalty(raw_mask):
    mask_extreme = mx.mean(raw_mask * (1.0 - raw_mask))
    mask_saturation_loss = 1.0 - 4.0 * mask_extreme
    return mx.clip(mask_saturation_loss, 0.0, 1.0)


def fixed_saturation_penalty(raw_mask):
    mask_entropy = mx.mean(raw_mask * (1.0 - raw_mask))
    return 4.0 * mask_entropy


print("Confident mask (near 0 or 1):")
conf_entropy = float(mx.mean(confident_mask * (1.0 - confident_mask)))
print(f"  mask * (1-mask) = {conf_entropy:.4f}")
print(f"  Current penalty = {float(current_saturation_penalty(confident_mask)):.4f}")
print(f"  Fixed penalty   = {float(fixed_saturation_penalty(confident_mask)):.4f}")

print("")
print("Uncertain mask (near 0.5):")
unc_entropy = float(mx.mean(uncertain_mask * (1.0 - uncertain_mask)))
print(f"  mask * (1-mask) = {unc_entropy:.4f}")
print(f"  Current penalty = {float(current_saturation_penalty(uncertain_mask)):.4f}")
print(f"  Fixed penalty   = {float(fixed_saturation_penalty(uncertain_mask)):.4f}")

print("")
print("Interpretation:")
print("  Current: Lower loss for uncertain masks (WRONG - rewards uncertainty)")
print("  Fixed:   Lower loss for confident masks (CORRECT - rewards confidence)")
