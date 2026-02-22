# Configuration Schemas for DeepFilterNet4

This directory contains JSON Schema files for TOML configuration validation and IDE autocomplete.

## Quick Setup (VS Code)

### 1. Install TOML Extension

Install **Even Better TOML** from the VS Code marketplace:
- Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on macOS)
- Search for "Even Better TOML"
- Click Install

Or install via command line:
```bash
code --install-extension tamasfe.even-better-toml
```

### 2. Configure Schema Association

Add to your `.vscode/settings.json` (or use the provided template below):

```json
{
  "evenBetterToml.schema.associations": {
    "^file:///ABSOLUTE/PATH/TO/CONFIGS/.*\\.toml$": "file:///ABSOLUTE/PATH/TO/REPO/schemas/run-config.schema.json"
  }
}
```

Important:
- Keys are matched against **absolute document URIs** (e.g. `file:///Users/name/.../run.toml`)
- Values must be **absolute schema URIs** (e.g. `file:///Users/name/repo/schemas/run-config.schema.json`)

### 3. (Optional) Set Up Local Development Environment

If you have a local `.vscode/` directory (not tracked by git):

```bash
# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Copy the schema association to your local settings
cat >> .vscode/settings.json << 'EOF'
{
  "evenBetterToml.schema.associations": {
    "^file:///ABSOLUTE/PATH/TO/CONFIGS/.*\\.toml$": "file:///ABSOLUTE/PATH/TO/REPO/schemas/run-config.schema.json"
  }
}
EOF
```

## Available Schemas

### `run-config.schema.json`

JSON Schema for `df_mlx.train_dynamic` run configurations.

**Features:**
- ✅ Autocomplete for all configuration keys
- ✅ Inline documentation on hover
- ✅ Type validation (numbers, booleans, strings, enums)
- ✅ Default value hints
- ✅ Range validation (min/max for numeric values)
- ✅ Enum validation for choice fields (e.g., `backbone_type`, `dynamic_loss`)

**Covers:**
- `[dataset]` - Dataset paths and SNR ranges
- `[augmentation]` - Reverb and clipping probabilities
- `[training]` - Epochs, batch size, learning rate, gradient accumulation
- `[dataloader]` - Workers, prefetch, MLX data settings
- `[checkpoint]` - Save/resume strategy and directory
- `[model]` - Backbone type and variant
- `[loss]` - Dynamic loss type, awesome loss, multi-res STFT
- `[gan]` - GAN training configuration
- `[vad]` - Voice activity detection settings
- `[metrics]` - Evaluation metrics
- `[debug]` - Debugging and numeric stability options
- `[train_ini]` - Legacy INI compatibility tables

## Usage Example

When editing a TOML file matched by the schema association:

```toml
[training]
# Press Ctrl+Space after the '=' to see autocomplete
epochs = 100
batch_size = 20
# Hover over 'fp16' to see: "Enable FP16 (true/false) or 'auto' for hardware default"
fp16 = "auto"  # IDE will suggest: "auto", true, false

[model]
# Type 'backbone_type = ' and see available options
backbone_type = "attention"  # Options: "mamba", "gru", "attention"

[loss]
dynamic_loss = "pipeline_awesome"  # Options: "baseline", "awesome", "pipeline_awesome"

[gan]
enabled = true
start_epoch = 40  # Must be >= 0 (validated by schema)
adv_weight = 0.07  # Must be >= 0.0
```

**Validation Examples:**
```toml
[training]
# ❌ Error: must be >= 1
epochs = 0

# ❌ Error: must be one of ["auto", true, false]
fp16 = "maybe"

[model]
# ❌ Error: must be one of ["mamba", "gru", "attention"]
backbone_type = "transformer"
```

## Alternative: Per-File Schema Reference

You can also add a schema comment at the top of any TOML file:

```toml
#:schema file:///ABSOLUTE/PATH/TO/REPO/schemas/run-config.schema.json

[training]
epochs = 100
```

This explicitly associates the file with the schema, regardless of filename pattern.

## Maintaining Schemas

If you modify `DeepFilterNet/df_mlx/run_config.py`:

1. Update `run-config.schema.json` to reflect new fields/constraints
2. Test with a sample TOML file to verify autocomplete
3. Commit the updated schema

The schema structure mirrors the Python dataclasses in `run_config.py`.

## See Also

- [DeepFilterNet/df_mlx/run_config.py](../DeepFilterNet/df_mlx/run_config.py) - Source of truth for config structure
- [Even Better TOML Documentation](https://github.com/tamasfe/taplo)
