#!/usr/bin/env python
"""ONNX export script for DeepFilterNet4 models.

This script exports DFNet4 models to ONNX format with validation.
Supports both full model export and component-wise export (encoder, decoders).

Usage:
    python -m df.scripts.export_onnx --model-base-dir /path/to/model --export-dir ./export
    python -m df.scripts.export_onnx --model-base-dir /path/to/model --export-dir ./export --full-model
    python -m df.scripts.export_onnx --checkpoint model.pt --config config.ini --export-dir ./export
"""

import argparse
import os
import shutil
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

try:
    import onnx     # type: ignore[import]
    import onnx.checker # type: ignore[import]
    import onnx.helper  # type: ignore[import]
    import onnxruntime as ort   # type: ignore[import]
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from df.config import config
from df.enhance import df_features
from df.io import get_test_sample, save_audio
from df.logger import init_logger
from df.utils import get_device
from libdf import DF


def shapes_dict(
    tensors: Tuple[Tensor, ...],
    names: Union[Tuple[str, ...], List[str]]
) -> Dict[str, Tuple[int, ...]]:
    """Create dictionary mapping tensor names to shapes."""
    if len(tensors) != len(names):
        logger.warning(
            f"Number of tensors ({len(tensors)}) does not match names: {names}"
        )
    return {k: tuple(v.shape) for (k, v) in zip(names, tensors)}


def onnx_simplify(
    path: str,
    input_data: Dict[str, Tensor],
    input_shapes: Dict[str, Iterable[int]]
) -> str:
    """Simplify ONNX model using onnxsim."""
    try:
        import onnxsim  # type: ignore[possibly-undefined]
    except ImportError:
        logger.warning("onnxsim not available, skipping simplification")
        return path
    
    model = onnx.load(path)  # type: ignore[possibly-undefined]
    model_simp, check = onnxsim.simplify(  # type: ignore[possibly-undefined]
        model,
        input_data=input_data,
        test_input_shapes=input_shapes,
    )
    model_n = os.path.splitext(os.path.basename(path))[0]
    
    if not check:
        logger.warning(f"Simplified model {model_n} could not be validated")
        return path
    
    try:
        onnx.checker.check_model(model_simp, full_check=True)  # type: ignore[possibly-undefined]
    except Exception as e:
        logger.error(f"Failed to simplify model {model_n}: {e}")
        return path
    
    onnx.save_model(model_simp, path)  # type: ignore[possibly-undefined]
    logger.info(f"Saved simplified model: {path}")
    return path


def onnx_check(
    path: str,
    input_dict: Dict[str, Tensor],
    output_names: Tuple[str, ...],
    providers: List[str] = None,
) -> List[np.ndarray]:
    """Validate ONNX model and run inference."""
    if providers is None:
        providers = ["CPUExecutionProvider"]
    
    model = onnx.load(path)  # type: ignore[possibly-undefined]
    logger.debug(f"{os.path.basename(path)}: {onnx.helper.printable_graph(model.graph)}")  # type: ignore[possibly-undefined]
    onnx.checker.check_model(model, full_check=True)  # type: ignore[possibly-undefined]
    
    sess = ort.InferenceSession(path, providers=providers)  # type: ignore[possibly-undefined]
    return sess.run(
        list(output_names),
        {k: v.numpy() for (k, v) in input_dict.items()}
    )


def export_impl(
    path: str,
    model: nn.Module,
    inputs: Tuple[Tensor, ...],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    jit: bool = False,
    opset_version: int = 14,
    check: bool = True,
    simplify: bool = True,
    print_graph: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-4,
) -> Tuple[Tensor, ...]:
    """Export a PyTorch model to ONNX format.
    
    Args:
        path: Output ONNX file path
        model: PyTorch model to export
        inputs: Example input tensors
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axis specification
        jit: Whether to JIT compile before export
        opset_version: ONNX opset version
        check: Whether to validate the exported model
        simplify: Whether to simplify the model
        print_graph: Whether to print the ONNX graph
        rtol: Relative tolerance for output comparison
        atol: Absolute tolerance for output comparison
        
    Returns:
        Model outputs as tuple of tensors
    """
    export_dir = os.path.dirname(path)
    if export_dir and not os.path.isdir(export_dir):
        logger.info(f"Creating export directory: {export_dir}")
        os.makedirs(export_dir)
    
    model_name = os.path.splitext(os.path.basename(path))[0]
    logger.info(f"Exporting model '{model_name}' to {export_dir or '.'}")
    
    input_shapes = shapes_dict(inputs, input_names)
    logger.info(f"  Input shapes: {input_shapes}")
    
    # Run model to get outputs
    outputs = model(*inputs)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    
    output_shapes = shapes_dict(outputs, output_names)
    logger.info(f"  Output shapes: {output_shapes}")
    logger.info(f"  Dynamic axes: {dynamic_axes}")
    
    # Optionally JIT compile
    export_model = deepcopy(model)
    if jit:
        try:
            export_model = torch.jit.script(export_model)
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}, using eager mode")
    
    # Export to ONNX
    torch.onnx.export(
        model=export_model,
        f=path,
        args=inputs,
        input_names=input_names,
        dynamic_axes=dynamic_axes,
        output_names=output_names,
        opset_version=opset_version,
        keep_initializers_as_inputs=False,
    )
    logger.info(f"  Exported to {path}")
    
    # Validate
    input_dict = {k: v for (k, v) in zip(input_names, inputs)}
    if check:
        try:
            onnx_outputs = onnx_check(path, input_dict, tuple(output_names))
            
            # Compare outputs
            all_close = True
            for name, pt_out, onnx_out in zip(output_names, outputs, onnx_outputs):
                try:
                    np.testing.assert_allclose(
                        pt_out.detach().numpy().squeeze(),
                        onnx_out.squeeze(),
                        rtol=rtol,
                        atol=atol,
                    )
                    logger.info(f"  ✓ Output '{name}' matches within tolerance")
                except AssertionError as e:
                    logger.warning(f"  ✗ Output '{name}' mismatch: {e}")
                    all_close = False
            
            if all_close:
                logger.info(f"  ✓ All outputs match PyTorch within tolerance")
        except Exception as e:
            logger.error(f"  Validation failed: {e}")
    
    # Simplify
    if simplify:
        path = onnx_simplify(path, input_dict, shapes_dict(inputs, input_names))
    
    # Print graph
    if print_graph:
        model_onnx = onnx.load(path)  # type: ignore[possibly-undefined]
        print(onnx.helper.printable_graph(model_onnx.graph))  # type: ignore[possibly-undefined]
    
    return outputs


@torch.no_grad()
def export_dfnet4(
    model: nn.Module,
    export_dir: str,
    df_state: DF,
    check: bool = True,
    simplify: bool = True,
    opset: int = 14,
    export_full: bool = False,
    export_components: bool = True,
    print_graph: bool = False,
    save_test_data: bool = True,
) -> Dict[str, str]:
    """Export DFNet4 model to ONNX format.
    
    Args:
        model: DFNet4 model instance
        export_dir: Directory for exported files
        df_state: DF state for feature extraction
        check: Whether to validate exported models
        simplify: Whether to simplify models
        opset: ONNX opset version
        export_full: Export full model as single ONNX file
        export_components: Export encoder/decoder components separately
        print_graph: Print ONNX graph structure
        save_test_data: Save test input/output data as npz files
        
    Returns:
        Dict mapping component names to ONNX file paths
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX and onnxruntime are required for export")
    
    model = deepcopy(model).to("cpu")
    model.eval()
    
    # Get model parameters
    from df.deepfilternet4 import ModelParams4
    p = ModelParams4()
    
    # Create test input
    audio = torch.randn((1, 1 * p.sr))
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")
    
    exported_paths = {}
    
    # Export full model
    if export_full:
        path = os.path.join(export_dir, "deepfilternet4.onnx")
        input_names = ["spec", "feat_erb", "feat_spec"]
        output_names = ["enh", "m", "lsnr", "coefs"]
        dynamic_axes = {
            "spec": {2: "T"},
            "feat_erb": {2: "T"},
            "feat_spec": {2: "T"},
            "enh": {2: "T"},
            "m": {2: "T"},
            "lsnr": {1: "T"},
            "coefs": {2: "T"},
        }
        
        try:
            export_impl(
                path,
                model,
                inputs=(spec, feat_erb, feat_spec),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                jit=False,
                check=check,
                simplify=simplify,
                opset_version=opset,
                print_graph=print_graph,
            )
            exported_paths["full"] = path
            logger.info(f"✓ Exported full model: {path}")
        except Exception as e:
            logger.error(f"Failed to export full model: {e}")
    
    # Export components
    enc_outputs: Optional[Tuple[Tensor, ...]] = None
    if export_components:
        # Prepare encoder input (transpose feat_spec for channel-first)
        feat_spec_enc = feat_spec.transpose(1, 4).squeeze(4)  # [B, 2, T, F]
        
        # Export encoder
        if hasattr(model, 'enc'):
            path = os.path.join(export_dir, "enc.onnx")
            input_names = ["feat_erb", "feat_spec"]
            output_names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]
            dynamic_axes = {
                "feat_erb": {2: "T"},
                "feat_spec": {2: "T"},
                "e0": {2: "T"},
                "e1": {2: "T"},
                "e2": {2: "T"},
                "e3": {2: "T"},
                "emb": {1: "T"},
                "c0": {2: "T"},
                "lsnr": {1: "T"},
            }
            
            try:
                enc_outputs = export_impl(
                    path,
                    model.enc,
                    inputs=(feat_erb, feat_spec_enc),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    jit=False,
                    check=check,
                    simplify=simplify,
                    opset_version=opset,
                    print_graph=print_graph,
                )
                exported_paths["enc"] = path
                e0, e1, e2, e3, emb, c0, lsnr = enc_outputs
                
                if save_test_data:
                    np.savez_compressed(
                        os.path.join(export_dir, "enc_input.npz"),
                        feat_erb=feat_erb.numpy(),
                        feat_spec=feat_spec_enc.numpy(),
                    )
                    np.savez_compressed(
                        os.path.join(export_dir, "enc_output.npz"),
                        e0=e0.numpy(), e1=e1.numpy(), e2=e2.numpy(), e3=e3.numpy(),
                        emb=emb.numpy(), c0=c0.numpy(), lsnr=lsnr.numpy(),
                    )
                logger.info(f"✓ Exported encoder: {path}")
            except Exception as e:
                logger.error(f"Failed to export encoder: {e}")
                enc_outputs = None
        
        # Export ERB decoder
        if hasattr(model, 'erb_dec') and enc_outputs is not None:
            e0, e1, e2, e3, emb, c0, lsnr = enc_outputs
            path = os.path.join(export_dir, "erb_dec.onnx")
            input_names = ["emb", "e3", "e2", "e1", "e0"]
            output_names = ["m"]
            dynamic_axes = {
                "emb": {1: "T"},
                "e3": {2: "T"},
                "e2": {2: "T"},
                "e1": {2: "T"},
                "e0": {2: "T"},
                "m": {2: "T"},
            }
            
            try:
                m_output = export_impl(
                    path,
                    model.erb_dec,
                    inputs=(emb.clone(), e3, e2, e1, e0),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    jit=False,
                    check=check,
                    simplify=simplify,
                    opset_version=opset,
                    print_graph=print_graph,
                )
                exported_paths["erb_dec"] = path
                
                if save_test_data:
                    np.savez_compressed(
                        os.path.join(export_dir, "erb_dec_input.npz"),
                        emb=emb.numpy(), e3=e3.numpy(), e2=e2.numpy(),
                        e1=e1.numpy(), e0=e0.numpy(),
                    )
                    np.savez_compressed(
                        os.path.join(export_dir, "erb_dec_output.npz"),
                        m=m_output[0].numpy(),
                    )
                logger.info(f"✓ Exported ERB decoder: {path}")
            except Exception as e:
                logger.error(f"Failed to export ERB decoder: {e}")
        
        # Export DF decoder
        if hasattr(model, 'df_dec') and enc_outputs is not None:
            e0, e1, e2, e3, emb, c0, lsnr = enc_outputs
            path = os.path.join(export_dir, "df_dec.onnx")
            input_names = ["emb", "c0"]
            output_names = ["coefs"]
            dynamic_axes = {
                "emb": {1: "T"},
                "c0": {2: "T"},
                "coefs": {2: "T"},
            }
            
            try:
                coefs_output = export_impl(
                    path,
                    model.df_dec,
                    inputs=(emb.clone(), c0),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    jit=False,
                    check=check,
                    simplify=simplify,
                    opset_version=opset,
                    print_graph=print_graph,
                )
                exported_paths["df_dec"] = path
                
                if save_test_data:
                    np.savez_compressed(
                        os.path.join(export_dir, "df_dec_input.npz"),
                        emb=emb.numpy(), c0=c0.numpy(),
                    )
                    np.savez_compressed(
                        os.path.join(export_dir, "df_dec_output.npz"),
                        coefs=coefs_output[0].numpy(),
                    )
                logger.info(f"✓ Exported DF decoder: {path}")
            except Exception as e:
                logger.error(f"Failed to export DF decoder: {e}")
    
    return exported_paths


def create_export_archive(
    export_dir: str,
    archive_name: str,
    include_test_data: bool = False,
) -> str:
    """Create a tar.gz archive of exported ONNX models.
    
    Args:
        export_dir: Directory containing exported files
        archive_name: Name for the archive (without extension)
        include_test_data: Whether to include test npz files
        
    Returns:
        Path to created archive
    """
    export_path = Path(export_dir)
    archive_path = export_path / f"{archive_name}_onnx.tar.gz"
    
    with tarfile.open(archive_path, mode="w:gz") as tar:
        # Add ONNX files
        for onnx_file in export_path.glob("*.onnx"):
            tar.add(onnx_file, arcname=onnx_file.name)
        
        # Add config
        config_file = export_path / "config.ini"
        if config_file.exists():
            tar.add(config_file, arcname="config.ini")
        
        # Add version info
        version_file = export_path / "version.txt"
        if version_file.exists():
            tar.add(version_file, arcname="version.txt")
        
        # Optionally add test data
        if include_test_data:
            for npz_file in export_path.glob("*.npz"):
                tar.add(npz_file, arcname=npz_file.name)
    
    logger.info(f"Created archive: {archive_path}")
    return str(archive_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export DeepFilterNet4 models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--model-base-dir",
        type=str,
        default=None,
        help="Base directory containing model checkpoint and config",
    )
    parser.add_argument(
        "export_dir",
        type=str,
        help="Directory for exported ONNX models",
    )
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Export full model as single ONNX file",
    )
    parser.add_argument(
        "--no-components",
        action="store_true",
        help="Don't export individual components (enc, erb_dec, df_dec)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip ONNX validation",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX models using onnxsim",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--no-test-data",
        action="store_true",
        help="Don't save test input/output data",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create tar.gz archive of exported files",
    )
    parser.add_argument(
        "--print-graph",
        action="store_true",
        help="Print ONNX graph structure",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if not ONNX_AVAILABLE:
        print("Error: onnx and onnxruntime are required for export")
        print("Install with: pip install onnx onnxruntime")
        return 1
    
    # Initialize logging
    init_logger(level="DEBUG" if args.debug else "INFO", file=None)
    
    # Load config
    if args.config:
        config.load(args.config, allow_reload=True)
    elif args.model_base_dir:
        config_path = os.path.join(args.model_base_dir, "config.ini")
        if os.path.exists(config_path):
            config.load(config_path, allow_reload=True)
    
    # Load model
    from df.deepfilternet4 import init_model, ModelParams4
    
    p = ModelParams4()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
    )
    
    model = init_model(df_state=df_state, run_df=True, train_mask=True)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    elif args.model_base_dir:
        # Try to find checkpoint in model base dir
        for ckpt_name in ["best_checkpoint.pt", "checkpoint.pt", "model.pt"]:
            ckpt_path = os.path.join(args.model_base_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded checkpoint: {ckpt_path}")
                break
    
    # Create export directory
    os.makedirs(args.export_dir, exist_ok=True)
    
    # Copy config to export directory
    if args.config and os.path.exists(args.config):
        shutil.copy(args.config, os.path.join(args.export_dir, "config.ini"))
    elif args.model_base_dir:
        config_src = os.path.join(args.model_base_dir, "config.ini")
        if os.path.exists(config_src):
            shutil.copy(config_src, os.path.join(args.export_dir, "config.ini"))
    
    # Write version info
    version_file = os.path.join(args.export_dir, "version.txt")
    with open(version_file, "w") as f:
        f.write(f"deepfilternet4_onnx_export\n")
        f.write(f"opset_version: {args.opset}\n")
    
    # Export
    exported = export_dfnet4(
        model,
        args.export_dir,
        df_state,
        check=not args.no_check,
        simplify=args.simplify,
        opset=args.opset,
        export_full=args.full_model,
        export_components=not args.no_components,
        print_graph=args.print_graph,
        save_test_data=not args.no_test_data,
    )
    
    logger.info(f"\nExported {len(exported)} model(s):")
    for name, path in exported.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"  {name}: {path} ({size_mb:.2f} MB)")
    
    # Create archive if requested
    if args.archive:
        archive_name = "deepfilternet4"
        if args.model_base_dir:
            archive_name = Path(args.model_base_dir).name
        create_export_archive(
            args.export_dir,
            archive_name,
            include_test_data=not args.no_test_data,
        )
    
    return 0


if __name__ == "__main__":
    exit(main())
