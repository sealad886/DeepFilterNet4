import os
import shutil
import tarfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from df.enhance import (
    ModelParams,
    df_features,
    enhance,
    get_model_basedir,
    init_df,
    setup_df_argument_parser,
)
from df.io import get_test_sample, save_audio
from df.scripts.export_onnx import export_impl
from libdf import DF


@torch.no_grad()
def export(
    model,
    export_dir: str,
    df_state: DF,
    check: bool = True,
    simplify: bool = True,
    opset=14,
    export_full: bool = False,
    print_graph: bool = False,
):
    model = deepcopy(model).to("cpu")
    model.eval()
    p = ModelParams()
    audio = torch.randn((1, 1 * p.sr))
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")

    # Export full model
    if export_full:
        path = os.path.join(export_dir, "deepfilternet2.onnx")
        input_names = ["spec", "feat_erb", "feat_spec"]
        dynamic_axes = {
            "spec": {2: "S"},
            "feat_erb": {2: "S"},
            "feat_spec": {2: "S"},
            "enh": {2: "S"},
            "m": {2: "S"},
            "lsnr": {1: "S"},
        }
        inputs = (spec, feat_erb, feat_spec)
        output_names = ["enh", "m", "lsnr", "coefs"]
        export_impl(
            path,
            model,
            inputs=inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            jit=False,
            check=check,
            simplify=simplify,
            opset_version=opset,
            print_graph=print_graph,
            rtol=1e-6,
            atol=1e-5,
        )

    # Export encoder
    feat_spec = feat_spec.transpose(1, 4).squeeze(4)  # re/im into channel axis
    path = os.path.join(export_dir, "enc.onnx")
    inputs = (feat_erb, feat_spec)
    input_names = ["feat_erb", "feat_spec"]
    dynamic_axes = {
        "feat_erb": {2: "S"},
        "feat_spec": {2: "S"},
        "e0": {2: "S"},
        "e1": {2: "S"},
        "e2": {2: "S"},
        "e3": {2: "S"},
        "emb": {1: "S"},
        "c0": {2: "S"},
        "lsnr": {1: "S"},
    }
    output_names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]
    e0, e1, e2, e3, emb, c0, lsnr = export_impl(
        path,
        model.enc,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=True,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
        rtol=1e-6,
        atol=1e-5,
    )
    np.savez_compressed(
        os.path.join(export_dir, "enc_input.npz"),
        feat_erb=feat_erb.numpy(),
        feat_spec=feat_spec.numpy(),
    )
    np.savez_compressed(
        os.path.join(export_dir, "enc_output.npz"),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
        emb=emb.numpy(),
        c0=c0.numpy(),
        lsnr=lsnr.numpy(),
    )

    # Export erb decoder
    np.savez_compressed(
        os.path.join(export_dir, "erb_dec_input.npz"),
        emb=emb.numpy(),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
    )
    inputs = (emb.clone(), e3, e2, e1, e0)
    input_names = ["emb", "e3", "e2", "e1", "e0"]
    output_names = ["m"]
    dynamic_axes = {
        "emb": {1: "S"},
        "e3": {2: "S"},
        "e2": {2: "S"},
        "e1": {2: "S"},
        "e0": {2: "S"},
        "m": {2: "S"},
    }
    path = os.path.join(export_dir, "erb_dec.onnx")
    m = export_impl(  # noqa
        path,
        model.erb_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=True,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
        rtol=1e-6,
        atol=1e-5,
    )
    np.savez_compressed(os.path.join(export_dir, "erb_dec_output.npz"), m=m[0].numpy())

    # Export df decoder
    np.savez_compressed(
        os.path.join(export_dir, "df_dec_input.npz"), emb=emb.numpy(), c0=c0.numpy()
    )
    inputs = (emb.clone(), c0)
    input_names = ["emb", "c0"]
    output_names = ["coefs"]
    dynamic_axes = {
        "emb": {1: "S"},
        "c0": {2: "S"},
        "coefs": {1: "S"},
    }
    path = os.path.join(export_dir, "df_dec.onnx")
    coefs = export_impl(  # noqa
        path,
        model.df_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=False,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
        rtol=1e-6,
        atol=1e-5,
    )
    np.savez_compressed(os.path.join(export_dir, "df_dec_output.npz"), coefs=coefs[0].numpy())


def main(args):
    try:
        import monkeytype  # noqa: F401
    except ImportError:
        print("Failed to import monkeytype. Please install it via")
        print("$ pip install MonkeyType")
        exit(1)

    print(args)
    model, df_state, _, epoch = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file="export.log",
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    sample = get_test_sample(df_state.sr())
    enhanced = enhance(model, df_state, sample, True)
    out_dir = Path("out")
    if out_dir.is_dir():
        # attempt saving enhanced audio
        save_audio(os.path.join(out_dir, "enhanced.wav"), enhanced, df_state.sr())
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export(
        model,
        export_dir,
        df_state=df_state,
        opset=args.opset,
        check=args.check,
        simplify=args.simplify,
    )
    model_base_dir = get_model_basedir(args.model_base_dir)
    if model_base_dir != args.export_dir:
        shutil.copyfile(
            os.path.join(model_base_dir, "config.ini"),
            os.path.join(args.export_dir, "config.ini"),
        )
    model_name = Path(model_base_dir).name
    version_file = os.path.join(args.export_dir, "version.txt")
    with open(version_file, "w") as f:
        f.write(f"{model_name}_epoch_{epoch}")
    tar_name = export_dir / (Path(model_base_dir).name + "_onnx.tar.gz")
    with tarfile.open(tar_name, mode="w:gz") as f:
        f.add(os.path.join(args.export_dir, "enc.onnx"))
        f.add(os.path.join(args.export_dir, "erb_dec.onnx"))
        f.add(os.path.join(args.export_dir, "df_dec.onnx"))
        f.add(os.path.join(args.export_dir, "config.ini"))
        f.add(os.path.join(args.export_dir, "version.txt"))


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument("export_dir", help="Directory for exporting the onnx model.")
    parser.add_argument(
        "--no-check",
        help="Don't check models with onnx checker.",
        action="store_false",
        dest="check",
    )
    parser.add_argument("--simplify", help="Simply onnx models using onnxsim.", action="store_true")
    parser.add_argument("--opset", help="ONNX opset version", type=int, default=12)
    args = parser.parse_args()
    main(args)
