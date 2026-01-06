import argparse
import glob
import os
import string
import sys
from pathlib import Path
from typing import List, Union

import editdistance
import numpy as np
import pandas as pd
import torch

# Add DeepFilterNet to path for whisper_adapter import
sys.path.insert(0, str(Path(__file__).parent.parent / "DeepFilterNet"))
from df.whisper_adapter import get_whisper_backend  # noqa: E402

BACKEND = None
WHISPER_OPT = None
DT = torch.float32


def load_model():
    global BACKEND, WHISPER_OPT, DT

    # Auto-selects MLX on Apple Silicon for 5-10x speedup
    BACKEND = get_whisper_backend("small")

    has_cuda = torch.cuda.is_available()
    # MLX uses unified memory, so fp16 is safe there too
    use_fp16 = has_cuda or BACKEND.backend_name == "mlx"
    DT = torch.float16 if use_fp16 else torch.float32

    if has_cuda and BACKEND.backend_name == "pytorch":
        print("Running with cuda")
    elif BACKEND.backend_name == "mlx":
        print("Running with MLX (Apple Silicon optimized)")

    WHISPER_OPT = BACKEND.create_decoding_options(
        task="transcribe", language="en", beam_size=20, fp16=use_fp16
    )


def normalize(input: str) -> List[str]:
    return input.translate(str.maketrans("", "", string.punctuation)).lower().split(" ")


def eval_wacc(args):
    load_model()
    audio_clips_list = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    transcriptions_df = pd.read_csv(
        args.transcription_file, sep="\t", names=["filename", "transcription"]
    )
    scores = []
    n_edits = 0
    n = 0
    scores = []
    for i, fpath in enumerate(audio_clips_list):
        progress = i / len(audio_clips_list) * 100  # Percent
        idx = transcriptions_df.index[transcriptions_df["filename"] == os.path.basename(fpath)]
        if len(idx) == 0:
            print(f"WARN: file not found {fpath}")
            continue
        # Use backend's load_audio and processing methods
        audio = BACKEND.load_audio(fpath)
        audio = BACKEND.pad_or_trim(audio)
        mel = BACKEND.log_mel_spectrogram(audio)
        # Convert to torch tensor with proper device/dtype for decoding
        if BACKEND.backend_name == "pytorch":
            mel = mel.to(device=BACKEND.device, dtype=DT)
        else:
            # MLX backend returns mx.array, convert to numpy for decode
            mel = np.array(mel)

        result = BACKEND.decode(mel, WHISPER_OPT)
        target = transcriptions_df["transcription"][idx].to_list()[0]
        if "<UNKNOWN" in target or "unknown" in target:
            # target may contain <UNKNOWN\>, <UNKNOWN>, or unknown
            fpath = os.path.basename(fpath)
            print(f"Target {fpath} contains the '<UNKNOWN>' token. Skipping.")
            continue
        pred = result.text
        target, pred = normalize(target), normalize(pred)
        errors = editdistance.eval(pred, target)
        wer = errors / len(target)
        scores.append({"file_name": os.path.basename(fpath), "wacc": 1 - wer})
        n_edits += errors
        n += len(target)
        print("Progress {:2.1f} % | Cur WER: {:.1f} %".format(progress, wer * 100))
        if wer >= 1:
            print(fpath)
            print("target: '{}'".format(" ".join(target)))
            print("prediction: '{}'".format(" ".join(pred)))
    df = pd.DataFrame(scores)
    print("WER:", n_edits / n)

    df = pd.DataFrame(scores)
    print("Mean WAcc for the files is ", np.mean(df["wacc"]))

    if args.csv_file:
        df.to_csv(args.csv_file)


def print_csv(df: Union[pd.DataFrame, str]):
    if isinstance(df, str):
        df = pd.read_csv(df)
    print(df.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")
    mn_parser = subparsers.add_parser("mean", aliases=["m"])
    mn_parser.add_argument("csv_file", type=str)
    eval_parser = subparsers.add_parser("eval", aliases=["e"])
    eval_parser.add_argument(
        "testset_dir", help="Path to the dir containing audio clips to be evaluated"
    )
    eval_parser.add_argument("transcription_file", help="Path to transcription tsv file")
    eval_parser.add_argument(
        "-o", "--csv-file", help="If you want the scores in a CSV file provide the full path"
    )

    args = parser.parse_args()
    if args.subparser_name is None:
        parser.print_help()
        exit(1)
    if args.subparser_name in ("m", "mean"):
        print_csv(args.csv_file)
    else:
        eval_wacc(args)
