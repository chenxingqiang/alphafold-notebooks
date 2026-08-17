"""Download, validate, and inspect AlphaFold 3 model parameters."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

from . import record_io, schema
from .finetuner import WeightsComplianceError

AF3_WEIGHTS_URL = "https://storage.googleapis.com/alphafold3/af3.bin.zst"
WEIGHTS_TERMS_URL = (
    "https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md"
)

_TERMS_SUMMARY = (
    "AlphaFold 3 model parameters are for non-commercial use only, must not be "
    "redistributed, and may not be used to train competing structure-prediction models. "
    f"Full terms: {WEIGHTS_TERMS_URL}"
)


def download_weights(
    dest_dir: os.PathLike[str] | str,
    *,
    url: str = AF3_WEIGHTS_URL,
    accept_terms: bool = False,
    expected_sha256: str | None = None,
    chunk_size: int = 1 << 20,
) -> Path:
    """Download official AF3 weights after explicit terms acknowledgement."""
    if not accept_terms:
        raise WeightsComplianceError(
            f"Downloading AlphaFold 3 weights requires accept_terms=True. {_TERMS_SUMMARY}"
        )

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(url).name
    tmp = target.with_suffix(target.suffix + ".partial")

    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)

    if expected_sha256 is not None:
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"SHA256 mismatch: expected {expected_sha256}, got {digest}")

    tmp.replace(target)
    return target


def load_weights(model_dir: os.PathLike[str] | str) -> dict:
    return record_io.read_params(model_dir)


def check_weights(
    model_dir: os.PathLike[str] | str,
    schema_specs: tuple[schema.ParamSpec, ...] | None = None,
) -> schema.ValidationReport:
    params = record_io.read_params(model_dir)
    return schema.validate_params(params, schema_specs)


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        path = download_weights(args.dest, accept_terms=args.accept_terms, url=args.url)
    except WeightsComplianceError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"Downloaded weights to {path}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    report = check_weights(args.model_dir)
    print(report.describe())
    return 0 if report.ok else 1


def _cmd_info(_: argparse.Namespace) -> int:
    summary = schema.summarize()
    print("AlphaFold 3 parameter schema (metadata only)")
    print(f"  entries: {summary.num_entries:,}")
    print(f"  parameters: {summary.num_params:,}")
    print(f"  dtypes: {summary.params_by_dtype}")
    print(f"  download URL: {AF3_WEIGHTS_URL}")
    print(f"  terms: {WEIGHTS_TERMS_URL}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaFold 3 weight utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download af3.bin.zst")
    download.add_argument("dest", type=Path, help="Destination directory")
    download.add_argument("--url", default=AF3_WEIGHTS_URL)
    download.add_argument(
        "--accept-terms",
        action="store_true",
        help="Acknowledge the AlphaFold 3 Model Parameters Terms of Use",
    )
    download.set_defaults(func=_cmd_download)

    check = sub.add_parser("check", help="Validate a checkpoint against the schema")
    check.add_argument("model_dir", type=Path)
    check.set_defaults(func=_cmd_check)

    info = sub.add_parser("info", help="Show schema summary and download URL")
    info.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
