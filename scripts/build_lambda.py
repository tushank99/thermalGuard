"""Build the Lambda deployment zip: Linux wheels for requirements-serve.txt + serving code + ONNX model.

Usage:  python scripts/build_lambda.py   ->  build/lambda.zip
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
PKG = BUILD / "lambda"


def main() -> None:
    shutil.rmtree(PKG, ignore_errors=True)
    PKG.mkdir(parents=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements-serve.txt"),
        "--target", str(PKG), "--python-version", "3.12",
        # Lambda python3.12 (AL2023, glibc 2.34) accepts all of these tags
        "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux_2_27_x86_64",
        "--platform", "manylinux2014_x86_64",
        "--implementation", "cp", "--only-binary=:all:", "--upgrade",
    ])
    # Only the serving module ships; training code stays out of the runtime.
    dst = PKG / "thermalguard"
    (dst / "serving").mkdir(parents=True)
    (dst / "__init__.py").write_text("")
    shutil.copy(ROOT / "src/thermalguard/serving/__init__.py", dst / "serving")
    shutil.copy(ROOT / "src/thermalguard/serving/app.py", dst / "serving")
    shutil.copytree(ROOT / "artifacts/lstm", PKG / "model", ignore=shutil.ignore_patterns("*.data"))
    for junk in PKG.rglob("__pycache__"):
        shutil.rmtree(junk, ignore_errors=True)
    out = shutil.make_archive(str(BUILD / "lambda"), "zip", PKG)
    print(f"built {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
