"""
utils/quantize

CLI pipeline for downloading a Hugging Face model in safetensors format,
converting it to GGUF with llama.cpp's convert_hf_to_gguf.py, quantizing
to Q4_K_M with llama.cpp's quantize binary, and recording a provenance
entry (SHA-256 hashes of source and output) in models/provenance.json.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

from huggingface_hub import snapshot_download

from utils.config import load_config
from utils.logger import setup_logger

logger = setup_logger()

_QUANTIZATION = "Q4_K_M"
_CONVERT_SCRIPT = "convert_hf_to_gguf.py"
_QUANTIZE_BINARY_NAMES = ["llama-quantize", "quantize"]
_COMMON_LLAMACPP_PATHS = ["./llama.cpp", os.path.expanduser("~/llama.cpp")]

def _find_llamacpp_path(config: dict) -> Path:
    """
    Resolve the llama.cpp root directory from config, env, or common paths.

    llamacpp_path must point to the repo root — the directory that contains
    both convert_hf_to_gguf.py and the build/ subdirectory.

    Checks in order:
        1. llamacpp_path key in config
        2. LLAMACPP_PATH environment variable
        3. ./llama.cpp and ~/llama.cpp

    Args:
        config (dict): Loaded config dict from load_config()

    Returns:
        Path: Resolved llama.cpp root directory (contains convert_hf_to_gguf.py)

    Raises:
        FileNotFoundError: If no candidate directory containing convert_hf_to_gguf.py
            can be located
    """
    candidates = []

    cfg_path = config.get("llamacpp_path")
    if cfg_path:
        candidates.append(Path(cfg_path))

    env_path = os.environ.get("LLAMACPP_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(Path(p) for p in _COMMON_LLAMACPP_PATHS)

    for candidate in candidates:
        if (candidate / _CONVERT_SCRIPT).exists():
            logger.info(f"Found llama.cpp root at {candidate}")
            return candidate.resolve()

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find {_CONVERT_SCRIPT} in any of: {checked}\n"
        "Set llamacpp_path to the llama.cpp repo root in config.yaml "
        "or export LLAMACPP_PATH=/path/to/llama.cpp"
    )


def _find_quantize_binary(llamacpp_dir: Path) -> Path:
    """
    Find the quantize binary in the llama.cpp build output directory.

    Looks in {llamacpp_dir}/build/bin/ for both 'llama-quantize' (current
    CMake output name) and 'quantize' (older build name).

    Args:
        llamacpp_dir (Path): Resolved llama.cpp root directory

    Returns:
        Path: Path to the quantize executable

    Raises:
        FileNotFoundError: If neither binary variant is found under build/bin/
    """
    bin_dir = llamacpp_dir / "build" / "bin"
    for name in _QUANTIZE_BINARY_NAMES:
        candidate = bin_dir / name
        if candidate.exists():
            logger.info(f"Found quantize binary at {candidate}")
            return candidate

    raise FileNotFoundError(
        f"Could not find quantize binary ({' or '.join(_QUANTIZE_BINARY_NAMES)}) "
        f"in {bin_dir}. Run cmake --build in your llama.cpp directory first."
    )

def _sha256_file(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file.

    Args:
        path (Path): File to hash

    Returns:
        str: Lowercase hex digest
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_dir_safetensors(download_dir: Path) -> dict[str, str]:
    """
    Hash all .safetensors files in a directory.

    Args:
        download_dir (Path): Directory containing downloaded model files

    Returns:
        dict[str, str]: Mapping of filename -> SHA-256 hex digest
    """
    hashes = {}
    for sf in sorted(download_dir.glob("*.safetensors")):
        logger.info(f"Hashing {sf.name} ...")
        hashes[sf.name] = _sha256_file(sf)
    return hashes

def _update_provenance(
    provenance_path: Path,
    model_id: str,
    source_sha256: dict[str, str],
    output_sha256: str,
    output_path: str,
    quantization: str,
) -> None:
    """
    Write or update a provenance entry in models/provenance.json.

    One entry per model_id, keyed by model_id. Overwrites any previous
    entry for the same model_id so re-runs stay idempotent.

    Args:
        provenance_path (Path): Path to provenance.json
        model_id (str): Hugging Face model ID used as the dict key
        source_sha256 (dict): Mapping of safetensors filename -> SHA-256
        output_sha256 (str): SHA-256 of the final quantized GGUF
        output_path (str): Relative path to the output GGUF in models/
        quantization (str): Quantization type applied (e.g. Q4_K_M)
    """
    provenance_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if provenance_path.exists():
        with open(provenance_path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                logger.warning("provenance.json is malformed — will overwrite")

    existing[model_id] = {
        "model_id": model_id,
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "output_path": output_path,
        "quantization": quantization,
        "date": date.today().isoformat(),
    }

    with open(provenance_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Provenance written to {provenance_path}")

def _clean_model_name(model_id: str) -> str:
    """
    Derive a clean filename stem from a Hugging Face model ID.

    Strips the org prefix, preserves the repo name, and normalises
    to a format safe for filenames.

    Args:
        model_id (str): e.g. "PleIAs/Pleias-RAG-1B"

    Returns:
        str: e.g. "Pleias-RAG-1B"
    """
    return model_id.split("/")[-1]


def _download_model(model_id: str, download_dir: Path) -> None:
    """
    Download a Hugging Face model's safetensors files to a local directory.

    Tries local cache first (local_files_only=True) to avoid re-downloading
    files that huggingface_hub has already cached. Falls back to a normal
    network download if the cache is missing or incomplete.

    Args:
        model_id (str): Hugging Face model ID (e.g. "PleIAs/Pleias-RAG-1B")
        download_dir (Path): Destination directory for downloaded files
    """
    _CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
    _IGNORE = ["*.bin", "*.pt", "*.msgpack", "flax_model*", "tf_model*"]

    logger.info(f"Checking local cache for {model_id} ...")
    print(f"[1/4] Checking cache for {model_id} ...")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(download_dir),
            cache_dir=_CACHE_DIR,
            local_files_only=True,
            ignore_patterns=_IGNORE,
        )
        logger.info(f"Loaded {model_id} from local cache")
        print(f"[1/4] Using cached files for {model_id}")
    except Exception:
        logger.info(f"Cache miss for {model_id} — downloading from Hugging Face")
        print(f"[1/4] Downloading {model_id} from Hugging Face ...")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(download_dir),
            cache_dir=_CACHE_DIR,
            ignore_patterns=_IGNORE,
        )
        logger.info(f"Download complete for {model_id}")

    safetensors_files = list(download_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(
            f"No .safetensors files found after downloading {model_id}. "
            "The model may not publish safetensors weights."
        )

    logger.info(f"Resolved {len(safetensors_files)} safetensors file(s) to {download_dir}")

def _convert_to_gguf(
    llamacpp_dir: Path,
    model_dir: Path,
    f16_output_path: Path,
) -> None:
    """
    Convert a safetensors model directory to a float16 GGUF file.

    Calls llama.cpp's convert_hf_to_gguf.py with --outtype f16.

    Args:
        llamacpp_dir (Path): llama.cpp install directory containing the script
        model_dir (Path): Directory with downloaded safetensors and config files
        f16_output_path (Path): Destination path for the intermediate F16 GGUF
    """
    logger.info(f"Converting {model_dir} to GGUF (f16) ...")
    print("[2/4] Converting to GGUF (f16) ...")

    convert_script = llamacpp_dir / _CONVERT_SCRIPT

    cmd = [
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outtype", "f16",
        "--outfile", str(f16_output_path),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"convert_hf_to_gguf.py failed with exit code {result.returncode}"
        )

    if not f16_output_path.exists():
        raise FileNotFoundError(
            f"Conversion appeared to succeed but output file not found: {f16_output_path}"
        )

    logger.info(f"F16 GGUF written to {f16_output_path}")

def _quantize_gguf(
    quantize_binary: Path,
    f16_path: Path,
    output_path: Path,
    quantization: str,
) -> None:
    """
    Quantize an F16 GGUF file to the specified quantization type.

    Args:
        quantize_binary (Path): Path to the llama-quantize (or quantize) binary
        f16_path (Path): Input F16 GGUF file
        output_path (Path): Destination path for the quantized GGUF
        quantization (str): Quantization type (e.g. "Q4_K_M")
    """
    logger.info(f"Quantizing {f16_path.name} to {quantization} ...")
    print(f"[3/4] Quantizing to {quantization} ...")

    cmd = [
        str(quantize_binary),
        str(f16_path),
        str(output_path),
        quantization,
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"quantize binary failed with exit code {result.returncode}"
        )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Quantization appeared to succeed but output file not found: {output_path}"
        )

    logger.info(f"Quantized GGUF written to {output_path}")

def run(model_id: str) -> int:
    """
    Execute the full download → convert → quantize → provenance pipeline.

    Args:
        model_id (str): Hugging Face model ID (e.g. "PleIAs/Pleias-RAG-1B")

    Returns:
        int: 0 on success, 1 on failure
    """
    config = load_config()
    models_dir = Path(config.get("models_path", "./models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    provenance_path = models_dir / "provenance.json"
    model_name = _clean_model_name(model_id)
    output_filename = f"{model_name}-{_QUANTIZATION}.gguf"
    output_path = models_dir / output_filename

    try:
        llamacpp_dir = _find_llamacpp_path(config)
        quantize_binary = _find_quantize_binary(llamacpp_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print(f"[ERROR] {exc}")
        return 1

    with tempfile.TemporaryDirectory(prefix="privet_quantize_") as tmp:
        tmp_path = Path(tmp)
        download_dir = tmp_path / "model"
        download_dir.mkdir()
        f16_gguf = tmp_path / f"{model_name}-f16.gguf"

        try:
            _download_model(model_id, download_dir)

            source_hashes = _sha256_dir_safetensors(download_dir)
            if not source_hashes:
                raise FileNotFoundError("No .safetensors files found to hash")

            _convert_to_gguf(llamacpp_dir, download_dir, f16_gguf)
            _quantize_gguf(quantize_binary, f16_gguf, output_path, _QUANTIZATION)

        except (FileNotFoundError, RuntimeError) as exc:
            logger.error(str(exc))
            print(f"[ERROR] {exc}")
            return 1

    print("[4/4] Computing output hash and writing provenance ...")
    output_sha256 = _sha256_file(output_path)

    _update_provenance(
        provenance_path=provenance_path,
        model_id=model_id,
        source_sha256=source_hashes,
        output_sha256=output_sha256,
        output_path=str(output_path),
        quantization=_QUANTIZATION,
    )

    print()
    print(f"Done. Output: {output_path}")
    print(f"  Output SHA-256 : {output_sha256[:16]}...")
    print(f"  Provenance log : {provenance_path}")
    logger.info(f"Quantization pipeline complete: {output_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m utils.quantize <hf_model_id>")
        print("Example: python3 -m utils.quantize PleIAs/Pleias-RAG-1B")
        sys.exit(1)

    sys.exit(run(sys.argv[1]))
