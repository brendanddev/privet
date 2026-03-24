
"""
utils/hardware.py 

Detects RAM, CPU, GPU (Metal/CUDA/ROCm), and platform at startup.
Classifies the machine into a hardware tier and recommends optimal
config values for that tier. All detection is local — no external calls.
"""

import platform
import subprocess
import psutil
from dataclasses import dataclass, field, asdict
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger()


# Data classes

@dataclass
class GPUInfo:
    """Information about an available GPU / accelerator."""
    backend: str
    name: str = "Unknown"
    vram_gb: Optional[float] = None
    available: bool = False


@dataclass
class HardwareProfile:
    """
    Complete hardware snapshot for the current machine.

    All memory values are in GB. Times are in seconds.
    """
    # Platform
    os: str = ""
    arch: str = ""
    cpu_brand: str = ""
    is_apple_silicon: bool = False

    # Memory
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    used_ram_gb: float = 0.0
    ram_percent: float = 0.0

    # CPU
    physical_cores: int = 0
    logical_cores: int = 0
    cpu_freq_mhz: Optional[float] = None

    # GPU
    gpu: GPUInfo = field(default_factory=lambda: GPUInfo(backend="none"))

    # Tier
    tier: str = "standard" 
    tier_label: str = ""

    # Recommendations (config overrides for this tier)
    recommended_n_ctx: int = 4096
    recommended_chunk_size: int = 256
    recommended_top_k: int = 5
    recommended_llm_model: str = "gemma3:1b"
    recommended_embed_model: str = "nomic-embed-text"
    recommended_n_gpu_layers: int = -1
    unload_embed_after_index: bool = False
    notes: list = field(default_factory=list)


# Tier configuration
TIER_CONFIG = {
    "high": {
        "label": "High (16GB+)",
        "n_ctx": 8192,
        "chunk_size": 512,
        "top_k": 5,
        "llm_model": "gemma3:4b",
        "embed_model": "nomic-embed-text",
        "n_gpu_layers": -1,
        "unload_embed": False,
        "notes": [
            "Running in high-performance mode.",
            "gemma3:4b recommended — larger context window, better reasoning.",
            "similarity_top_k: 5 for improved cross-document retrieval.",
        ]
    },
    "standard": {
        "label": "Standard (8-15GB)",
        "n_ctx": 4096,
        "chunk_size": 256,
        "top_k": 3,
        "llm_model": "gemma3:1b",
        "embed_model": "nomic-embed-text",
        "n_gpu_layers": -1,
        "unload_embed": False,
        "notes": [
            "Running in standard mode — your current default config.",
            "gemma3:1b is well-suited to this memory budget.",
        ]
    },
    "low": {
        "label": "Low (4-7GB)",
        "n_ctx": 1024,
        "chunk_size": 128,
        "top_k": 3,
        "llm_model": "gemma3:1b",
        "embed_model": "nomic-embed-text",
        "n_gpu_layers": -1,
        "unload_embed": True,
        "notes": [
            "Low RAM mode — conservative settings applied.",
            "n_ctx reduced to 1024 to save ~600MB of model memory.",
            "Embedding model will unload after indexing to free ~274MB.",
            "chunk_size reduced to 128 for more precise retrieval.",
        ]
    },
    "minimal": {
        "label": "Minimal (<4GB)",
        "n_ctx": 512,
        "chunk_size": 128,
        "top_k": 2,
        "llm_model": "gemma3:1b",
        "embed_model": "nomic-embed-text",
        "n_gpu_layers": 0,
        "unload_embed": True,
        "notes": [
            "Minimal RAM mode — CPU-only inference.",
            "n_gpu_layers set to 0 — not enough RAM to safely use GPU layers.",
            "Embed model unloads after indexing.",
            "Expect slower responses — hardware is at the limit.",
            "Consider Q4_0 quantization for smaller model footprint.",
        ]
    }
}


# Profiler
class HardwareProfiler:
    """
    Detects machine hardware and produces a HardwareProfile with
    tier classification and config recommendations.

    All detection runs locally with no external calls.
    """

    def __init__(self):
        self._profile: Optional[HardwareProfile] = None
        logger.info("HardwareProfiler initialized")

    def profile(self) -> HardwareProfile:
        """
        Run full hardware detection and return a HardwareProfile.

        Result is cached, subsequent calls return the same object.
        Call refresh() to re-detect (useful for live RAM monitoring).

        Returns:
            HardwareProfile: Populated profile for this machine.
        """
        if self._profile is None:
            self._profile = self._detect()
        return self._profile

    def refresh_ram(self) -> HardwareProfile:
        """
        Re-sample live RAM stats without re-running full detection.

        Cheap to call, safe to use every few seconds for a live monitor.

        Returns:
            HardwareProfile: Updated profile with current RAM values.
        """
        if self._profile is None:
            return self.profile()

        mem = psutil.virtual_memory()
        self._profile.total_ram_gb = round(mem.total / 1e9, 2)
        self._profile.available_ram_gb = round(mem.available / 1e9, 2)
        self._profile.used_ram_gb = round(mem.used / 1e9, 2)
        self._profile.ram_percent = mem.percent
        return self._profile

    def get_tier(self) -> str:
        """Return the hardware tier string: 'high' | 'standard' | 'low' | 'minimal'."""
        return self.profile().tier

    def get_recommendations(self) -> dict:
        """
        Return a dict of config keys that should be overridden for this tier.

        These can be applied directly on top of the values loaded from config.yaml.

        Returns:
            dict: Config key -> recommended value
        """
        p = self.profile()
        return {
            "n_ctx": p.recommended_n_ctx,
            "chunk_size": p.recommended_chunk_size,
            "similarity_top_k": p.recommended_top_k,
            "llm_model": p.recommended_llm_model,
            "embed_model": p.recommended_embed_model,
            "n_gpu_layers": p.recommended_n_gpu_layers,
            "unload_embed_after_index": p.unload_embed_after_index,
        }

    def as_dict(self) -> dict:
        """Return the profile as a plain dict for logging or JSON serialization."""
        p = self.profile()
        d = asdict(p)
        return d

    def summary_lines(self) -> list[str]:
        """
        Return a human readable list of strings summarising the profile.

        Suitable for printing to a CLI or rendering line-by-line in a UI.
        """
        p = self.profile()
        lines = [
            f"OS:            {p.os} ({p.arch})",
            f"CPU:           {p.cpu_brand}",
            f"Cores:         {p.physical_cores} physical / {p.logical_cores} logical",
            f"RAM (total):   {p.total_ram_gb} GB",
            f"RAM (used):    {p.used_ram_gb} GB ({p.ram_percent}%)",
            f"RAM (free):    {p.available_ram_gb} GB",
            f"GPU backend:   {p.gpu.backend.upper()}",
            f"GPU name:      {p.gpu.name}",
            f"Apple Silicon: {'Yes' if p.is_apple_silicon else 'No'}",
            f"Tier:          {p.tier_label}",
            "---",
            f"Recommended n_ctx:       {p.recommended_n_ctx}",
            f"Recommended chunk_size:  {p.recommended_chunk_size}",
            f"Recommended top_k:       {p.recommended_top_k}",
            f"Recommended LLM:         {p.recommended_llm_model}",
            f"Unload embed after idx:  {p.unload_embed_after_index}",
        ]
        if p.notes:
            lines.append("---")
            for note in p.notes:
                lines.append(f"  • {note}")
        return lines

    def _detect(self) -> HardwareProfile:
        """Run all detection routines and assemble a HardwareProfile."""
        logger.info("Running hardware detection")
        p = HardwareProfile()

        self._detect_platform(p)
        self._detect_ram(p)
        self._detect_cpu(p)
        self._detect_gpu(p)
        self._classify_tier(p)
        self._apply_tier_recommendations(p)

        logger.info(
            f"Hardware detected | Tier: {p.tier} | RAM: {p.total_ram_gb}GB | "
            f"GPU: {p.gpu.backend} | Apple Silicon: {p.is_apple_silicon}"
        )
        return p

    def _detect_platform(self, p: HardwareProfile):
        """Detect OS and architecture."""
        p.os = platform.system()
        p.arch = platform.machine()
        p.is_apple_silicon = (
            p.os == "Darwin" and p.arch == "arm64"
        )
        logger.info(f"Platform: {p.os} {p.arch} | Apple Silicon: {p.is_apple_silicon}")

    def _detect_ram(self, p: HardwareProfile):
        """Sample current RAM stats via psutil."""
        mem = psutil.virtual_memory()
        p.total_ram_gb = round(mem.total / 1e9, 2)
        p.available_ram_gb = round(mem.available / 1e9, 2)
        p.used_ram_gb = round(mem.used / 1e9, 2)
        p.ram_percent = mem.percent
        logger.info(f"RAM: {p.total_ram_gb}GB total | {p.available_ram_gb}GB free")

    def _detect_cpu(self, p: HardwareProfile):
        """Detect CPU core counts, frequency, and brand string."""
        p.physical_cores = psutil.cpu_count(logical=False) or 0
        p.logical_cores = psutil.cpu_count(logical=True) or 0

        freq = psutil.cpu_freq()
        if freq:
            p.cpu_freq_mhz = round(freq.current, 1)

        p.cpu_brand = self._get_cpu_brand()
        logger.info(f"CPU: {p.cpu_brand} | {p.physical_cores}P/{p.logical_cores}L cores")

    def _get_cpu_brand(self) -> str:
        """
        Extract a human-readable CPU brand string.

        Platform specific: macOS uses sysctl, Linux reads /proc/cpuinfo,
        Windows uses the registry via platform module.
        """
        system = platform.system()
        try:
            if system == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=2
                )
                brand = result.stdout.strip()
                if brand:
                    return brand
                # Apple Silicon doesn't expose brand_string the same way
                result2 = subprocess.run(
                    ["sysctl", "-n", "hw.model"],
                    capture_output=True, text=True, timeout=2
                )
                return result2.stdout.strip() or "Apple Silicon"

            elif system == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()

            elif system == "Windows":
                return platform.processor()

        except Exception as e:
            logger.warning(f"CPU brand detection failed: {e}")

        return platform.processor() or "Unknown CPU"

    def _detect_gpu(self, p: HardwareProfile):
        """
        Detect available GPU acceleration backend.

        Checks for Metal (macOS Apple Silicon), CUDA (NVIDIA), and
        ROCm (AMD). Falls back to "none" if nothing is detected.
        """
        gpu = GPUInfo(backend="none", name="None detected", available=False)

        # Metal (Apple Silicon)
        if p.is_apple_silicon:
            gpu = self._detect_metal()
            if gpu.available:
                p.gpu = gpu
                return

        # CUDA (Nvidia)
        cuda_gpu = self._detect_cuda()
        if cuda_gpu.available:
            p.gpu = cuda_gpu
            return

        # ROCm (amd)
        rocm_gpu = self._detect_rocm()
        if rocm_gpu.available:
            p.gpu = rocm_gpu
            return

        p.gpu = gpu
        logger.info("No GPU acceleration detected — CPU inference only")

    def _detect_metal(self) -> GPUInfo:
        """Detect Apple Metal via system_profiler on macOS."""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=5
            )
            import json
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                gpu_name = displays[0].get("sppci_model", "Apple GPU")
                vram_str = displays[0].get("spdisplays_vram", "")
                # Apple Silicon has unified memory so VRAM isn't separately reported
                return GPUInfo(
                    backend="metal",
                    name=gpu_name,
                    vram_gb=None,
                    available=True
                )
        except Exception as e:
            logger.warning(f"Metal detection failed: {e}")

        # Fallback: Apple Silicon always has Metal even if system_profiler fails
        return GPUInfo(backend="metal", name="Apple GPU (Metal)", available=True)

    def _detect_cuda(self) -> GPUInfo:
        """Detect NVIDIA CUDA via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]
                parts = line.split(",")
                name = parts[0].strip()
                vram_mb = float(parts[1].strip()) if len(parts) > 1 else None
                vram_gb = round(vram_mb / 1024, 1) if vram_mb else None
                logger.info(f"CUDA GPU detected: {name} | VRAM: {vram_gb}GB")
                return GPUInfo(backend="cuda", name=name, vram_gb=vram_gb, available=True)
        except FileNotFoundError:
            pass   # nvidia-smi not installed
        except Exception as e:
            logger.warning(f"CUDA detection failed: {e}")

        return GPUInfo(backend="cuda", available=False)

    def _detect_rocm(self) -> GPUInfo:
        """Detect AMD ROCm via rocm-smi."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                name = "AMD GPU (ROCm)"
                for line in result.stdout.splitlines():
                    if "GPU" in line and ":" in line:
                        name = line.split(":")[-1].strip()
                        break
                logger.info(f"ROCm GPU detected: {name}")
                return GPUInfo(backend="rocm", name=name, available=True)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"ROCm detection failed: {e}")

        return GPUInfo(backend="rocm", available=False)

    def _classify_tier(self, p: HardwareProfile):
        """
        Assign a hardware tier based on total RAM.

        Thresholds:
            high:     16GB+
            standard: 8-15GB
            low:      4-7GB
            minimal:  <4GB
        """
        gb = p.total_ram_gb
        if gb >= 16:
            p.tier = "high"
        elif gb >= 8:
            p.tier = "standard"
        elif gb >= 4:
            p.tier = "low"
        else:
            p.tier = "minimal"

        p.tier_label = TIER_CONFIG[p.tier]["label"]
        logger.info(f"Hardware tier: {p.tier} ({gb}GB RAM)")

    def _apply_tier_recommendations(self, p: HardwareProfile):
        """
        Populate recommendation fields from the tier config table.

        Notes are assembled in order: tier notes first, then any
        hardware-specific additions (e.g. Metal detected, CPU-only warning).
        """
        cfg = TIER_CONFIG[p.tier]
        p.recommended_n_ctx = cfg["n_ctx"]
        p.recommended_chunk_size = cfg["chunk_size"]
        p.recommended_top_k = cfg["top_k"]
        p.recommended_llm_model = cfg["llm_model"]
        p.recommended_embed_model = cfg["embed_model"]
        p.recommended_n_gpu_layers = cfg["n_gpu_layers"]
        p.unload_embed_after_index = cfg["unload_embed"]
        p.notes = list(cfg["notes"])

        if p.gpu.available and p.gpu.backend == "metal":
            p.notes.append(
                f"Apple Metal detected ({p.gpu.name}) — GPU acceleration active via llama.cpp."
            )
        elif p.gpu.available and p.gpu.backend == "cuda":
            vram_str = f"{p.gpu.vram_gb}GB VRAM" if p.gpu.vram_gb else "VRAM unknown"
            p.notes.append(
                f"NVIDIA CUDA detected ({p.gpu.name}, {vram_str}) — n_gpu_layers=-1 for full GPU offload."
            )
        elif p.gpu.available and p.gpu.backend == "rocm":
            p.notes.append(
                f"AMD ROCm detected ({p.gpu.name}) — GPU acceleration active."
            )
        else:
            if p.tier != "minimal":
                p.notes.append(
                    "No GPU acceleration detected — running CPU inference."
                )

        if p.available_ram_gb < 1.5:
            p.notes.append(
                f"WARNING: Only {p.available_ram_gb}GB RAM currently free. "
                "Close other applications before loading models."
            )


# CLI entry point
if __name__ == "__main__":
    profiler = HardwareProfiler()
    print("\n" + "=" * 55)
    print("  LOCAL RAG ASSISTANT — Hardware Profile")
    print("=" * 55)
    for line in profiler.summary_lines():
        print(f"  {line}")
    print("=" * 55 + "\n")


    