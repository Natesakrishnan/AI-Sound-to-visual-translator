"""
Hardware Detection and Acceleration Management Module
Detects CPU, RAM, discrete NVIDIA GPU, CUDA status, and Keras/PyTorch backend configuration.
"""

import os
import sys
import platform
import subprocess

# Ensure Keras backend is torch
os.environ.setdefault("KERAS_BACKEND", "torch")


def get_cpu_info() -> dict:
    """Retrieve CPU model name, core counts, and architecture."""
    cpu_name = platform.processor() or "Unknown CPU"
    cores = os.cpu_count() or 1
    logical_cores = cores

    if platform.system() == "Windows":
        try:
            cmd = "powershell -Command \"Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name\""
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                cpu_name = res.stdout.strip().split("\n")[0].strip()
        except Exception:
            pass

    return {
        "model": cpu_name,
        "logical_cores": logical_cores,
        "architecture": platform.machine()
    }


def get_ram_info() -> dict:
    """Retrieve total and approximate available system RAM in GB."""
    total_gb = 16.0
    if platform.system() == "Windows":
        try:
            cmd = "powershell -Command \"(Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1GB\""
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                total_gb = round(float(res.stdout.strip()), 2)
        except Exception:
            pass

    return {
        "total_gb": total_gb,
        "formatted": f"{total_gb:.1f} GB"
    }


def get_gpu_info() -> dict:
    """Detect available GPUs (NVIDIA RTX 3050 Laptop GPU, Intel UHD) and CUDA status."""
    gpus_detected = []
    nvidia_detected = False
    vram_str = "Unknown"

    if platform.system() == "Windows":
        try:
            cmd = "powershell -Command \"Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name\""
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                names = [n.strip() for n in res.stdout.strip().split("\n") if n.strip()]
                gpus_detected = names
                for name in names:
                    if "NVIDIA" in name.upper() or "RTX" in name.upper() or "GEFORCE" in name.upper():
                        nvidia_detected = True
                        if "6GB" in name or "3050" in name:
                            vram_str = "6 GB"
        except Exception:
            pass

    # Check PyTorch CUDA availability
    cuda_available = False
    device_name = "CPU"
    pytorch_version = "Not Installed"
    cuda_device_name = ""

    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)
            device_name = f"GPU ({cuda_device_name})"
            try:
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_str = f"{round(vram_bytes / (1024**3), 1)} GB"
            except Exception:
                pass
        else:
            device_name = "CPU (Fallback)"
    except Exception:
        pass

    primary_gpu = "None"
    if gpus_detected:
        for g in gpus_detected:
            if "NVIDIA" in g.upper():
                primary_gpu = g
                break
        if primary_gpu == "None":
            primary_gpu = gpus_detected[0]

    return {
        "all_gpus": gpus_detected,
        "primary_gpu": primary_gpu,
        "nvidia_detected": nvidia_detected,
        "vram": vram_str,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "training_device": "cuda:0" if cuda_available else "cpu",
        "device_display": device_name,
        "pytorch_version": pytorch_version
    }


def get_keras_info() -> dict:
    """Detect Keras version and backend."""
    version = "Not Loaded"
    backend = "torch"
    try:
        import keras
        version = keras.__version__
        backend = keras.config.backend()
    except Exception as e:
        version = f"Error: {e}"

    return {
        "version": version,
        "backend": backend
    }


def get_system_hardware_info() -> dict:
    """Aggregate complete hardware and ML engine configuration."""
    cpu = get_cpu_info()
    ram = get_ram_info()
    gpu = get_gpu_info()
    keras_info = get_keras_info()

    return {
        "cpu": cpu,
        "ram": ram,
        "gpu": gpu,
        "keras": keras_info,
        "recommended_batch_size": 32 if gpu["cuda_available"] else 16
    }


def get_hardware_summary_text() -> str:
    """Format hardware specs into a clean human-readable summary block."""
    info = get_system_hardware_info()
    lines = [
        "==================================================",
        " SYSTEM HARDWARE & ACCELERATION CONFIGURATION",
        "==================================================",
        f" CPU:               {info['cpu']['model']}",
        f" Logical Cores:     {info['cpu']['logical_cores']}",
        f" Total System RAM:  {info['ram']['formatted']}",
        f" System GPU(s):     {', '.join(info['gpu']['all_gpus']) if info['gpu']['all_gpus'] else 'None detected'}",
        f" Primary GPU:       {info['gpu']['primary_gpu']} (VRAM: {info['gpu']['vram']})",
        f" PyTorch Version:   {info['gpu']['pytorch_version']}",
        f" CUDA Available:    {'YES - ' + info['gpu']['cuda_device_name'] if info['gpu']['cuda_available'] else 'NO (CPU Active)'}",
        f" Keras 3 Backend:   {info['keras']['backend'].upper()} (v{info['keras']['version']})",
        f" Active DL Device:  {info['gpu']['device_display']}",
        f" Recommended Batch: {info['recommended_batch_size']}",
        "=================================================="
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_hardware_summary_text())
