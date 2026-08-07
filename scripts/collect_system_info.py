#!/usr/bin/env python3
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command(argv):
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return f"unavailable: {error}"


output = Path(sys.argv[1] if len(sys.argv) > 1 else "results/system.json")
try:
    import torch
    torch_info = {"version": torch.__version__, "cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version()}
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            l2_bytes = getattr(
                props, "L2_cache_size", getattr(props, "l2_cache_size", None)
            )
            cuda_devices.append({
                "visible_index": index,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
                "multiprocessor_count": props.multi_processor_count,
                "l2_cache_bytes": l2_bytes,
            })
except Exception as error:
    torch_info = {"error": str(error)}
    cuda_devices = []
payload = {"python": platform.python_version(), "platform": platform.platform(),
           "torch": torch_info, "selected_gpu": os.environ.get("GPU"),
           "cuda_devices": cuda_devices,
           "nvidia_smi": command(["nvidia-smi", "-q"]),
           "nvcc": command(["nvcc", "--version"]),
           "pip_freeze": command([sys.executable, "-m", "pip", "freeze"])}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(output)
