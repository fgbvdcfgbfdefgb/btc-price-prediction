import os
import subprocess
import torch
import psutil

def detect_system_capabilities():
    """
    Detects hardware resources, GPU count, and NVLink interconnect status.
    Determines the optimal training execution strategy:
    1. 'NVLINK_MULTI_GPU': High-bandwidth NVLink multi-GPU unified training.
    2. 'PARALLEL_GPU_INSTANCES': Parallel evolutionary workers across discrete GPUs (no NVLink).
    3. 'PARALLEL_CPU_INSTANCES': Multi-process parallel evolutionary workers across CPU cores.
    """
    system_info = {
        "cpu_count_logical": os.cpu_count() or 1,
        "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_devices": [],
        "nvlink_available": False,
        "nvlink_details": [],
        "strategy": "PARALLEL_CPU_INSTANCES",
        "worker_count": max(1, (os.cpu_count() or 2) - 1)
    }

    if system_info["cuda_available"] and system_info["gpu_count"] > 0:
        for i in range(system_info["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            system_info["gpu_devices"].append({
                "device_id": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}"
            })

        # Check NVLink between GPUs if >= 2 GPUs
        if system_info["gpu_count"] >= 2:
            nvlink_found = False
            # Check 1: Peer Access matrix
            p2p_matrix = {}
            for i in range(system_info["gpu_count"]):
                for j in range(system_info["gpu_count"]):
                    if i != j:
                        can_p2p = torch.cuda.can_device_access_peer(i, j)
                        p2p_matrix[f"GPU_{i}->GPU_{j}"] = can_p2p

            # Check 2: nvidia-smi nvlink command
            try:
                res = subprocess.run(
                    ["nvidia-smi", "nvlink", "-s"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                if res.returncode == 0 and "Link" in res.stdout and "Active" in res.stdout:
                    nvlink_found = True
                    system_info["nvlink_details"].append("nvidia-smi detected active NVLink channels.")
            except Exception:
                pass

            # Check 3: nvidia-smi topo -m
            try:
                res_topo = subprocess.run(
                    ["nvidia-smi", "topo", "-m"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                if res_topo.returncode == 0 and "NV" in res_topo.stdout:
                    nvlink_found = True
                    system_info["nvlink_details"].append("nvidia-smi topo detected NVLink (NV#) interconnect.")
            except Exception:
                pass

            system_info["nvlink_available"] = nvlink_found
            system_info["p2p_matrix"] = p2p_matrix

            if nvlink_found:
                system_info["strategy"] = "NVLINK_MULTI_GPU"
                system_info["worker_count"] = system_info["gpu_count"]
            else:
                system_info["strategy"] = "PARALLEL_GPU_INSTANCES"
                system_info["worker_count"] = system_info["gpu_count"]
        else:
            # Single GPU
            system_info["strategy"] = "SINGLE_GPU_INSTANCE"
            system_info["worker_count"] = 1
    else:
        # CPU
        system_info["strategy"] = "PARALLEL_CPU_INSTANCES"
        system_info["worker_count"] = min(8, max(1, os.cpu_count() or 1))

    return system_info

if __name__ == "__main__":
    import pprint
    info = detect_system_capabilities()
    print("Detected System Configuration:")
    pprint.pprint(info)
