import time
import psutil
import numpy as np

def measure_latency(fn, runs=20):
    times = []
    for _ in range(runs):
        start = time.time()
        fn()
        end = time.time()
        times.append((end - start) * 1000)  # ms
    return np.mean(times), np.percentile(times, 95)

def get_cpu_memory_mb():
    process = psutil.Process()
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)

def get_gpu_util():
    try:
        import subprocess
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        )
        return int(result.decode().strip())
    except:
        return None
