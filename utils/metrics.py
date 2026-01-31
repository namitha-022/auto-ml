import time
import psutil
import torch

def measure_latency(fn, runs=20):
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        timings.append((end - start) * 1000)  # ms
    timings.sort()
    avg = sum(timings) / len(timings)
    p95 = timings[int(0.95 * len(timings)) - 1]
    return avg, p95


def get_cpu_memory_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_memory_mb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

