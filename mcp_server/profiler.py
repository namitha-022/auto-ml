import torch
from utils.metrics import measure_latency, get_cpu_memory_mb, get_gpu_memory_mb
import onnxruntime as ort
import numpy as np

def profile_pytorch(
    model,
    input_shape,
    batch_size=1,
    precision="FP32",
    device="cpu"
):
    model = model.to(device)
    model.eval()

    if precision == "FP16":
        model = model.half()

    dummy_input = torch.randn(
        batch_size, *input_shape[1:],
        device=device
    )

    if precision == "FP16":
        dummy_input = dummy_input.half()

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # Reset GPU stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Timed run
    with torch.no_grad():
        avg_latency, p95_latency = measure_latency(
            lambda: model(dummy_input)
        )

    memory_mb = (
        get_gpu_memory_mb() if device == "cuda"
        else get_cpu_memory_mb()
    )

    return {
        "runtime": "PyTorch",
        "precision": precision,
        "batch": batch_size,
        "latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "memory_mb": round(memory_mb, 2),
        "gpu_util": None  # optional later
    }
def profile_onnx(
    session,
    input_shape,
    batch_size=1,
    precision="FP32"
):
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(
        batch_size, *input_shape[1:]
    ).astype(
        np.float16 if precision == "FP16" else np.float32
    )

    # Warm-up
    for _ in range(5):
        session.run(None, {input_name: dummy_input})

    avg_latency, p95_latency = measure_latency(
        lambda: session.run(None, {input_name: dummy_input})
    )

    memory_mb = get_cpu_memory_mb()

    return {
        "runtime": "ONNX",
        "precision": precision,
        "batch": batch_size,
        "latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "memory_mb": round(memory_mb, 2),
        "gpu_util": None
    }
def run_full_profile(
    model,
    onnx_session,
    input_shape,
    device="cpu"
):
    results = []

    for batch in [1, 4]:
        results.append(
            profile_pytorch(
                model,
                input_shape,
                batch_size=batch,
                precision="FP32",
                device=device
            )
        )

        if device == "cuda":
            results.append(
                profile_pytorch(
                    model,
                    input_shape,
                    batch_size=batch,
                    precision="FP16",
                    device=device
                )
            )

        if onnx_session is not None:
            results.append(
                profile_onnx(
                    onnx_session,
                    input_shape,
                    batch_size=batch,
                    precision="FP32"
                )
            )

    return results
