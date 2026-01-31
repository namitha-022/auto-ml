import torch
import numpy as np
from utils.metrics import (
    measure_latency,
    get_cpu_memory_mb,
    get_gpu_util
)
import onnxruntime as ort

def profile_model(model_context):
    results = []

    input_shape = model_context["input_shape"]
    model_path = model_context["model_path"]
    onnx_path = model_context.get("onnx_path")
    providers = model_context.get("execution_providers", [])

    runtimes = ["pytorch"]
    if onnx_path:
        runtimes.append("onnx")

    precisions = ["FP32", "FP16"]
    batch_sizes = [1, 4]

    for runtime in runtimes:
        for precision in precisions:
            for batch in batch_sizes:

                shape = (batch, *input_shape[1:])
                dummy_input = torch.randn(shape)

                # ----- PyTorch Runtime -----
                if runtime == "pytorch":
                    model = torch.load(model_path, map_location="cpu")
                    model.eval()

                    if precision == "FP16":
                        model = model.half()
                        dummy_input = dummy_input.half()

                    def run():
                        with torch.no_grad():
                            model(dummy_input)

                # ----- ONNX Runtime -----
                else:
                    session = ort.InferenceSession(
                        onnx_path,
                        providers=providers
                    )

                    input_name = session.get_inputs()[0].name
                    input_np = dummy_input.numpy()

                    def run():
                        session.run(None, {input_name: input_np})

                # 🔥 Warm‑up
                for _ in range(5):
                    run()

                # ⏱ Measure
                avg_latency, p95_latency = measure_latency(run)
                memory_mb = get_cpu_memory_mb()
                gpu_util = get_gpu_util()

                results.append({
                    "runtime": runtime.upper(),
                    "precision": precision,
                    "batch": batch,
                    "latency_ms": round(avg_latency, 2),
                    "p95_latency_ms": round(p95_latency, 2),
                    "memory_mb": round(memory_mb, 2),
                    "gpu_util": gpu_util
                })

    return results
