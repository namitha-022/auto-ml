from fastapi import FastAPI, UploadFile, File, Form
import torch
import tempfile
import os
import onnxruntime as ort

from mcp_server.context import ModelContext
from mcp_server.converter import convert_pytorch_to_onnx
from mcp_server.profiler import run_full_profile



app = FastAPI()

# Global state
loaded_model = None
onnx_session = None
model_input_shape = None

@app.post("/upload-model")
async def upload_model(
    model: UploadFile = File(...),
    input_shape: str = Form(...)
):
    global loaded_model, onnx_session, model_input_shape
    
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(await model.read())
        model_path = tmp.name

    # Load model safely
    loaded_model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )
    loaded_model.eval()

    model_name = loaded_model.__class__.__name__
    param_count = sum(p.numel() for p in loaded_model.parameters())
    input_shape_list = [int(x.strip()) for x in input_shape.split(",")]

    # ---- ONNX Conversion ----
    success, message, onnx_path = convert_pytorch_to_onnx(
        loaded_model,
        input_shape_list
    )

    os.remove(model_path)

    model_input_shape = input_shape_list

    onnx_session = None
    if success:
        try:
            onnx_session = ort.InferenceSession(onnx_path)
        except Exception as e:
            print(f"Warning: Failed to create ONNX session: {e}")

    return {
        "model_name": model_name,
        "params": f"{param_count / 1e6:.2f}M",
        "input_shape": input_shape_list,
        "onnx_export": success,
        "onnx_message": message
    }

@app.post("/run-profile")
async def run_profile():
    global loaded_model, onnx_session, model_input_shape

    results = run_full_profile(
        model=loaded_model,
        onnx_session=onnx_session,
        input_shape=model_input_shape,
        device="cpu"  # safe default for hackathon
    )

    return {
        "profiling_results": results
    }
