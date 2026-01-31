from fastapi import FastAPI, UploadFile, File, Form
import torch
import tempfile
import os

from mcp_server.context import ModelContext
from mcp_server.converter import convert_pytorch_to_onnx


app = FastAPI()

@app.post("/upload-model")
async def upload_model(
    model: UploadFile = File(...),
    input_shape: str = Form(...)
):
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

    return {
        "model_name": model_name,
        "params": f"{param_count / 1e6:.2f}M",
        "input_shape": input_shape_list,
        "onnx_export": success,
        "onnx_message": message
    }
