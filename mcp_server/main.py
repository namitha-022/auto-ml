from fastapi import FastAPI, UploadFile, File, Form
import torch
import tempfile
import os

from context import ModelContext

app = FastAPI()

@app.post("/upload-model")
async def upload_model(
    model: UploadFile = File(...),
    input_shape: str = Form(...)
):
    # Save model temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(await model.read())
        model_path = tmp.name

    # Load PyTorch model
    loaded_model = torch.load(model_path, map_location="cpu")
    loaded_model.eval()

    # Model name (best-effort)
    model_name = loaded_model.__class__.__name__

    # Parameter count
    param_count = sum(p.numel() for p in loaded_model.parameters())

    # Parse input shape
    input_shape_list = [int(x.strip()) for x in input_shape.split(",")]

    context = ModelContext(
        model_name=model_name,
        param_count=param_count,
        input_shape=input_shape_list
    )

    os.remove(model_path)

    return context.to_dict()

