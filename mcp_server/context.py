import torch
import os
import uuid

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def create_model_context(file, input_shape):
    model_id = str(uuid.uuid4())
    model_path = f"{MODELS_DIR}/{model_id}_{file.filename}"

    # Save uploaded model
    with open(model_path, "wb") as f:
        f.write(file.file.read())

    # Load model
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    # Extract metadata
    param_count = sum(p.numel() for p in model.parameters())

    shape = tuple(map(int, input_shape.split(",")))

    model_context = {
        "model_id": model_id,
        "model_name": file.filename,
        "params": f"{param_count/1e6:.2f}M",
        "input_shape": shape
    }

    return model_context
