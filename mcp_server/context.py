import torch
import os
import uuid

from mcp_server.converter import (
    export_to_onnx,
    validate_onnx,
    get_onnx_session
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def create_model_context(file, input_shape):
    model_id = str(uuid.uuid4())
    model_path = f"{MODELS_DIR}/{model_id}_{file.filename}"

    # 1️⃣ Save uploaded model
    with open(model_path, "wb") as f:
        f.write(file.file.read())
    
    model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )
    model.eval()

    # 3️⃣ Extract metadata
    param_count = sum(p.numel() for p in model.parameters())
    shape = tuple(map(int, input_shape.split(",")))

    # 4️⃣ ONNX paths
    onnx_path = model_path.replace(".pt", ".onnx")

    # 5️⃣ Export to ONNX (NON-FATAL)
    onnx_exported, export_error = export_to_onnx(
        model_path,
        shape,
        onnx_path
    )

    onnx_valid = False
    execution_providers = []

    # 6️⃣ Validate + load ONNX Runtime session
    if onnx_exported:
        onnx_valid, _ = validate_onnx(onnx_path)

        if onnx_valid:
            _, execution_providers = get_onnx_session(onnx_path)

    # 7️⃣ Final MCP model context
    model_context = {
        "model_id": model_id,
        "model_name": file.filename,
        "params": f"{param_count/1e6:.2f}M",
        "input_shape": shape,

        # Phase 3 additions
        "onnx_exported": onnx_exported,
        "onnx_valid": onnx_valid,
        "onnx_path": onnx_path if onnx_exported else None,
        "execution_providers": execution_providers
    }

    return model_context
