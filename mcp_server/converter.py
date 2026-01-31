import torch
import os
import onnx
import onnx.checker

def convert_pytorch_to_onnx(
    model,
    input_shape,
    onnx_path="model.onnx"
):
    """
    Converts a PyTorch model to ONNX and validates it.
    Returns (success: bool, message: str, onnx_path: str | None)
    """

    try:
        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,          # 👈 critical
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    use_dynamo=False           # 👈 THIS IS THE KEY LINE
)


        # Validate ONNX
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        return True, "ONNX export successful", onnx_path

    except Exception as e:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

        return False, str(e), None
