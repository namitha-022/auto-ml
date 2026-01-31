from fastapi import FastAPI, UploadFile, File, Form
from mcp_server.context import create_model_context
from profiler import profile_model

@app.post("/run-profile")
def run_profile(model_context: dict):
    results = profile_model(model_context)
    return results

app = FastAPI()

@app.post("/upload-model")
async def upload_model(
    file: UploadFile = File(...),
    input_shape: str = Form(...)
):
    model_context = create_model_context(file, input_shape)
    return model_context
