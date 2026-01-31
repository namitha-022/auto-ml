from fastapi import FastAPI, UploadFile, File, Form
from mcp_server.context import create_model_context
from profiler import profile_model

@app.post("/run-profile")
def run_profile(model_context: dict):
    results = profile_model(model_context)
    return results

app = FastAPI()

# In-memory storage for profiling results (for demo purposes)
PROFILE_RESULTS = {}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/upload-model")
async def upload_model(
    file: UploadFile = File(...),
    input_shape: str = Form(...)
):
    model_context = create_model_context(file, input_shape)
    return model_context


@app.post("/run-profile")
async def run_profiling(config: dict):
    """
    Run profiling for a model configuration.
    Returns a job ID for tracking.
    """
    job_id = str(uuid.uuid4())
    PROFILE_RESULTS[job_id] = {
        "status": "running",
        "config": config,
        "results": []
    }
    return {"job_id": job_id, "status": "running"}


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get profiling results for a job ID"""
    if job_id not in PROFILE_RESULTS:
        return {"error": "Job not found"}, 404
    return PROFILE_RESULTS[job_id].get("results", [])


@app.get("/recommendations/{job_id}")
async def get_recommendations(job_id: str):
    """Get bottleneck analysis and recommendations for a job ID"""
    if job_id not in PROFILE_RESULTS:
        return {"error": "Job not found"}, 404
    
    results = PROFILE_RESULTS[job_id].get("results", [])
    if not results:
        return {"error": "No results available"}, 404
    
    summary = analyze_bottleneck(results)
    details = analyze_bottlenecks(results)
    return {
        "job_id": job_id,
        "summary": summary,
        "details": details,
        "analysis_timestamp": str(uuid.uuid1())
    }


@app.post("/analyze")
async def analyze(profile_results: list):
    """Direct analysis endpoint for profiling results"""
    return analyze_bottlenecks(profile_results)
