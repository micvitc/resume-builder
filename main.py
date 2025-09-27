from fastapi import FastAPI
from models import ResumeRequest, ResumeResponse   
from utils import optimize_resume_embeddings       

app = FastAPI(title="ATS Resume Optimizer API")

@app.post("/optimize_resume", response_model=ResumeResponse)
def optimize_resume(request: ResumeRequest):
    result = optimize_resume_embeddings(
        request.resume_text,
        request.job_description,
        ollama_model=request.ollama_model
    )
    return ResumeResponse(
        ats_score=result.get("ats_score", 0.0),
        recommendations=result.get("recommendations", [])
    )
