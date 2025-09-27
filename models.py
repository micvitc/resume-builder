from pydantic import BaseModel
from typing import List

class ResumeRequest(BaseModel):
    resume_text: str
    job_description: str
    ollama_model: str = "llama3:instruct"  

class ResumeResponse(BaseModel):
    ats_score: float
    recommendations: List[str]
