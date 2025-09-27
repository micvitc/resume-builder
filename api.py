from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import uuid

import resume_parser

app = FastAPI(title="Resume Parser API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/parse-resume")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[-1]
        if file_ext.lower() not in [".pdf", ".docx"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        temp_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, temp_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = resume_parser.parse_resume(file_path)

        os.remove(file_path)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
