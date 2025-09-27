import os
import fitz  
import docx2txt
import json
import re
from ollama import chat
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
#OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are missing. Set SUPABASE_URL and SUPABASE_KEY.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
#ollama_client = Client(host=OLLAMA_HOST)

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def extract_text_from_docx(path):
    return docx2txt.process(path).strip()


def query_mistral_local(text):
    response = chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts structured information from resumes."},
            {"role": "user", "content": f"""
You are a resume parsing AI. Extract the following fields from the resume and return a JSON object:

- name
- email
- phone
- education (degree, institution, year)
- experience (title, company, duration, description)
- skills

Resume:
\"\"\"
{text}
\"\"\"
"""
             }
        ]
    )
    return response['message']['content']


def extract_json_from_output(text):
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
    
    if json_match:
        try:
            return json.loads(json_match.group(1) if json_match.lastindex else json_match.group())
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
    else:
        print("No JSON found in model output.")
    return None


def transform_parsed_data(data, filename):
    return {
        "filename": filename,
        "name": data.get("name"),
        "email": data.get("email"),
        "phone": data.get("phone"),
        "skills": json.dumps(data.get("skills", [])),
        "projects": json.dumps(data.get("projects", [])),
        "experience": json.dumps(data.get("experience", [])),
        "certification": json.dumps(data.get("certifications", [])),
        "other_info": json.dumps({
            "education": data.get("education", []),
            "publications": data.get("publications", []),
            "awards": data.get("awards", []),
            "interests": data.get("interests", []),
            "languages": data.get("languages", []),
            **{
                k: v for k, v in data.items()
                if k not in {
                    "name", "email", "phone", "skills",
                    "projects", "experience", "certifications",
                    "education", "publications", "awards",
                    "interests", "languages"
                }
            }
        })
    }


def store_in_supabase(data):
    response = supabase.table("parsed_resumes").insert(data).execute()
    return response


def parse_resume(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

    print("Parsing resume...")
    raw_output = query_mistral_local(text)
    parsed_json = extract_json_from_output(raw_output)

    if parsed_json:
        print("Parsed successfully. Storing in Supabase...")
        transformed_data = transform_parsed_data(parsed_json, os.path.basename(file_path))
        store_in_supabase(transformed_data)
        return transformed_data
    else:
        print("Failed to parse JSON.")
        return raw_output


def parse_resume_filelike(file, filename):
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file)
    elif ext == ".docx":
        text = extract_text_from_docx(file)
    else:
        raise ValueError("Unsupported file type")

    print("Parsing resume...")
    raw_output = query_mistral_local(text)
    parsed_json = extract_json_from_output(raw_output)

    if parsed_json:
        print("Parsed successfully. Storing in Supabase...")
        transformed_data = transform_parsed_data(parsed_json, filename)
        store_in_supabase(transformed_data)
        return transformed_data
    else:
        print("Failed to parse JSON.")
        return raw_output


if __name__ == "__main__":
    folder = "resumes"
    for file in os.listdir(folder):
        if file.lower().endswith((".pdf", ".docx")):
            print(f"\nProcessing: {file}")
            result = parse_resume(os.path.join(folder, file))
            print(json.dumps(result, indent=2) if isinstance(result, dict) else result)
