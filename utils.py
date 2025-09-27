import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
import re
import nltk
import json
import numpy as np
import spacy
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))

# Load spaCy model
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded successfully.")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model downloaded and loaded successfully.")

# ------------------ OLLAMA CONFIGURATION ------------------ #
# Read from environment variables (useful for Docker)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct")
OLLAMA_GENERATE_ENDPOINT = f'{OLLAMA_BASE_URL}/api/generate'
OLLAMA_LIST_MODELS_ENDPOINT = f'{OLLAMA_BASE_URL}/api/tags'
OLLAMA_TIMEOUT = 180
OLLAMA_MAX_TOKENS = 200

# ------------------ OLLAMA HELPERS ------------------ #
def is_ollama_service_available(ollama_url=OLLAMA_BASE_URL):
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        return 'models' in response.json()
    except Exception as e:
        logging.error(f"Ollama service not available: {e}")
        return False

def get_available_ollama_models(ollama_url=OLLAMA_BASE_URL):
    if not is_ollama_service_available(ollama_url):
        return []
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        logging.error(f"Error retrieving Ollama models: {e}")
        return []

# ------------------ SCORING FUNCTIONS ------------------ #
def calculate_keyword_score(tfidf_matrix):
    if tfidf_matrix is None or tfidf_matrix.shape[0] < 2:
        return 0.0
    try:
        cosine_similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
        return cosine_similarity * 40
    except Exception as e:
        logging.error(f"Error calculating keyword score: {e}")
        return 0.0

def calculate_formatting_score(resume_text):
    if not resume_text:
        return 0.0
    headers = ["experience", "skills", "education", "projects", "certifications", "achievements"]
    score = 0
    for header in headers:
        if re.search(rf"\b{header}\b", resume_text, re.IGNORECASE):
            score += 30 / len(headers)
    return score

def calculate_relevance_score(resume_text, job_description, vectorizer):
    if not resume_text or not job_description or vectorizer is None:
        return 0.0
    try:
        feature_names = set(vectorizer.get_feature_names_out())
        resume_keywords = set(resume_text.lower().split())
        common_keywords = feature_names.intersection(resume_keywords)
        return (len(common_keywords) / max(len(feature_names), 1)) * 20
    except Exception as e:
        logging.error(f"Error calculating relevance score: {e}")
        return 0.0

def calculate_readability_score(resume_text):
    if not resume_text:
        return 0.0
    try:
        wc = len(resume_text.split())
        return 10 if 200 <= wc <= 600 else 5
    except Exception as e:
        logging.error(f"Error calculating readability score: {e}")
        return 0.0

def calculate_ats_score(resume_text, job_description):
    if not resume_text or not job_description:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        documents = [resume_text, job_description]
        tfidf_matrix = vectorizer.fit_transform(documents)

        return min(round(
            calculate_keyword_score(tfidf_matrix)
            + calculate_formatting_score(resume_text)
            + calculate_relevance_score(resume_text, job_description, vectorizer)
            + calculate_readability_score(resume_text),
            2), 100)
    except Exception as e:
        logging.error(f"Error during ATS score calculation: {e}")
        return 0.0

# ------------------ EMBEDDINGS ------------------ #
def generate_embeddings(text):
    if not text:
        return [], None
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        tokenizer = PunktSentenceTokenizer()
        sentences = tokenizer.tokenize(text)
        embeddings = model.encode(sentences, convert_to_tensor=True)
        return sentences, embeddings
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return [], None

def identify_missing_content_and_recommendations(resume_sentences, resume_embeddings, jd_sentences, jd_embeddings):
    if resume_embeddings is None or jd_embeddings is None:
        return []
    try:
        cosine_scores = util.cos_sim(resume_embeddings, jd_embeddings)
        missing_content = []
        for i, jd_sentence in enumerate(jd_sentences):
            max_score = cosine_scores[:, i].max().item()
            if max_score < 0.55:
                missing_content.append(jd_sentence)
        return missing_content
    except Exception as e:
        logging.error(f"Error identifying missing content: {e}")
        return []

# ------------------ OLLAMA PROMPT ------------------ #
def generate_ollama_recommendations(missing_content, ollama_model=OLLAMA_MODEL, ollama_url=OLLAMA_GENERATE_ENDPOINT):
    if not missing_content:
        prompt = """
You are an expert ATS (Applicant Tracking System) resume consultant.

The resume already matches most of the job description.
Still, provide exactly 3 improvement tips to make it stronger.

Guidelines:
- Each recommendation must be specific, actionable, and less than 20 words.
- Focus on quantifiable achievements, formatting, and keyword optimization.

Output format (strictly):
1. [recommendation]
2. [recommendation]
3. [recommendation]
"""
    else:
        limited_content = missing_content[:3]
        prompt = f"""
You are an expert ATS (Applicant Tracking System) resume consultant.

Analyze the resume against the job description.
Missing content: {' | '.join(limited_content)}

Guidelines:
- Suggest exactly 3 resume improvements.
- Each must be specific, actionable, and <20 words.
- Focus on missing skills, quantifiable achievements, and ATS readability.

Output format (strictly):
1. [recommendation]
2. [recommendation]
3. [recommendation]
"""

    if not is_ollama_service_available(OLLAMA_BASE_URL):
        return [f"Ollama not running at {OLLAMA_BASE_URL}. Start it with 'ollama serve'."]

    if ollama_model not in get_available_ollama_models(OLLAMA_BASE_URL):
        return [f"Model '{ollama_model}' not found. Run 'ollama pull {ollama_model}'."]

    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": OLLAMA_MAX_TOKENS, "num_ctx": 1024}
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        generated_text = response.json().get('response', '').strip()

        recs = [re.sub(r'^\d+\.\s', '', line).strip()
                for line in generated_text.split('\n') if re.match(r'^\d+\.\s', line)]
        return recs[:3] if recs else [
            "Add measurable results to achievements",
            "Highlight job-specific technical skills",
            "Improve resume formatting for ATS readability"
        ]
    except Exception as e:
        logging.error(f"Ollama error: {e}")
        return ["Error contacting Ollama", "Check if it's running", "Verify model availability"]

# ------------------ MAIN OPTIMIZER ------------------ #
def optimize_resume_embeddings(resume_text, job_description, ollama_model=OLLAMA_MODEL):
    if not resume_text and not job_description:
        return {"ats_score": 0.0, "recommendations": ["Provide both resume and job description."]}

    ats_score = calculate_ats_score(resume_text, job_description)
    resume_sentences, resume_embeddings = generate_embeddings(resume_text)
    jd_sentences, jd_embeddings = generate_embeddings(job_description)
    missing_content = identify_missing_content_and_recommendations(resume_sentences, resume_embeddings, jd_sentences, jd_embeddings)

    recommendations = generate_ollama_recommendations(missing_content, ollama_model=ollama_model)

    return {"ats_score": ats_score, "recommendations": recommendations}
