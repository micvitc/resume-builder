# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first (to leverage caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app files
COPY . .

# Download NLTK and spaCy models
RUN python -m nltk.downloader stopwords punkt
RUN python -m spacy download en_core_web_sm

# Expose port 8000
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
