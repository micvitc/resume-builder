# Resume Parser API with FastAPI, Ollama (Mistral), and Supabase

This repository contains a containerized FastAPI application that processes uploaded resume files (PDF or DOCX), uses a local Large Language Model (Mistral via Ollama) to extract structured data, and stores the results in a Supabase database.

## Getting Started

Follow these steps to set up and run the complete application stack on your local machine using Docker Compose.

### Prerequisites

You must have the following software installed:
1. **Git:** For cloning the repository.
2. **Docker Desktop:** Includes Docker Engine and Docker Compose.

### 1. Clone the Repository

Clone the project to your local machine:

```bash
git clone https://github.com/SSanjay0614/resume-builder.git
cd resume-builder
```

### 2. Configure Environment Variables

Update the file `.env` in the root directory of the project to securely pass your credentials to the Docker containers.

`.env` file content:

```plaintext
# Supabase Credentials (Required for database storage)
SUPABASE_URL=YOUR_SUPABASE_PROJECT_URL
SUPABASE_KEY=YOUR_SUPABASE_ANON_KEY
```

**Important:** Replace the placeholder values with your actual Supabase Project URL and Anon Key. 

### 3. Run the Application with Docker Compose

The `docker-compose.yml` file defines two services: `ollama` and `api`. The `api` service automatically pulls the necessary `mistral` model before starting.

Execute the following command to build the Docker image and start both services in detached mode:

```bash
docker compose up --build -d
```

### 4. Monitor Startup (Wait for Model Download)

The first run requires the **Mistral model** (approx. 4.1GB) to be downloaded into the `ollama` container. The Uvicorn server in the `api` container will **wait** until this pull is complete.

Monitor the startup process to ensure the API is fully ready:

```bash
docker compose logs -f api
```

Wait until you see the Uvicorn success message:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## API Usage

Once the server is running, you can interact with your API.

### 1. Access the Swagger UI

Open your web browser and navigate to the interactive API documentation:
http://localhost:8000/docs

### 2. Test the Endpoint

* Find the `POST /parse-resume` endpoint.
* Click **"Try it out"**.
* Upload a sample `.pdf` or `.docx` resume file.
* Click **"Execute"**.

The API will return the structured JSON data extracted by Mistral and saved to your Supabase database.

## Stopping the Services

To stop and clean up all resources created by Docker Compose, run:

```bash
docker compose down
```

If you wish to remove the downloaded Mistral model (to free up disk space), use the `--volumes` flag:

```bash
docker compose down --volumes
```


