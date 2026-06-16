from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response

from ner_tool.cache import ExtractionCache
from ner_tool.csv_io import export_results_csv, load_csv_documents
from ner_tool.jobs import JobStore
from ner_tool.models import DefaultsResponse, JobResultsResponse, JobStatusResponse
from ner_tool.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_PATH,
    DEFAULT_LABELS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)


app = FastAPI(title="NER LLM Tool", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = ExtractionCache(Path(DEFAULT_CACHE_PATH))
jobs = JobStore(cache)


def _labels_from_json(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="labels must be a JSON array of strings") from exc
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="labels must be a JSON array of strings")
    labels = [str(item).strip() for item in value if str(item).strip()]
    labels = list(dict.fromkeys(labels))
    if not labels:
        raise HTTPException(status_code=400, detail="At least one label is required")
    return labels


def _get_job_or_404(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.get("/api/defaults", response_model=DefaultsResponse)
def defaults() -> DefaultsResponse:
    return DefaultsResponse(
        default_prompt=DEFAULT_PROMPT,
        default_labels=DEFAULT_LABELS,
        model=DEFAULT_LLM_MODEL,
        llm_url=DEFAULT_LLM_URL,
        max_tokens=DEFAULT_MAX_TOKENS,
        batch_size=DEFAULT_BATCH_SIZE,
    )


@app.post("/api/jobs", response_model=dict[str, str])
async def create_job(
    file: UploadFile = File(...),
    labels: str = Form(...),
    prompt: str = Form(DEFAULT_PROMPT),
    model: str = Form(DEFAULT_LLM_MODEL),
    llm_url: str = Form(DEFAULT_LLM_URL),
    max_tokens: int = Form(DEFAULT_MAX_TOKENS),
    batch_size: int = Form(DEFAULT_BATCH_SIZE),
) -> dict[str, str]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload must be a CSV file")
    if "{{text}}" not in prompt:
        raise HTTPException(status_code=400, detail='Prompt must contain the "{{text}}" placeholder')
    if max_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be positive")
    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be positive")

    try:
        documents = load_csv_documents(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = jobs.create(
        documents=documents,
        labels=_labels_from_json(labels),
        prompt_template=prompt,
        config_values={
            "model": model,
            "llm_url": llm_url,
            "max_tokens": max_tokens,
            "batch_size": batch_size,
            "timeout_seconds": DEFAULT_REQUEST_TIMEOUT_SECONDS,
        },
    )
    return {"job_id": job.job_id}


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    return _get_job_or_404(job_id).snapshot()


@app.get("/api/jobs/{job_id}/results", response_model=JobResultsResponse)
def job_results(job_id: str) -> JobResultsResponse:
    return _get_job_or_404(job_id).response()


@app.get("/api/jobs/{job_id}/export.csv")
def export_csv(job_id: str) -> Response:
    job = _get_job_or_404(job_id)
    response = job.response()
    if not response.results:
        raise HTTPException(status_code=400, detail="No job results are available")
    csv_text = export_results_csv(response.results)
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="ner_results_{job_id}.csv"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("NER_BACKEND_HOST", "0.0.0.0"),
        port=int(os.environ.get("NER_BACKEND_PORT", "5002")),
        reload=False,
    )
