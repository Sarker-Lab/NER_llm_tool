from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EntitySpan(BaseModel):
    start: int
    end: int
    label: str
    text: str = ""


class DocumentResult(BaseModel):
    ID: str
    text: str
    spans: list[EntitySpan] = Field(default_factory=list)
    cached: bool = False
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    total: int
    completed: int
    cached: int
    failed: int
    percent: float
    elapsed_seconds: float
    eta_seconds: float | None = None
    message: str | None = None


class JobResultsResponse(BaseModel):
    job_id: str
    status: JobStatus
    labels: list[str]
    results: list[DocumentResult]


class DefaultsResponse(BaseModel):
    default_prompt: str
    default_labels: list[str]
    model: str
    llm_url: str
    max_tokens: int
    batch_size: int


@dataclass
class InputDocument:
    ID: str
    text: str
    columns: dict[str, Any] = field(default_factory=dict)

