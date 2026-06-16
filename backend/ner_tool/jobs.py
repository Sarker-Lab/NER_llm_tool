from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .cache import ExtractionCache
from .llm_client import LLMClient, LLMConfig
from .models import DocumentResult, InputDocument, JobResultsResponse, JobStatus, JobStatusResponse
from .settings import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .text_utils import parse_llm_entities, render_prompt


@dataclass
class PendingPrompt:
    index: int
    prompt: str
    cache_key: str


@dataclass
class ExtractionJob:
    job_id: str
    documents: list[InputDocument]
    labels: list[str]
    prompt_template: str
    config: LLMConfig
    cache: ExtractionCache
    status: JobStatus = JobStatus.queued
    results: list[DocumentResult | None] = field(default_factory=list)
    completed: int = 0
    cached: int = 0
    failed: int = 0
    message: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    uncached_completed: int = 0
    uncached_seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.results = [None] * len(self.documents)

    def snapshot(self) -> JobStatusResponse:
        with self.lock:
            now = self.finished_at or time.time()
            elapsed = 0.0 if self.started_at is None else now - self.started_at
            total = len(self.documents)
            percent = 0.0 if total == 0 else round((self.completed / total) * 100, 2)
            remaining_uncached = max(total - self.completed, 0)
            eta = None
            if self.uncached_completed > 0 and remaining_uncached > 0:
                eta = (self.uncached_seconds / self.uncached_completed) * remaining_uncached
            return JobStatusResponse(
                job_id=self.job_id,
                status=self.status,
                total=total,
                completed=self.completed,
                cached=self.cached,
                failed=self.failed,
                percent=percent,
                elapsed_seconds=round(elapsed, 2),
                eta_seconds=round(eta, 2) if eta is not None else None,
                message=self.message,
            )

    def response(self) -> JobResultsResponse:
        with self.lock:
            results = [
                result
                for result in self.results
                if result is not None
            ]
            return JobResultsResponse(
                job_id=self.job_id,
                status=self.status,
                labels=self.labels,
                results=results,
            )

    def _set_result(self, index: int, result: DocumentResult, duration: float | None = None) -> None:
        with self.lock:
            self.results[index] = result
            self.completed += 1
            if result.cached:
                self.cached += 1
            if result.error:
                self.failed += 1
            if duration is not None and not result.cached:
                self.uncached_completed += 1
                self.uncached_seconds += duration

    def run(self) -> None:
        with self.lock:
            self.status = JobStatus.running
            self.started_at = time.time()
            self.message = "Running extraction"

        try:
            client = LLMClient(self.config)
            pending: list[PendingPrompt] = []
            for index, document in enumerate(self.documents):
                try:
                    prompt = render_prompt(self.prompt_template, document.text, self.labels, document.columns)
                    cache_key = self.cache.key(self.config.model, self.config.url, prompt, self.labels, document.text)
                    cached_spans = self.cache.get(cache_key, document.text)
                    if cached_spans is not None:
                        self._set_result(
                            index,
                            DocumentResult(
                                ID=document.ID,
                                text=document.text,
                                spans=cached_spans,
                                cached=True,
                                extra=document.columns,
                            ),
                        )
                    else:
                        pending.append(PendingPrompt(index=index, prompt=prompt, cache_key=cache_key))
                except Exception as exc:
                    self._set_result(
                        index,
                        DocumentResult(ID=document.ID, text=document.text, error=str(exc), extra=document.columns),
                    )

            batch_size = max(1, self.config.batch_size)
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset : offset + batch_size]
                started = time.time()
                try:
                    answers = client.complete_batch([item.prompt for item in batch])
                except Exception as exc:
                    duration = (time.time() - started) / max(len(batch), 1)
                    for item in batch:
                        document = self.documents[item.index]
                        self._set_result(
                            item.index,
                            DocumentResult(ID=document.ID, text=document.text, error=str(exc), extra=document.columns),
                            duration=duration,
                        )
                    continue

                duration = (time.time() - started) / max(len(batch), 1)
                for item, raw_output in zip(batch, answers):
                    document = self.documents[item.index]
                    try:
                        spans = parse_llm_entities(document.text, raw_output, self.labels)
                        self.cache.set(item.cache_key, spans)
                        self._set_result(
                            item.index,
                            DocumentResult(ID=document.ID, text=document.text, spans=spans, extra=document.columns),
                            duration=duration,
                        )
                    except Exception as exc:
                        self._set_result(
                            item.index,
                            DocumentResult(ID=document.ID, text=document.text, error=str(exc), extra=document.columns),
                            duration=duration,
                        )

            with self.lock:
                self.status = JobStatus.completed
                self.finished_at = time.time()
                self.message = "Extraction completed"
        except Exception as exc:
            with self.lock:
                self.status = JobStatus.failed
                self.finished_at = time.time()
                self.message = str(exc)


class JobStore:
    def __init__(self, cache: ExtractionCache):
        self.cache = cache
        self._jobs: dict[str, ExtractionJob] = {}
        self._lock = threading.Lock()

    def create(
        self,
        documents: list[InputDocument],
        labels: list[str],
        prompt_template: str,
        config_values: dict[str, Any],
    ) -> ExtractionJob:
        job_id = uuid.uuid4().hex
        config = LLMConfig(
            model=str(config_values["model"]),
            url=str(config_values["llm_url"]),
            max_tokens=int(config_values["max_tokens"]),
            batch_size=int(config_values["batch_size"]),
            timeout_seconds=float(config_values.get("timeout_seconds") or DEFAULT_REQUEST_TIMEOUT_SECONDS),
        )
        job = ExtractionJob(
            job_id=job_id,
            documents=documents,
            labels=labels,
            prompt_template=prompt_template,
            config=config,
            cache=self.cache,
        )
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=job.run, name=f"ner-job-{job_id}", daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> ExtractionJob | None:
        with self._lock:
            return self._jobs.get(job_id)
