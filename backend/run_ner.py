#!/usr/bin/env python3
"""Offline CLI for running NER extraction without the web app.

Reuses the same ner_tool package (prompt rendering, LLM client, cache,
CSV parsing) as the backend, so results match the web UI exactly.

Usage:
    python run_ner.py --input cases.csv --output results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running this script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ner_tool.cache import ExtractionCache
from ner_tool.csv_io import export_results_csv, load_csv_documents
from ner_tool.jobs import ExtractionJob
from ner_tool.llm_client import LLMConfig
from ner_tool.models import JobStatus
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NER extraction on a CSV file offline.")
    parser.add_argument("--input", required=True, help="Path to input CSV (must have a 'text' column)")
    parser.add_argument("--output", required=True, help="Path to write the results CSV")
    parser.add_argument(
        "--labels",
        help="Comma-separated labels, or path to a JSON file with a list of labels (default: built-in labels)",
    )
    parser.add_argument("--prompt", help="Path to a prompt template file containing {{text}} (default: built-in prompt)")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help=f"default: {DEFAULT_LLM_MODEL}")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL, help=f"default: {DEFAULT_LLM_URL}")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-request timeout in seconds")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="Path to the SQLite cache file (shared with the web app by default)")
    return parser.parse_args()


def load_labels(raw: str | None) -> list[str]:
    if not raw:
        return DEFAULT_LABELS
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text())
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_prompt(raw: str | None) -> str:
    if not raw:
        return DEFAULT_PROMPT
    return Path(raw).read_text()


def main() -> None:
    args = parse_args()

    labels = load_labels(args.labels)
    prompt = load_prompt(args.prompt)
    if "{{text}}" not in prompt:
        raise SystemExit('Prompt must contain the "{{text}}" placeholder')

    documents = load_csv_documents(Path(args.input).read_bytes())
    print(f"Loaded {len(documents)} document(s) from {args.input}")

    job = ExtractionJob(
        job_id="offline",
        documents=documents,
        labels=labels,
        prompt_template=prompt,
        config=LLMConfig(
            model=args.model,
            url=args.llm_url,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            timeout_seconds=args.timeout,
        ),
        cache=ExtractionCache(Path(args.cache)),
    )

    started = time.time()
    job.run()
    elapsed = time.time() - started

    results = job.response().results
    Path(args.output).write_text(export_results_csv(results))

    print(
        f"Done in {elapsed:.1f}s -- completed={job.completed} cached={job.cached} "
        f"failed={job.failed} -> wrote {args.output}"
    )
    if job.status == JobStatus.failed:
        raise SystemExit(f"Job failed: {job.message}")


if __name__ == "__main__":
    main()
