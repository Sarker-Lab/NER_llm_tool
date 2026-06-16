from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from .models import EntitySpan
from .text_utils import normalize_text


class ExtractionCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    spans_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def key(model: str, llm_url: str, prompt: str, labels: list[str], text: str) -> str:
        payload = {
            "model": model,
            "llm_url": llm_url,
            "prompt": prompt,
            "labels": labels,
            "text": normalize_text(text),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, cache_key: str, text: str) -> list[EntitySpan] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT spans_json FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None

        items = json.loads(row[0])
        spans: list[EntitySpan] = []
        for item in items:
            start = int(item["start"])
            end = int(item["end"])
            spans.append(
                EntitySpan(
                    start=start,
                    end=end,
                    label=str(item["label"]),
                    text=text[start:end],
                )
            )
        return spans

    def set(self, cache_key: str, spans: list[EntitySpan]) -> None:
        stripped = [{"start": span.start, "end": span.end, "label": span.label} for span in spans]
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries (cache_key, spans_json, created_at)
                VALUES (?, ?, ?)
                """,
                (cache_key, json.dumps(stripped, ensure_ascii=False), time.time()),
            )

