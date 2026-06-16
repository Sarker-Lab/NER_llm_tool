from __future__ import annotations

import csv
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import _labels_from_json, defaults
from ner_tool.cache import ExtractionCache
from ner_tool.csv_io import export_results_csv, load_csv_documents
from ner_tool.jobs import ExtractionJob
from ner_tool.llm_client import LLMConfig
from ner_tool.models import DocumentResult, EntitySpan
from ner_tool.settings import DEFAULT_PROMPT
from ner_tool.text_utils import parse_llm_entities, render_prompt


class CsvIoTests(unittest.TestCase):
    def test_load_csv_requires_text_and_generates_ids(self) -> None:
        docs = load_csv_documents(b"text,CID\nhello world,c1\n")
        self.assertEqual(docs[0].ID, "case-1")
        self.assertEqual(docs[0].text, "hello world")
        self.assertEqual(docs[0].columns["CID"], "c1")

        with self.assertRaisesRegex(ValueError, "text"):
            load_csv_documents(b"body\nhello\n")

    def test_export_results_labels_column(self) -> None:
        csv_text = export_results_csv(
            [
                DocumentResult(
                    ID="doc-1",
                    text="mild cyanosis",
                    spans=[EntitySpan(start=5, end=13, label="Cyanosis", text="cyanosis")],
                    extra={"CID": "c1"},
                )
            ]
        )
        row = next(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(row["ID"], "doc-1")
        self.assertEqual(row["CID"], "c1")
        labels = json.loads(row["labels"])
        self.assertEqual(labels[0], {"start": 5, "end": 13, "label": "Cyanosis", "text": "cyanosis"})


class TextUtilsTests(unittest.TestCase):
    def test_prompt_requires_text_placeholder(self) -> None:
        with self.assertRaisesRegex(ValueError, "text"):
            render_prompt("no placeholder", "abc", ["X"], {"ID": "1"})

        prompt = render_prompt(DEFAULT_PROMPT, "abc", ["X"], {"ID": "1"})
        self.assertIn("abc", prompt)
        self.assertIn("- X", prompt)

    def test_parse_llm_entities_uses_context_filters_and_dedupes(self) -> None:
        note = "date 1/1/2020. Procedure 1/1/2020. cyanosis."
        raw = """
        ```json
        {"entities":[
          {"label":"Procedure date","text":"1/1/2020","context":"Procedure 1/1/2020"},
          {"label":"Procedure date","text":"1/1/2020","context":"Procedure 1/1/2020"},
          {"label":"Unknown","text":"cyanosis","context":"cyanosis"}
        ]}
        ```
        """
        spans = parse_llm_entities(note, raw, ["Procedure date"])
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].start, note.rfind("1/1/2020"))
        self.assertEqual(spans[0].text, "1/1/2020")


class CacheTests(unittest.TestCase):
    def test_cache_round_trip_does_not_store_source_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.sqlite"
            cache = ExtractionCache(cache_path)
            key = cache.key("model", "url", "prompt", ["Label"], "private text")
            cache.set(key, [EntitySpan(start=8, end=12, label="Label", text="text")])

            spans = cache.get(key, "private text")
            self.assertIsNotNone(spans)
            self.assertEqual(spans[0].text, "text")

            with sqlite3.connect(cache_path) as conn:
                stored = conn.execute("SELECT spans_json FROM cache_entries").fetchone()[0]
            self.assertNotIn("private", stored)


class JobTests(unittest.TestCase):
    def test_job_uses_cache_on_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ExtractionCache(Path(tmp) / "cache.sqlite")
            docs = load_csv_documents(b"ID,text\n1,mild cyanosis\n")
            config = LLMConfig(
                model="gpt-oss-120b",
                url="http://example/v1/completions",
                max_tokens=100,
                batch_size=1,
                timeout_seconds=5,
            )

            class FakeClient:
                calls = 0

                def __init__(self, _config: LLMConfig):
                    pass

                def complete_batch(self, prompts: list[str]) -> list[str]:
                    FakeClient.calls += 1
                    assert "mild cyanosis" in prompts[0]
                    return ['{"entities":[{"label":"Cyanosis","text":"cyanosis","context":"mild cyanosis"}]}']

            with patch("ner_tool.jobs.LLMClient", FakeClient):
                first = ExtractionJob(
                    job_id="one",
                    documents=docs,
                    labels=["Cyanosis"],
                    prompt_template=DEFAULT_PROMPT,
                    config=config,
                    cache=cache,
                )
                first.run()
                self.assertEqual(first.snapshot().completed, 1)
                self.assertEqual(first.response().results[0].spans[0].text, "cyanosis")

                second = ExtractionJob(
                    job_id="two",
                    documents=docs,
                    labels=["Cyanosis"],
                    prompt_template=DEFAULT_PROMPT,
                    config=config,
                    cache=cache,
                )
                second.run()
                self.assertEqual(second.snapshot().cached, 1)
                self.assertEqual(FakeClient.calls, 1)


class ApiTests(unittest.TestCase):
    def test_defaults_and_label_validation(self) -> None:
        payload = defaults()
        self.assertEqual(payload.model, "gpt-oss-120b")
        self.assertIn("{{text}}", payload.default_prompt)
        self.assertEqual(_labels_from_json(json.dumps(["X", "X", "Y"])), ["X", "Y"])
        with self.assertRaises(HTTPException):
            _labels_from_json("not-json")


if __name__ == "__main__":
    unittest.main()
