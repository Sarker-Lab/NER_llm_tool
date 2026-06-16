from __future__ import annotations

import json
import re
from typing import Iterable

from .models import EntitySpan


def normalize_text(value: object) -> str:
    return str(value).replace("_x000D_", "\n").replace("\r\n", "\n").replace("\r", "\n")


def labels_to_prompt(labels: Iterable[str]) -> str:
    return "\n".join(f"- {label}" for label in labels)


def render_prompt(template: str, text: str, labels: list[str], row: dict[str, object]) -> str:
    if "{{text}}" not in template:
        raise ValueError('Prompt must contain the "{{text}}" placeholder.')

    prompt = template.replace("{{text}}", text)
    prompt = prompt.replace("{{labels}}", labels_to_prompt(labels))
    for key, value in row.items():
        prompt = prompt.replace("{{" + key + "}}", "" if value is None else str(value))
    return prompt


def extract_json_object(raw_output: str) -> dict[str, object]:
    value = raw_output.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM output JSON must be an object.")
    return parsed


def _find_text_span(note_text: str, entity_text: str, context: str | None = None) -> tuple[int, int] | None:
    entity_text = entity_text.strip()
    if not entity_text:
        return None

    if context:
        context_idx = note_text.find(context)
        if context_idx != -1:
            local_idx = note_text.find(entity_text, context_idx, context_idx + len(context))
            if local_idx != -1:
                return local_idx, local_idx + len(entity_text)

    idx = note_text.find(entity_text)
    if idx == -1:
        return None
    return idx, idx + len(entity_text)


def dedupe_spans(spans: Iterable[EntitySpan]) -> list[EntitySpan]:
    seen: set[tuple[int, int, str]] = set()
    result: list[EntitySpan] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end, item.label, item.text)):
        key = (span.start, span.end, span.label)
        if key in seen:
            continue
        seen.add(key)
        result.append(span)
    return result


def parse_llm_entities(note_text: str, raw_output: str, labels: list[str]) -> list[EntitySpan]:
    allowed = set(labels)
    parsed = extract_json_object(raw_output)
    entities = parsed.get("entities", [])
    if not isinstance(entities, list):
        return []

    spans: list[EntitySpan] = []
    for item in entities:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        entity_text = str(item.get("text", "")).strip()
        context = item.get("context")
        if label not in allowed or not entity_text:
            continue
        location = _find_text_span(note_text, entity_text, str(context) if context else None)
        if location is None:
            continue
        start, end = location
        spans.append(EntitySpan(start=start, end=end, label=label, text=note_text[start:end]))
    return dedupe_spans(spans)

