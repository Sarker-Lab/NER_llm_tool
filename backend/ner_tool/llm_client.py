from __future__ import annotations

import time
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class LLMConfig:
    model: str
    url: str
    max_tokens: int
    batch_size: int
    timeout_seconds: float


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete_batch(self, prompts: list[str]) -> list[str]:
        payload = {
            "model": self.config.model,
            "prompt": prompts,
            "max_tokens": self.config.max_tokens,
            "temperature": 1,
            "top_p": 1,
        }
        headers = {"Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    self.config.url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                return self._extract_texts(response.json(), len(prompts))
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after 3 attempts: {last_error}") from last_error

    @staticmethod
    def _extract_texts(data: object, expected_count: int) -> list[str]:
        if isinstance(data, list):
            answers = [str(item["choices"][0]["text"]).strip() for item in data]
        elif isinstance(data, dict) and isinstance(data.get("choices"), list):
            answers = [str(item.get("text", "")).strip() for item in data["choices"]]
        else:
            raise ValueError(f"Unexpected LLM response format: {data!r}")

        if len(answers) != expected_count:
            raise ValueError(f"Expected {expected_count} LLM answers, got {len(answers)}.")
        return answers

