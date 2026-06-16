from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
from charset_normalizer import from_bytes

from .models import DocumentResult, InputDocument
from .text_utils import normalize_text


PRESERVED_EXPORT_COLUMNS = ("CID", "NOTE_ID", "variant")

# Tried in order before falling back to charset detection. latin-1 is
# excluded here since it never raises UnicodeDecodeError (it maps every byte
# value), which would make the detection step below unreachable.
_FALLBACK_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252")


def _decode_csv_bytes(content: bytes) -> str:
    for encoding in _FALLBACK_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    # None of the common encodings matched; fall back to detection, then
    # latin-1 as a last-resort catch-all that is guaranteed to decode.
    detected = from_bytes(content).best()
    if detected is not None:
        return str(detected)
    return content.decode("latin-1")


def load_csv_documents(content: bytes) -> list[InputDocument]:
    try:
        text = _decode_csv_bytes(content)
        frame = pd.read_csv(io.StringIO(text), dtype=object).fillna("")
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if "text" not in frame.columns:
        raise ValueError('CSV must include a required "text" column.')

    documents: list[InputDocument] = []
    for index, row in frame.iterrows():
        row_dict = {str(key): value for key, value in row.to_dict().items()}
        text = normalize_text(row_dict.get("text", ""))
        if not text.strip():
            continue
        doc_id = str(row_dict.get("ID") or f"case-{index + 1}")
        row_dict["ID"] = doc_id
        row_dict["text"] = text
        documents.append(InputDocument(ID=doc_id, text=text, columns=row_dict))

    if not documents:
        raise ValueError("CSV did not contain any non-empty text rows.")
    return documents


def result_to_export_row(result: DocumentResult) -> dict[str, Any]:
    row: dict[str, Any] = {"ID": result.ID, "text": result.text}
    for column in PRESERVED_EXPORT_COLUMNS:
        if column in result.extra:
            row[column] = result.extra[column]
    row["labels"] = json.dumps([span.model_dump() for span in result.spans], ensure_ascii=False)
    if result.error:
        row["error"] = result.error
    return row


def export_results_csv(results: list[DocumentResult]) -> str:
    rows = [result_to_export_row(result) for result in results]
    frame = pd.DataFrame(rows)
    preferred = ["ID", "text", "labels", "CID", "NOTE_ID", "variant", "error"]
    ordered = [column for column in preferred if column in frame.columns]
    ordered.extend(column for column in frame.columns if column not in ordered)
    return frame[ordered].to_csv(index=False)

