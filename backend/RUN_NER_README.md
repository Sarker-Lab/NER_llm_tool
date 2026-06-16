# run_ner.py — offline NER extraction

Runs the same extraction pipeline as the web app, from the command line, on
a CSV file. No server needed.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python run_ner.py --input cases.csv --output results.csv
```

`cases.csv` must have a `text` column (an `ID` column is optional). The
output CSV has `ID`, `text`, `labels` (JSON array of extracted entity
spans), and an `error` column for any rows that failed.

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `--labels` | built-in labels | Comma-separated list, or path to a JSON file with a list of labels |
| `--prompt` | built-in prompt | Path to a prompt template file (must contain `{{text}}`) |
| `--model` | `gpt-oss-120b` | LLM model name |
| `--llm-url` | production LLM endpoint | LLM completions URL |
| `--max-tokens` | `1000` | Max tokens per LLM response |
| `--batch-size` | `8` | Documents sent per LLM request |
| `--timeout` | `120` | Per-request timeout (seconds) |
| `--cache` | `backend/data/cache.sqlite` | SQLite cache path — shared with the web app by default, so repeated rows are reused instead of re-querying the LLM |

Run `python run_ner.py --help` for the full list.
