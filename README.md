# NER LLM Tool

Local webapp for LLM-based named entity extraction.

## Backend

```bash
cd webapp/backend
python -m pip install -r requirements.txt
python app.py
```

## Frontend

```bash
cd webapp/frontend
npm install
npm run dev
```

The frontend dev server proxies `/api` to `http://localhost:5002`.

## Input and Output

- Input CSV requires a `text` column. `ID` is optional and generated as `case-N` when absent.
- Entity labels are entered one per line.
- Exported CSV contains `ID`, `text`, and `labels`, where `labels` is a JSON array of `{start,end,label,text}` spans.
