import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Download,
  FileUp,
  Loader2,
  Play,
  RefreshCcw,
} from 'lucide-react';
import {
  ApiError,
  createJob,
  DefaultsResponse,
  DocumentResult,
  EntitySpan,
  exportUrl,
  getDefaults,
  getJobResults,
  getJobStatus,
  JobStatusResponse,
} from './api';

function jobIdFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get('job');
}

function setJobIdInUrl(jobId: string | null): void {
  const url = new URL(window.location.href);
  if (jobId) {
    url.searchParams.set('job', jobId);
  } else {
    url.searchParams.delete('job');
  }
  window.history.replaceState({}, '', url);
}

const PALETTE = [
  '#dc2626',
  '#2563eb',
  '#16a34a',
  '#ca8a04',
  '#9333ea',
  '#ea580c',
  '#db2777',
  '#0891b2',
  '#4f46e5',
  '#059669',
  '#64748b',
  '#7c3aed',
];

function formatSeconds(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return 'estimating';
  if (value < 60) return `${Math.max(0, Math.round(value))}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

function parseLabels(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean)
    .filter((item, index, arr) => arr.indexOf(item) === index);
}

function colorForLabel(label: string, labels: string[]): string {
  const index = Math.max(0, labels.indexOf(label));
  return PALETTE[index % PALETTE.length];
}

function HighlightedText({ doc, labels }: { doc: DocumentResult; labels: string[] }) {
  const spans = useMemo(() => {
    let last = 0;
    return [...doc.spans]
      .sort((a, b) => a.start - b.start || a.end - b.end)
      .filter((span) => {
        const valid = span.start >= last && span.end > span.start && span.end <= doc.text.length;
        if (valid) last = span.end;
        return valid;
      });
  }, [doc]);

  const parts: Array<string | EntitySpan> = [];
  let cursor = 0;
  spans.forEach((span) => {
    if (span.start > cursor) {
      parts.push(doc.text.slice(cursor, span.start));
    }
    parts.push(span);
    cursor = span.end;
  });
  if (cursor < doc.text.length) {
    parts.push(doc.text.slice(cursor));
  }

  return (
    <div className="highlighted-text">
      {parts.map((part, index) => {
        if (typeof part === 'string') {
          return <span key={`text-${index}`}>{part}</span>;
        }
        const color = colorForLabel(part.label, labels);
        return (
          <mark
            key={`${part.start}-${part.end}-${part.label}-${index}`}
            className="entity-mark"
            style={{
              borderColor: color,
              backgroundColor: `${color}22`,
              color,
            }}
            title={`${part.label}: ${part.start}-${part.end}`}
          >
            {doc.text.slice(part.start, part.end)}
            <span className="entity-label">{part.label}</span>
          </mark>
        );
      })}
    </div>
  );
}

function ProgressPanel({ status }: { status: JobStatusResponse | null }) {
  if (!status) {
    return (
      <section className="panel progress-panel">
        <div className="panel-heading">
          <h2>Progress</h2>
        </div>
        <div className="empty-state compact">No active job</div>
      </section>
    );
  }

  const done = status.status === 'completed';
  const failed = status.status === 'failed';
  return (
    <section className="panel progress-panel">
      <div className="panel-heading row-heading">
        <h2>Progress</h2>
        <span className={`status-pill ${failed ? 'failed' : done ? 'done' : 'running'}`}>
          {failed ? <AlertCircle size={14} /> : done ? <CheckCircle2 size={14} /> : <Loader2 size={14} className="spin" />}
          {status.status}
        </span>
      </div>
      <div className="progress-track" aria-label="Extraction progress">
        <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, status.percent))}%` }} />
      </div>
      <div className="metrics-grid">
        <div>
          <span>Completed</span>
          <strong>{status.completed} / {status.total}</strong>
        </div>
        <div>
          <span>Cached</span>
          <strong>{status.cached}</strong>
        </div>
        <div>
          <span>Failed</span>
          <strong>{status.failed}</strong>
        </div>
        <div>
          <span>ETA</span>
          <strong>{formatSeconds(status.eta_seconds)}</strong>
        </div>
      </div>
      <p className="muted single-line">{status.message || 'Waiting'}</p>
    </section>
  );
}

export function App() {
  const [defaults, setDefaults] = useState<DefaultsResponse | null>(null);
  const [labelsText, setLabelsText] = useState('');
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('');
  const [llmUrl, setLlmUrl] = useState('');
  const [maxTokens, setMaxTokens] = useState(1000);
  const [batchSize, setBatchSize] = useState(8);
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(jobIdFromUrl);
  const [status, setStatus] = useState<JobStatusResponse | null>(null);
  const [results, setResults] = useState<DocumentResult[]>([]);
  const [resultLabels, setResultLabels] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getDefaults()
      .then((payload) => {
        setDefaults(payload);
        setLabelsText(payload.default_labels.join('\n'));
        setPrompt(payload.default_prompt);
        setModel(payload.model);
        setLlmUrl(payload.llm_url);
        setMaxTokens(payload.max_tokens);
        setBatchSize(payload.batch_size);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    setJobIdInUrl(jobId);
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let timeout: number | undefined;

    const tick = async () => {
      try {
        const nextStatus = await getJobStatus(jobId);
        if (cancelled) return;
        setStatus(nextStatus);

        if (nextStatus.status === 'completed' || nextStatus.status === 'failed') {
          const payload = await getJobResults(jobId);
          if (cancelled) return;
          setResults(payload.results);
          setResultLabels(payload.labels);
          setSelectedIndex(0);
          return;
        }
        timeout = window.setTimeout(tick, 1000);
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setJobId(null);
          setStatus(null);
          setError('That job is no longer available (the server may have restarted). Start a new run.');
        } else {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    };

    tick();
    return () => {
      cancelled = true;
      if (timeout) window.clearTimeout(timeout);
    };
  }, [jobId]);

  const labels = useMemo(() => parseLabels(labelsText), [labelsText]);
  const selectedDoc = results[selectedIndex] || null;
  const canRun = Boolean(file && labels.length > 0 && prompt.includes('{{text}}') && !isSubmitting);

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0] || null;
    setFile(selected);
    setError(null);
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError('Choose a CSV file first.');
      return;
    }
    if (labels.length === 0) {
      setError('Enter at least one entity label.');
      return;
    }
    if (!prompt.includes('{{text}}')) {
      setError('Prompt must include {{text}}.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setResults([]);
    setStatus(null);
    try {
      const nextJobId = await createJob({
        file,
        labels,
        prompt,
        model,
        llmUrl,
        maxTokens,
        batchSize,
      });
      setJobId(nextJobId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const resetDefaults = () => {
    if (!defaults) return;
    setLabelsText(defaults.default_labels.join('\n'));
    setPrompt(defaults.default_prompt);
    setModel(defaults.model);
    setLlmUrl(defaults.llm_url);
    setMaxTokens(defaults.max_tokens);
    setBatchSize(defaults.batch_size);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>NER LLM Tool</h1>
          <p>gpt-oss extraction with span-level entity output</p>
        </div>
        {jobId && (
          <a className={`button primary ${results.length === 0 ? 'disabled' : ''}`} href={results.length ? exportUrl(jobId) : undefined}>
            <Download size={16} />
            Export CSV
          </a>
        )}
      </header>

      {error && (
        <div className="error-banner" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <main className="workspace-grid">
        <form className="panel setup-panel" onSubmit={onSubmit}>
          <div className="panel-heading row-heading">
            <h2>Run Setup</h2>
            <button className="icon-button" type="button" onClick={resetDefaults} title="Reset defaults" aria-label="Reset defaults">
              <RefreshCcw size={16} />
            </button>
          </div>

          <div className="field">
            <label>CSV file</label>
            <input ref={fileInputRef} type="file" accept=".csv" onChange={onFileChange} hidden />
            <button className="file-button" type="button" onClick={() => fileInputRef.current?.click()}>
              <FileUp size={18} />
              <span>{file ? file.name : 'Choose CSV'}</span>
            </button>
          </div>

          <div className="field">
            <label htmlFor="labels">Entity labels</label>
            <textarea
              id="labels"
              className="labels-box"
              value={labelsText}
              onChange={(event) => setLabelsText(event.target.value)}
              spellCheck={false}
            />
          </div>

          <div className="field two-col">
            <div>
              <label htmlFor="model">Model</label>
              <input id="model" value={model} onChange={(event) => setModel(event.target.value)} />
            </div>
            <div>
              <label htmlFor="batch">Batch</label>
              <input
                id="batch"
                type="number"
                min={1}
                value={batchSize}
                onChange={(event) => setBatchSize(Number(event.target.value))}
              />
            </div>
          </div>

          <div className="field">
            <label htmlFor="url">LLM URL</label>
            <input id="url" value={llmUrl} onChange={(event) => setLlmUrl(event.target.value)} />
          </div>

          <div className="field two-col">
            <div>
              <label htmlFor="maxTokens">Max tokens</label>
              <input
                id="maxTokens"
                type="number"
                min={1}
                value={maxTokens}
                onChange={(event) => setMaxTokens(Number(event.target.value))}
              />
            </div>
            <div>
              <label>Labels</label>
              <div className="readout">{labels.length}</div>
            </div>
          </div>

          <div className="field">
            <label htmlFor="prompt">Prompt</label>
            <textarea
              id="prompt"
              className="prompt-box"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              spellCheck={false}
            />
          </div>

          <button className="button primary wide" type="submit" disabled={!canRun}>
            {isSubmitting ? <Loader2 size={16} className="spin" /> : <Play size={16} />}
            Run extraction
          </button>
        </form>

        <div className="side-stack">
          <ProgressPanel status={status} />
          <section className="panel legend-panel">
            <div className="panel-heading">
              <h2>Labels</h2>
            </div>
            <div className="legend-list">
              {(resultLabels.length ? resultLabels : labels).map((label) => (
                <span className="legend-chip" key={label}>
                  <span className="swatch" style={{ backgroundColor: colorForLabel(label, resultLabels.length ? resultLabels : labels) }} />
                  {label}
                </span>
              ))}
            </div>
          </section>
        </div>

        <section className="results-layout">
          <aside className="panel doc-list-panel">
            <div className="panel-heading row-heading">
              <h2>Documents</h2>
              <span className="count-pill">{results.length}</span>
            </div>
            <div className="doc-list">
              {results.length === 0 && <div className="empty-state compact">No results yet</div>}
              {results.map((doc, index) => (
                <button
                  key={`${doc.ID}-${index}`}
                  type="button"
                  className={`doc-row ${index === selectedIndex ? 'selected' : ''} ${doc.error ? 'has-error' : ''}`}
                  onClick={() => setSelectedIndex(index)}
                >
                  <span className="doc-id">{doc.ID}</span>
                  <span className="doc-meta">{doc.spans.length} spans{doc.cached ? ' · cached' : ''}</span>
                </button>
              ))}
            </div>
          </aside>

          <section className="panel review-panel">
            <div className="panel-heading row-heading">
              <div>
                <h2>{selectedDoc ? selectedDoc.ID : 'Result Review'}</h2>
                {selectedDoc && <p>{selectedDoc.spans.length} extracted spans</p>}
              </div>
              <div className="nav-buttons">
                <button
                  className="icon-button"
                  type="button"
                  onClick={() => setSelectedIndex((value) => Math.max(0, value - 1))}
                  disabled={selectedIndex === 0}
                  aria-label="Previous document"
                  title="Previous document"
                >
                  <ChevronLeft size={16} />
                </button>
                <button
                  className="icon-button"
                  type="button"
                  onClick={() => setSelectedIndex((value) => Math.min(results.length - 1, value + 1))}
                  disabled={selectedIndex >= results.length - 1}
                  aria-label="Next document"
                  title="Next document"
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>

            {!selectedDoc && <div className="empty-state">Run extraction to review highlighted entities.</div>}
            {selectedDoc && (
              <>
                {selectedDoc.error && (
                  <div className="row-error">
                    <AlertCircle size={16} />
                    {selectedDoc.error}
                  </div>
                )}
                <HighlightedText doc={selectedDoc} labels={resultLabels} />
                <div className="span-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Label</th>
                        <th>Text</th>
                        <th>Start</th>
                        <th>End</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedDoc.spans.map((span, index) => (
                        <tr key={`${span.start}-${span.end}-${span.label}-${index}`}>
                          <td>
                            <span className="table-label">
                              <span className="swatch" style={{ backgroundColor: colorForLabel(span.label, resultLabels) }} />
                              {span.label}
                            </span>
                          </td>
                          <td>{span.text}</td>
                          <td>{span.start}</td>
                          <td>{span.end}</td>
                        </tr>
                      ))}
                      {selectedDoc.spans.length === 0 && (
                        <tr>
                          <td colSpan={4} className="muted">No spans extracted</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
        </section>
      </main>
    </div>
  );
}

