export interface EntitySpan {
  start: number;
  end: number;
  label: string;
  text: string;
}

export interface DocumentResult {
  ID: string;
  text: string;
  spans: EntitySpan[];
  cached: boolean;
  error: string | null;
  extra: Record<string, unknown>;
}

export interface DefaultsResponse {
  default_prompt: string;
  default_labels: string[];
  model: string;
  llm_url: string;
  max_tokens: number;
  batch_size: number;
}

export interface JobStatusResponse {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  total: number;
  completed: number;
  cached: number;
  failed: number;
  percent: number;
  elapsed_seconds: number;
  eta_seconds: number | null;
  message: string | null;
}

export interface JobResultsResponse {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  labels: string[];
  results: DocumentResult[];
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      message = await response.text();
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export async function getDefaults(): Promise<DefaultsResponse> {
  return readJson<DefaultsResponse>(await fetch(`${API_BASE}/api/defaults`));
}

export async function createJob(options: {
  file: File;
  labels: string[];
  prompt: string;
  model: string;
  llmUrl: string;
  maxTokens: number;
  batchSize: number;
}): Promise<string> {
  const form = new FormData();
  form.append('file', options.file);
  form.append('labels', JSON.stringify(options.labels));
  form.append('prompt', options.prompt);
  form.append('model', options.model);
  form.append('llm_url', options.llmUrl);
  form.append('max_tokens', String(options.maxTokens));
  form.append('batch_size', String(options.batchSize));

  const payload = await readJson<{ job_id: string }>(
    await fetch(`${API_BASE}/api/jobs`, {
      method: 'POST',
      body: form,
    })
  );
  return payload.job_id;
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return readJson<JobStatusResponse>(await fetch(`${API_BASE}/api/jobs/${jobId}`));
}

export async function getJobResults(jobId: string): Promise<JobResultsResponse> {
  return readJson<JobResultsResponse>(await fetch(`${API_BASE}/api/jobs/${jobId}/results`));
}

export function exportUrl(jobId: string): string {
  return `${API_BASE}/api/jobs/${jobId}/export.csv`;
}

