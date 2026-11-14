export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface Source {
  id: string;
  text: string;
  source: string;
  chunk_index: number;
  score: number;
}

export interface AskResult {
  answer: string;
  grounded: boolean;
  attempts: number;
  sources: Source[];
  trace: string[];
}

export interface Health {
  status: string;
  mode: "demo" | "llm";
  chunks: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const getHealth = () => request<Health>("/health");

export const ask = (question: string) =>
  request<AskResult>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

export const uploadFile = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return request<{ source: string; chunks_added: number; total_chunks: number }>("/ingest/file", {
    method: "POST",
    body: form,
  });
};
