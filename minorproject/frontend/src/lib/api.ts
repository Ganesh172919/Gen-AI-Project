/** FaithForge API client. */

import type { QueryResponse } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types for streaming ─────────────────────────────────────────────────────

export interface StageEvent {
  stage: "planner" | "retriever" | "generator" | "verifier" | "corrector";
  status: "running" | "complete";
  data?: Record<string, unknown>;
}

export interface StreamCallbacks {
  onStage?: (event: StageEvent) => void;
  onDone?: (response: QueryResponse) => void;
  onError?: (error: string) => void;
}

// ── SSE Streaming ───────────────────────────────────────────────────────────

/**
 * Stream a query through the FaithForge pipeline via SSE.
 *
 * Calls onStage for each pipeline stage progress event,
 * and onDone with the final QueryResponse when complete.
 */
export async function queryFaithForgeStream(
  query: string,
  callbacks: StreamCallbacks,
  options?: { maxIterations?: number }
): Promise<void> {
  const params = new URLSearchParams({ q: query });
  if (options?.maxIterations) {
    params.set("max_iterations", String(options.maxIterations));
  }

  const url = `${API_BASE}/query/stream?${params}`;

  try {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Stream failed: ${response.status} ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      let currentEvent = "";
      for (const line of lines) {
        if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          const data = line.slice(5).trim();
          try {
            const parsed = JSON.parse(data);

            if (currentEvent === "stage" && callbacks.onStage) {
              callbacks.onStage(parsed as StageEvent);
            } else if (currentEvent === "done" && callbacks.onDone) {
              callbacks.onDone(parsed as QueryResponse);
            }
          } catch {
            // Skip unparseable data lines
          }
        }
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    callbacks.onError?.(message);
  }
}

// ── Standard POST ───────────────────────────────────────────────────────────

/**
 * Send a query through the FaithForge pipeline (non-streaming).
 */
export async function queryFaithForge(
  query: string,
  options?: { maxIterations?: number }
): Promise<QueryResponse> {
  const resp = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      max_iterations: options?.maxIterations,
    }),
  });

  if (!resp.ok) {
    throw new Error(`Query failed: ${resp.status} ${resp.statusText}`);
  }

  return resp.json();
}

// ── Evaluation ──────────────────────────────────────────────────────────────

/**
 * Submit a batch evaluation job.
 */
export async function submitEvaluation(params: {
  datasetName: string;
  sampleSize?: number;
  runAblations?: boolean;
}): Promise<{ job_id: string; status: string }> {
  const resp = await fetch(`${API_BASE}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_name: params.datasetName,
      sample_size: params.sampleSize,
      run_ablations: params.runAblations ?? true,
    }),
  });

  if (!resp.ok) {
    throw new Error(`Evaluation submission failed: ${resp.status}`);
  }

  return resp.json();
}

/**
 * Poll evaluation job status.
 */
export async function getEvaluationStatus(
  jobId: string
): Promise<{ job_id: string; status: string; error?: string }> {
  const resp = await fetch(`${API_BASE}/evaluate/status/${jobId}`);
  if (!resp.ok) throw new Error(`Status check failed: ${resp.status}`);
  return resp.json();
}

/**
 * Get evaluation results.
 */
export async function getEvaluationResults(jobId: string): Promise<unknown> {
  const resp = await fetch(`${API_BASE}/evaluate/results/${jobId}`);
  if (!resp.ok) throw new Error(`Results fetch failed: ${resp.status}`);
  return resp.json();
}
