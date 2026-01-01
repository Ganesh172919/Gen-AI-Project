"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  submitEvaluation,
  getEvaluationStatus,
  getEvaluationResults,
} from "@/lib/api";

type JobStatus = "pending" | "running" | "completed" | "failed";

interface JobInfo {
  job_id: string;
  status: JobStatus;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

export default function EvaluatePage() {
  const [datasetName, setDatasetName] = useState("ragtruth");
  const [sampleSize, setSampleSize] = useState<number>(50);
  const [runAblations, setRunAblations] = useState(true);
  const [currentJob, setCurrentJob] = useState<JobInfo | null>(null);
  const [results, setResults] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  // Poll for job status
  const pollJob = useCallback(async (jobId: string) => {
    try {
      const status = await getEvaluationStatus(jobId);
      setCurrentJob((prev) =>
        prev
          ? { ...prev, ...status, status: status.status as JobStatus }
          : { ...status, status: status.status as JobStatus }
      );

      if (status.status === "completed") {
        // Fetch results
        const res = await getEvaluationResults(jobId);
        setResults(JSON.stringify(res, null, 2));
        if (pollRef.current) clearInterval(pollRef.current);
      } else if (status.status === "failed") {
        setError(status.error || "Evaluation failed");
        if (pollRef.current) clearInterval(pollRef.current);
      }
    } catch (e) {
      // Polling error — keep trying
    }
  }, []);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setResults(null);
    setCurrentJob(null);

    try {
      const { job_id, status } = await submitEvaluation({
        datasetName,
        sampleSize,
        runAblations,
      });

      setCurrentJob({ job_id, status: status as JobStatus });

      // Start polling
      pollRef.current = setInterval(() => pollJob(job_id), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit evaluation");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Evaluation
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Run batch evaluations and ablation studies against your test datasets.
        </p>
      </div>

      {/* Submission form */}
      <section className="p-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          Submit Evaluation Job
        </h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Dataset */}
            <div>
              <label
                htmlFor="dataset"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
              >
                Dataset
              </label>
              <select
                id="dataset"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="ragtruth">RAGTruth</option>
                <option value="hotpotqa">HotpotQA</option>
                <option value="custom">Custom</option>
              </select>
            </div>

            {/* Sample size */}
            <div>
              <label
                htmlFor="sampleSize"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
              >
                Sample Size
              </label>
              <input
                id="sampleSize"
                type="number"
                min={1}
                max={1000}
                value={sampleSize}
                onChange={(e) => setSampleSize(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Ablations toggle */}
            <div className="flex items-end">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={runAblations}
                  onChange={(e) => setRunAblations(e.target.checked)}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Run ablation studies
                </span>
              </label>
            </div>
          </div>

          <button
            type="submit"
            disabled={isSubmitting || (currentJob?.status === "running" || currentJob?.status === "pending")}
            className="px-6 py-2 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isSubmitting ? "Submitting..." : "Run Evaluation"}
          </button>
        </form>
      </section>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {/* Job status */}
      {currentJob && (
        <section className="p-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            Job Status
          </h2>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-500">Job ID:</span>
              <span className="font-mono text-sm text-gray-700 dark:text-gray-300">
                {currentJob.job_id}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-500">Status:</span>
              <StatusBadge status={currentJob.status} />
            </div>
            {currentJob.created_at && (
              <div className="flex items-center gap-3">
                <span className="text-sm text-gray-500">Created:</span>
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  {new Date(currentJob.created_at).toLocaleString()}
                </span>
              </div>
            )}
            {(currentJob.status === "pending" || currentJob.status === "running") && (
              <div className="flex items-center gap-2 mt-2">
                <svg className="animate-spin h-4 w-4 text-blue-500" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span className="text-sm text-blue-600 dark:text-blue-400">
                  {currentJob.status === "pending" ? "Waiting in queue..." : "Processing..."}
                </span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Results */}
      {results && (
        <section className="p-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            Results
          </h2>
          <pre className="p-4 rounded-lg bg-gray-50 dark:bg-gray-900 text-sm font-mono text-gray-700 dark:text-gray-300 overflow-x-auto">
            {results}
          </pre>
        </section>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: JobStatus }) {
  const colors: Record<JobStatus, string> = {
    pending: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
    running: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
    completed: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
    failed: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  };

  return (
    <span
      className={`px-2.5 py-0.5 text-xs font-medium rounded-full ${colors[status]}`}
    >
      {status}
    </span>
  );
}
