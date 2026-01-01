"use client";

import { useState, useCallback } from "react";
import { QueryInput } from "@/components/QueryInput";
import { PipelineVisualization } from "@/components/PipelineVisualization";
import { ClaimsDisplay } from "@/components/ClaimsDisplay";
import { CorrectionHistory } from "@/components/CorrectionHistory";
import { FullPageSkeleton } from "@/components/SkeletonLoader";
import { queryFaithForgeStream, type StageEvent } from "@/lib/api";
import type { PipelineTrace, QueryResponse } from "@/types";

export default function Home() {
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeStage, setActiveStage] = useState<string | null>(null);
  const [stageData, setStageData] = useState<Record<string, Record<string, unknown>>>({});

  const handleQuery = useCallback(async (query: string) => {
    setIsLoading(true);
    setError(null);
    setResponse(null);
    setActiveStage(null);
    setStageData({});

    await queryFaithForgeStream(query, {
      onStage: (event: StageEvent) => {
        if (event.status === "running") {
          setActiveStage(event.stage);
        } else if (event.status === "complete") {
          setStageData((prev) => ({
            ...prev,
            [event.stage]: event.data ?? {},
          }));
          // Brief delay before clearing active stage for visual effect
          setTimeout(() => setActiveStage(null), 300);
        }
      },
      onDone: (result: QueryResponse) => {
        setResponse(result);
        setIsLoading(false);
        setActiveStage(null);
      },
      onError: (msg: string) => {
        setError(msg);
        setIsLoading(false);
        setActiveStage(null);
      },
    });
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Dashboard
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Ask a question and watch the FaithForge pipeline verify every claim.
        </p>
      </div>

      {/* Query Input */}
      <section>
        <QueryInput onSubmit={handleQuery} isLoading={isLoading} />
      </section>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
          <div className="flex items-center justify-between">
            <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
            <button
              onClick={() => setError(null)}
              className="text-red-500 hover:text-red-700 text-sm font-medium"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Loading skeleton */}
      {isLoading && !response && <FullPageSkeleton />}

      {/* Pipeline Visualization */}
      <section>
        <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          Pipeline Trace
        </h2>
        <PipelineVisualization
          trace={response?.trace ?? null}
          activeStage={activeStage}
          stageData={stageData}
        />
      </section>

      {/* Final Answer */}
      {response && (
        <section className="p-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
            Final Answer
          </h2>
          <p className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
            {response.final_answer}
          </p>
          <div className="mt-4 flex items-center gap-4 text-xs text-gray-500">
            <span>Latency: {response.latency_ms.toFixed(0)}ms</span>
            <span aria-hidden="true">•</span>
            <span>
              Faithfulness:{" "}
              {response.trace.all_claims_faithful ? (
                <span className="text-green-600 font-medium">All claims verified ✓</span>
              ) : (
                <span className="text-yellow-600 font-medium">Some claims need correction</span>
              )}
            </span>
          </div>
        </section>
      )}

      {/* Claims & Verifications */}
      {response && (
        <section>
          <ClaimsDisplay
            claims={response.trace.claims}
            verifications={response.trace.verifications}
          />
        </section>
      )}

      {/* Correction History */}
      {response && response.trace.corrections.length > 0 && (
        <section>
          <CorrectionHistory corrections={response.trace.corrections} />
        </section>
      )}

      {/* Retrieved Chunks */}
      {response && response.trace.retrieved_chunks.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            Retrieved Evidence
          </h2>
          <div className="space-y-2">
            {response.trace.retrieved_chunks.map((chunk) => (
              <details
                key={chunk.chunk_id}
                className="p-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800"
              >
                <summary className="flex items-center justify-between cursor-pointer">
                  <span className="text-sm font-mono text-gray-500">
                    [{chunk.chunk_id}]
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">{chunk.source}</span>
                    <span className="px-2 py-0.5 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full">
                      {chunk.retrieval_method}
                    </span>
                    <span className="text-xs font-mono text-gray-400">
                      score: {chunk.score.toFixed(3)}
                    </span>
                  </div>
                </summary>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                  {chunk.text}
                </p>
              </details>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
