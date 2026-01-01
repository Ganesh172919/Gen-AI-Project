"use client";

import type { CorrectionRecord } from "@/types";

interface CorrectionHistoryProps {
  corrections: CorrectionRecord[];
}

export function CorrectionHistory({ corrections }: CorrectionHistoryProps) {
  if (corrections.length === 0) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
        Correction History
      </h3>
      {corrections.map((correction, i) => (
        <div
          key={`${correction.claim_id}-${i}`}
          className="p-4 rounded-lg border border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/10"
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-mono text-gray-400">{correction.claim_id}</span>
            <span className="px-2 py-0.5 text-xs bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200 rounded-full">
              Iteration {correction.iteration}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Original</p>
              <p className="text-sm text-red-700 dark:text-red-300 line-through">
                {correction.original_claim}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Corrected</p>
              <p className="text-sm text-green-700 dark:text-green-300">
                {correction.corrected_claim || (
                  <span className="italic text-gray-400">Insufficient evidence</span>
                )}
              </p>
            </div>
          </div>

          {correction.new_evidence.length > 0 && (
            <details className="mt-2">
              <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                {correction.new_evidence.length} new evidence chunks
              </summary>
              <div className="mt-1 space-y-1">
                {correction.new_evidence.map((chunk) => (
                  <p key={chunk.chunk_id} className="text-xs text-gray-600 dark:text-gray-400 pl-2 border-l-2 border-gray-300">
                    [{chunk.chunk_id}] {chunk.text.slice(0, 150)}...
                  </p>
                ))}
              </div>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}
