"use client";

import type { Claim, ClaimVerification } from "@/types";

interface ClaimsDisplayProps {
  claims: Claim[];
  verifications: ClaimVerification[];
}

function EntailmentBadge({ label }: { label: string }) {
  const colors: Record<string, string> = {
    entailment: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
    contradiction: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
    neutral: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${colors[label] || colors.neutral}`}>
      {label}
    </span>
  );
}

function FaithfulnessBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? "bg-green-500" : pct >= 40 ? "bg-yellow-500" : "bg-red-500";

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-gray-600 dark:text-gray-400 w-10 text-right">
        {pct}%
      </span>
    </div>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (status === "verified") return <span className="text-green-500">✓</span>;
  if (status === "failed") return <span className="text-red-500">✗</span>;
  if (status === "corrected") return <span className="text-yellow-500">↻</span>;
  return null;
}

export function ClaimsDisplay({ claims, verifications }: ClaimsDisplayProps) {
  if (claims.length === 0) {
    return (
      <div className="p-8 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg text-center">
        <p className="text-gray-400 dark:text-gray-500">
          Claims will appear here after the generator produces an answer
        </p>
      </div>
    );
  }

  // Build verification lookup
  const verificationMap = new Map<string, ClaimVerification>();
  for (const v of verifications) {
    verificationMap.set(v.claim_id, v);
  }

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
        Claims & Faithfulness Scores
      </h3>
      {claims.map((claim) => {
        const verification = verificationMap.get(claim.claim_id);
        return (
          <div
            key={claim.claim_id}
            className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-mono text-gray-400">{claim.claim_id}</span>
                  {verification && <StatusIcon status={verification.status} />}
                </div>
                <p className="text-sm text-gray-800 dark:text-gray-200">{claim.text}</p>
                {claim.source_chunk_ids.length > 0 && (
                  <div className="flex gap-1 mt-2">
                    {claim.source_chunk_ids.map((id) => (
                      <span
                        key={id}
                        className="px-1.5 py-0.5 text-[10px] font-mono bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 rounded"
                      >
                        {id}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              {verification && (
                <div className="flex flex-col items-end gap-1 min-w-[120px]">
                  <EntailmentBadge label={verification.entailment_label} />
                  <FaithfulnessBar score={verification.faithfulness_score} />
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
