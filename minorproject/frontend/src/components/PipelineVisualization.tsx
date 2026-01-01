"use client";

import type { PipelineTrace } from "@/types";

interface PipelineVisualizationProps {
  trace: PipelineTrace | null;
  activeStage?: string | null;
  stageData?: Record<string, Record<string, unknown>>;
}

const STAGES = [
  { key: "planner", label: "Planner", icon: "🧠", ariaLabel: "Planner agent" },
  { key: "retriever", label: "Retriever", icon: "🔍", ariaLabel: "Retriever agent" },
  { key: "generator", label: "Generator", icon: "✍️", ariaLabel: "Generator agent" },
  { key: "verifier", label: "Verifier", icon: "✅", ariaLabel: "Verifier agent" },
  { key: "corrector", label: "Corrector", icon: "🔧", ariaLabel: "Corrector agent" },
] as const;

function StageCard({
  label,
  icon,
  ariaLabel,
  isActive,
  isCompleted,
  detail,
}: {
  label: string;
  icon: string;
  ariaLabel: string;
  isActive: boolean;
  isCompleted: boolean;
  detail?: string;
}) {
  return (
    <div
      role="status"
      aria-label={`${ariaLabel}: ${isActive ? "running" : isCompleted ? "complete" : "waiting"}`}
      className={`flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-all duration-300
        ${isActive ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-lg stage-active" : ""}
        ${isCompleted && !isActive ? "border-green-500 bg-green-50 dark:bg-green-900/20" : ""}
        ${!isActive && !isCompleted ? "border-gray-200 dark:border-gray-700 opacity-40" : ""}
      `}
    >
      <span className="text-2xl" role="img" aria-hidden="true">{icon}</span>
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</span>
      {isActive && (
        <span className="text-xs text-blue-600 dark:text-blue-400 font-medium">running...</span>
      )}
      {detail && !isActive && (
        <span className="text-xs text-gray-500 dark:text-gray-400 text-center">{detail}</span>
      )}
    </div>
  );
}

function Connector({ active }: { active: boolean }) {
  return (
    <div className="flex items-center px-1 sm:px-2">
      <div
        className={`h-0.5 w-4 sm:w-8 transition-colors duration-300 ${active ? "bg-green-500" : "bg-gray-300 dark:bg-gray-600"}`}
      />
      <svg
        className={`w-3 h-3 transition-colors duration-300 ${active ? "text-green-500" : "text-gray-300 dark:text-gray-600"}`}
        viewBox="0 0 12 12"
        aria-hidden="true"
      >
        <path d="M2 6h8M7 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
    </div>
  );
}

export function PipelineVisualization({ trace, activeStage, stageData }: PipelineVisualizationProps) {
  if (!trace && !activeStage) {
    return (
      <div className="flex items-center justify-center p-12 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg">
        <p className="text-gray-400 dark:text-gray-500">
          Submit a query to see the pipeline in action
        </p>
      </div>
    );
  }

  // Determine which stages are completed
  const stageOrder = ["planner", "retriever", "generator", "verifier", "corrector"];
  const activeIndex = activeStage ? stageOrder.indexOf(activeStage) : -1;

  const hasRetrieved = trace ? trace.retrieved_chunks.length > 0 : activeIndex > 1;
  const hasClaims = trace ? trace.claims.length > 0 : activeIndex > 2;
  const hasVerifications = trace ? trace.verifications.length > 0 : activeIndex > 3;
  const correctorCount = (stageData?.corrector?.corrections_count as number) ?? 0;
  const hasCorrections = trace ? trace.corrections.length > 0 : correctorCount > 0;

  const completedStages = [
    true, // planner always completes
    !!hasRetrieved,
    !!hasClaims,
    !!hasVerifications,
    !!(hasCorrections || (trace && trace.verifications.every(v => v.status === "verified"))),
  ];

  const details = [
    trace?.strategy ?? (stageData?.planner?.strategy as string) ?? undefined,
    trace ? `${trace.retrieved_chunks.length} chunks` : (stageData?.retriever?.chunks_found != null ? `${stageData.retriever.chunks_found} chunks` : undefined),
    trace ? `${trace.claims.length} claims` : (stageData?.generator?.claims_count != null ? `${stageData.generator.claims_count} claims` : undefined),
    trace
      ? `${trace.verifications.filter(v => v.status === "verified").length}/${trace.verifications.length} passed`
      : (stageData?.verifier?.passed != null ? `${stageData.verifier.passed}/${stageData.verifier.total} passed` : undefined),
    trace
      ? trace.corrections.length > 0 ? `${trace.corrections.length} corrections` : "No corrections"
      : (stageData?.corrector?.corrections_count != null ? `${stageData.corrector.corrections_count} corrections` : undefined),
  ];

  return (
    <div className="space-y-4">
      {/* Pipeline flow */}
      <div className="flex items-center justify-between overflow-x-auto pb-2">
        {STAGES.map((stage, i) => (
          <div key={stage.key} className="flex items-center">
            <StageCard
              label={stage.label}
              icon={stage.icon}
              ariaLabel={stage.ariaLabel}
              isActive={activeStage === stage.key}
              isCompleted={completedStages[i]}
              detail={details[i]}
            />
            {i < STAGES.length - 1 && (
              <Connector active={completedStages[i]} />
            )}
          </div>
        ))}
      </div>

      {/* Sub-queries display (when multi_hop) */}
      {trace && trace.strategy === "multi_hop" && trace.sub_queries.length > 1 && (
        <div className="p-3 rounded-lg bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800">
          <p className="text-xs font-semibold text-purple-700 dark:text-purple-300 mb-1">
            Multi-hop decomposition:
          </p>
          <ul className="list-disc list-inside space-y-0.5">
            {trace.sub_queries.map((sq, i) => (
              <li key={i} className="text-xs text-purple-600 dark:text-purple-400">{sq}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Iteration badge */}
      {trace && trace.total_iterations > 1 && (
        <div className="flex justify-center">
          <span className="px-3 py-1 text-xs font-medium bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300 rounded-full">
            {trace.total_iterations} iterations
          </span>
        </div>
      )}
    </div>
  );
}
