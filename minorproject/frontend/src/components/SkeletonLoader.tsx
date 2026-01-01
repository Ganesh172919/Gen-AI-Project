"use client";

/** Skeleton loader components for FaithForge. */

export function PipelineSkeleton() {
  return (
    <div className="flex items-center justify-between overflow-x-auto pb-2">
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className="flex items-center">
          <div className="flex flex-col items-center gap-2 p-4 rounded-lg border-2 border-gray-200 dark:border-gray-700">
            <div className="skeleton w-8 h-8 rounded-full" />
            <div className="skeleton w-16 h-3" />
          </div>
          {i < 5 && (
            <div className="flex items-center px-2">
              <div className="h-0.5 w-8 bg-gray-200 dark:bg-gray-700" />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function ClaimsSkeleton() {
  return (
    <div className="space-y-3">
      <div className="skeleton w-40 h-4" />
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 space-y-2">
              <div className="skeleton w-8 h-3" />
              <div className="skeleton w-full h-4" />
              <div className="skeleton w-3/4 h-4" />
              <div className="flex gap-1 mt-2">
                <div className="skeleton w-16 h-4 rounded" />
              </div>
            </div>
            <div className="flex flex-col items-end gap-1 min-w-[120px]">
              <div className="skeleton w-20 h-5 rounded-full" />
              <div className="skeleton w-full h-2 rounded-full" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export function AnswerSkeleton() {
  return (
    <div className="p-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
      <div className="skeleton w-28 h-5 mb-3" />
      <div className="space-y-2">
        <div className="skeleton w-full h-4" />
        <div className="skeleton w-full h-4" />
        <div className="skeleton w-2/3 h-4" />
      </div>
      <div className="mt-4 flex items-center gap-4">
        <div className="skeleton w-20 h-3" />
        <div className="skeleton w-32 h-3" />
      </div>
    </div>
  );
}

export function FullPageSkeleton() {
  return (
    <div className="space-y-8 animate-pulse">
      <PipelineSkeleton />
      <AnswerSkeleton />
      <ClaimsSkeleton />
    </div>
  );
}
