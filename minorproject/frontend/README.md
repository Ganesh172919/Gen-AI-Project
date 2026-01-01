# FaithForge Frontend

The FaithForge frontend is a Next.js application that provides a real-time interface for the multi-agent RAG system.

## Features

### Dashboard Page (`/`)

- **Query Input:** Submit questions to the FaithForge pipeline
- **Real-Time Pipeline Visualization:** Watch each stage execute in real-time via SSE
- **Claims Display:** View extracted claims with verification status
- **Correction History:** See how failed claims were corrected

### Evaluation Page (`/evaluate`)

- **Dataset Selection:** Choose from RAGTruth, HotpotQA, or custom datasets
- **Sample Size Control:** Configure evaluation scale
- **Ablation Toggle:** Enable/disable ablation studies
- **Job Monitoring:** Track evaluation progress
- **Results Display:** View metrics and ablation results

## Tech Stack

- **Framework:** Next.js 16.2
- **UI Library:** React 19
- **Styling:** Tailwind CSS 4
- **Language:** TypeScript 5
- **API Client:** Custom SSE + REST client

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at http://localhost:3000

### Build

```bash
npm run build
npm start
```

## Project Structure

```
src/
├── app/
│   ├── page.tsx              # Dashboard page
│   ├── evaluate/
│   │   └── page.tsx          # Evaluation page
│   ├── layout.tsx            # Root layout with navigation
│   └── globals.css           # Global styles
├── components/
│   ├── QueryInput.tsx        # Query input form
│   ├── PipelineVisualization.tsx  # Pipeline stage visualization
│   ├── ClaimsDisplay.tsx     # Claim verification results
│   ├── CorrectionHistory.tsx # Correction diff display
│   └── SkeletonLoader.tsx    # Loading skeletons
├── lib/
│   └── api.ts                # API client (SSE + REST)
└── types/
    └── index.ts              # TypeScript type definitions
```

## Components

### QueryInput

A form component for submitting queries with optional parameters:
- Query text input
- Top-K override
- Max iterations override

### PipelineVisualization

Real-time visualization of the FaithForge pipeline:
- Stage indicators (planner, retriever, generator, verifier, corrector)
- Status badges (running, complete, failed)
- Timing information

### ClaimsDisplay

Displays extracted claims with their verification status:
- Claim text
- Entailment label (entailment, contradiction, neutral)
- Faithfulness score
- Verification status (verified, failed)

### CorrectionHistory

Shows how failed claims were corrected:
- Original claim
- Corrected claim
- New evidence
- Iteration number

## API Integration

### SSE Streaming

The dashboard uses Server-Sent Events for real-time updates:

```typescript
const eventSource = new EventSource(`/query/stream?q=${query}`);

eventSource.addEventListener('stage', (event) => {
  const data = JSON.parse(event.data);
  // Update pipeline state
});

eventSource.addEventListener('done', (event) => {
  const result = JSON.parse(event.data);
  // Display final result
});
```

### REST API

The evaluation page uses REST endpoints:

```typescript
// Submit evaluation
const { job_id } = await api.post('/evaluate', {
  dataset_name: 'ragtruth',
  sample_size: 100,
});

// Poll status
const status = await api.get(`/evaluate/status/${job_id}`);

// Get results
const results = await api.get(`/evaluate/results/${job_id}`);
```

## Styling

Uses Tailwind CSS with custom components:

```tsx
<div className="bg-white rounded-lg shadow-md p-6">
  <h2 className="text-xl font-bold mb-4">Pipeline Results</h2>
  {/* ... */}
</div>
```

## Environment Variables

Create `.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## License

MIT
