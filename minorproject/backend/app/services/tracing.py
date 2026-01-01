"""OpenTelemetry tracing setup for FaithForge.

Wires tracing through all planned agent hops:
- Planner → complexity classification
- Retriever → hybrid retrieval (dense + sparse + fusion + rerank)
- Generator → LLM answer generation
- Verifier → claim-level verification
- Corrector → targeted correction

Each hop creates a child span with relevant attributes.
"""

from contextlib import asynccontextmanager
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("faithforge.tracing")

_tracer: Optional[trace.Tracer] = None


def setup_tracing() -> None:
    """Initialize OpenTelemetry tracing.

    Sets up:
    - TracerProvider with service name from config
    - Console exporter for development (swap to OTLP for production)
    - Batch span processor for efficient export
    """
    global _tracer

    resource = Resource.create({
        "service.name": settings.otel_service_name,
    })

    provider = TracerProvider(resource=resource)

    # Console exporter for local development
    # TODO: Add OTLP exporter for production (opentelemetry-exporter-otlp)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(settings.otel_service_name)

    logger.info("OpenTelemetry tracing initialized (service=%s)", settings.otel_service_name)


def get_tracer() -> trace.Tracer:
    """Get the application tracer.

    Returns:
        The OpenTelemetry tracer instance.

    Raises:
        RuntimeError: If tracing hasn't been initialized.
    """
    if _tracer is None:
        raise RuntimeError("Tracing not initialized — call setup_tracing() first")
    return _tracer


@asynccontextmanager
async def trace_agent_hop(name: str, **attributes):
    """Context manager for tracing an agent hop.

    Creates a child span with the given name and attributes.
    Automatically records exceptions if they occur.

    Args:
        name: Span name (e.g., "planner.classify", "retriever.retrieve").
        **attributes: Additional span attributes.

    Usage:
        async with trace_agent_hop("planner.classify", query="What is RAG?"):
            result = await planner.classify(query)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, str(value))
        try:
            yield span
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise
