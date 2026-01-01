"""Structured logging setup for FaithForge.

Provides a consistent log format across all modules with:
- Console handler with colored output for development
- File handler with daily rotation (optional)
- JSON structured logging for production
- Request correlation ID propagation
"""

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production environments.

    Outputs structured JSON logs with:
    - timestamp (ISO 8601)
    - level
    - module:function:line
    - message
    - request_id (if set via context)
    - Any extra fields passed via logger.info("msg", extra={...})
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": f"{record.module}:{record.funcName}:{record.lineno}",
            "message": record.getMessage(),
        }

        # Add request_id if set via LoggerAdapter or extra
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # Add any extra fields
        for key in ["latency_ms", "stage", "claims_count", "chunks_count", "iteration"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable log formatter for development.

    Format: timestamp | LEVEL | module:function:line | message
    """

    # ANSI color codes for terminal output
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",   # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Add color for terminal output
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"

        # Truncate logger name for readability
        if record.name.startswith("faithforge."):
            record.name = record.name[len("faithforge."):]

        return super().format(record)


def setup_logging() -> None:
    """Configure application-wide logging.

    Sets up:
    - Console handler with format based on LOG_FORMAT setting
    - File handler with rotation (if LOG_FILE_PATH is set)
    - Structured format with timestamp, module, and level

    Environment variables:
    - LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
    - LOG_FORMAT: "text" or "json" (default: "text")
    - LOG_FILE_PATH: Path to log file (optional)
    - LOG_MAX_BYTES: Max file size before rotation (default: 10MB)
    - LOG_BACKUP_COUNT: Number of backup files (default: 7)
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    log_format = getattr(settings, "log_format", "text")

    # Create root logger
    root_logger = logging.getLogger("faithforge")
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # ── Console Handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_format = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        )
        console_handler.setFormatter(HumanReadableFormatter(console_format))

    root_logger.addHandler(console_handler)

    # ── File Handler (optional) ──────────────────────────────────────────────
    log_file_path = getattr(settings, "log_file_path", None)
    if log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        max_bytes = getattr(settings, "log_max_bytes", 10_000_000)  # 10MB default
        backup_count = getattr(settings, "log_backup_count", 7)

        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)

        # Always use JSON for file logs (machine-readable)
        file_handler.setFormatter(JSONFormatter())

        root_logger.addHandler(file_handler)
        root_logger.info("File logging enabled: %s (max=%dMB, backups=%d)",
                         log_path, max_bytes // 1_000_000, backup_count)

    # ── Silence noisy loggers ────────────────────────────────────────────────
    noisy_loggers = [
        "httpx", "httpcore", "chromadb", "sentence_transformers",
        "transformers", "torch", "urllib3", "asyncio",
    ]
    for noisy in noisy_loggers:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root_logger.info(
        "Logging initialized (level=%s, format=%s)",
        settings.log_level, log_format,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a module.

    Args:
        name: Module name (e.g., 'faithforge.retriever', 'faithforge.verifier')

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class RequestContextFilter(logging.Filter):
    """Logging filter that injects request_id into log records.

    Use with FastAPI middleware to propagate request IDs through all logs.
    """

    def __init__(self):
        super().__init__()
        self._request_id: Optional[str] = None

    def set_request_id(self, request_id: str) -> None:
        """Set the current request ID."""
        self._request_id = request_id

    def clear_request_id(self) -> None:
        """Clear the current request ID."""
        self._request_id = None

    def filter(self, record: logging.LogRecord) -> bool:
        """Inject request_id into the log record."""
        record.request_id = self._request_id or "-"
        return True


# Global request context filter
request_filter = RequestContextFilter()
