"""
tunnel/logging.py
=================
Structlog configuration for Tunnel Engine.

Call configure_logging() once at process startup (done in tunnel.cli.main()).
All modules get a logger via: log = structlog.get_logger(__name__)

Log format is controlled by TUNNEL_LOG_FORMAT env var:
  console (default)  human-readable, coloured
  json               newline-delimited JSON for log aggregators (prod)
"""
from __future__ import annotations

import logging
import os

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for the current process.

    Should be called exactly once before any log.* calls are made.

    Args:
        level: Minimum log level string: DEBUG, INFO, WARNING, ERROR.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=numeric_level)

    use_json = os.getenv("TUNNEL_LOG_FORMAT", "console") == "json"
    renderer = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
