"""Centralized logging utilities for the CortexAI NLP module.

Purpose:
    Provide consistent logging for loading, validation, preprocessing,
    tokenization, embedding generation, saving, warnings, errors, and timing.
Author:
    Nour Hossam
Dependencies:
    contextlib, functools, logging, pathlib, time
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast


LOGGER_NAME = "cortexai.nlp"
F = TypeVar("F", bound=Callable[..., Any])


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure and return the shared NLP logger.

    Args:
        level: Logging threshold.
        log_file: Optional log file path.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        resolved_log_file = log_file.resolve()
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == resolved_log_file
            for handler in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    return logger


def get_logger() -> logging.Logger:
    """Return the shared NLP logger, configuring it on first use."""
    return configure_logging()


@contextmanager
def log_duration(message: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Log execution time for a block of work.

    Args:
        message: Human-readable operation name.
        logger: Optional logger override.
    """
    active_logger = logger or get_logger()
    start_time = time.perf_counter()
    active_logger.info("%s started.", message)
    try:
        yield
    except Exception:
        active_logger.exception("%s failed.", message)
        raise
    finally:
        elapsed = time.perf_counter() - start_time
        active_logger.info("%s finished in %.2f seconds.", message, elapsed)


def log_execution_time(message: str) -> Callable[[F], F]:
    """Decorate a function so its execution time is logged.

    Args:
        message: Human-readable operation name.

    Returns:
        Function decorator.
    """

    def decorator(function: F) -> F:
        """Wrap one function with duration logging."""

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute the wrapped function inside a timed log block."""
            with log_duration(message):
                return function(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
