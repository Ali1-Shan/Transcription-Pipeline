"""Structured logging setup using Loguru.

Provides consistent, structured logging across the entire application
with JSON-formatted output suitable for log aggregation systems.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure Loguru with structured JSON logging.

    Args:
        log_level: Minimum log level to capture.
        log_dir: Directory for log file output.
    """
    # Remove default handler
    logger.remove()

    # Structured format for production: timestamp, level, module, message
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console handler with structured format
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level.upper(),
        colorize=True,
        backtrace=False,
        diagnose=False,  # Disable variable inspection in production
    )

    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File handler with JSON serialization for log aggregation
    logger.add(
        str(log_path / "pipeline.log"),
        level=log_level.upper(),
        rotation="50 MB",
        retention="30 days",
        compression="gz",
        serialize=True,
        backtrace=False,
        diagnose=False,
    )

    logger.info("Logger initialized | level={} | log_dir={}", log_level, log_dir)
