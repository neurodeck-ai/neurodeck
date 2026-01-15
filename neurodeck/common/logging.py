"""Logging configuration for NeuroDeck using loguru."""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """
    Set up structured logging for NeuroDeck.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        enable_console: Whether to log to console
    """
    # Remove default handler
    logger.remove()
    
    # Console logging
    if enable_console:
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
    
    # File logging
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="1 week",
            compression="zip"
        )
    
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str):
    """Get a logger instance for a specific component."""
    return logger.bind(name=name)


# Component-specific loggers
orchestrator_logger = get_logger("orchestrator")
agent_logger = get_logger("agent")
console_logger = get_logger("console")
config_logger = get_logger("config")
protocol_logger = get_logger("protocol")