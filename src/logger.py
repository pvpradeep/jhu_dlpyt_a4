"""Logging utilities for UCF50 action recognition training.

Provides a global logger pattern: call setup_logger() once at startup,
then get_logger() anywhere to retrieve the same instance.
log_and_print writes to both the log file and tqdm.write.
"""

import logging
import os
from datetime import datetime

from tqdm import tqdm

LOGGER_NAME = "ucf50"


def setup_logger(log_dir="../logs"):
    """Initialize the global logger with a timestamped log file.

    Call once at the start of a run. Subsequent calls add a new
    file handler (new timestamp) but reuse the same logger.

    Args:
        log_dir: Directory to store log files.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{LOGGER_NAME}_{timestamp}.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    # Clear old handlers so repeated setup_logger calls don't duplicate
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger():
    """Retrieve the global logger. Call setup_logger() first.

    Returns:
        logging.Logger: The global logger instance.
    """
    return logging.getLogger(LOGGER_NAME)


def log_and_print(message):
    """Log message to file and print via tqdm.write (tqdm-safe).

    Args:
        message: String message to log and print.
    """
    get_logger().info(message)
    tqdm.write(message)


def log_args(args):
    """Log all argparse arguments.

    Args:
        args: argparse.Namespace object.
    """
    log_and_print("=" * 60)
    log_and_print("Run Configuration:")
    for key, value in sorted(vars(args).items()):
        log_and_print(f"  {key}: {value}")
    log_and_print("=" * 60)