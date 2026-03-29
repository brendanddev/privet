"""
utils/logger

Handles logging within privet.
"""

import logging
import os
from datetime import datetime

logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)


def setup_logger(log_dir: str = "./logs", level=logging.INFO) -> logging.Logger:
    """
    Configure and return a logger that writes to both the terminal and a log file.

    Args:
        log_dir (str): Directory to store log files. Defaults to ./logs
        level: Logging level. Defaults to INFO.

    Returns:
        logging.Logger: Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger("local-rag-assistant")
    logger.setLevel(level)

    # Prevent messages from propagating up to the root logger
    # This is what causes the duplicate lines
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger