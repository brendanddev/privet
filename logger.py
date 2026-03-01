
import logging
import os
from datetime import datetime

def setup_logger(log_dir: str = "./logs", level=logging.INFO) -> logging.Logger:
    """
    Configure and return a logger that writes to both the terminal and a log file.

    Args:
        log_dir (str): Directory to store log files. Defaults to ./logs
        level: Logging level. Defaults to INFO.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Name the log file by date so each day gets its own file
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

    # Create the logger
    logger = logging.getLogger("local-rag-assistant")
    logger.setLevel(level)

    # Avoid adding duplicate handlers if this is called multiple times
    if logger.handlers:
        return logger

    # Format: timestamp | level | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler to write to log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Handler to print to termimal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger