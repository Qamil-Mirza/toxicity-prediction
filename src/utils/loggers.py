import logging
import os

# Define loggers for training and testing
def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with the given name and log file."""
    handler = logging.FileHandler(log_file, mode='w')  # 'w' overwrites the file
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')  # Simpler format for console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger
