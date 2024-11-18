import logging
import os

# Define loggers for training and testing
def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)

    # Check if the logger already has handlers to avoid duplicates
    if not logger.hasHandlers():
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream logs to console
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    logger.setLevel(level)
    return logger
