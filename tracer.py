import logging
import os

# Expected Behavior
# With LOG_LEVEL=INFO, the logger will:

# Ignore: DEBUG messages.
# Display: INFO, WARNING, ERROR, and CRITICAL messages.

# You can override the log level programmatically if required:
#     tracer.setLevel(logging.WARNING)  # Show only warnings and above


import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def setup_logger():
    """
    Set up the global logger with specific format and handlers.
    Dynamically set log level from an environment variable.
    """
    logger = logging.getLogger("app_tracer")

    # Dynamically set the logging level from the LOG_LEVEL environment variable
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    logger.setLevel(getattr(logging, log_level, logging.DEBUG))  # Default to DEBUG if LOG_LEVEL is not set

    # Avoid adding duplicate handlers if logger already exists
    if logger.hasHandlers():
        return logger

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define a formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    # Optional: Add a file handler
    log_file = os.getenv("LOG_FILE", "application.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Global logger instance
tracer = setup_logger()
