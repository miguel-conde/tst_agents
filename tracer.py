import logging
import os

from utils import get_env_variable

# Expected Behavior
# With LOG_LEVEL=INFO, the logger will:

# Ignore: DEBUG messages.
# Display: INFO, WARNING, ERROR, and CRITICAL messages.

# You can override the log level programmatically if required:
#     tracer.setLevel(logging.WARNING)  # Show only warnings and above


import os
from dotenv import load_dotenv

# Load .env file
loaded = load_dotenv(verbose=True)

# print(f"El valor de la variable de entorno LOG_LEVEL es: {get_env_variable('LOG_LVL')}")

# if loaded:
#     print("Environment variables loaded successfully")
# else:
#     print("No environment variables loaded")

def setup_logger():
    """
    Set up the global logger with specific format and handlers.
    Dynamically set log level from an environment variable.
    """
    logger = logging.getLogger("app_tracer")

    # Dynamically set the logging level from the LOG_LEVEL environment variable
    log_level = os.getenv("LOG_LVL", "DEBUG").upper()
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        logging.warning(f"Invalid log level: {log_level}. Defaulting to DEBUG")
        log_level = "DEBUG"
    else: # format to 'logging.LEVEL' to use in the logging module
        log_level = getattr(logging, log_level, logging.INFO)  # Convert log_level string to int value from logging module. Default to DEBUG if LOG_LEVEL is not set
 
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if logger already exists
    if logger.hasHandlers():
        return logger

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

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
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Global logger instance
tracer = setup_logger()


# Obtén el nivel actual del logger
log_level = tracer.getEffectiveLevel()

# Convierte el nivel a su representación en texto
log_level_name = logging.getLevelName(log_level)

tracer.info(f"El nivel de log actual es: {log_level_name}")