
import logging
import sys
from logging.handlers import RotatingFileHandler
from src.config import config

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # If logger already has handlers, assume it's set up
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. Console Handler (INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. File Handler (DEBUG) - Standard FileHandler (No Rotation to avoid Win32 Locks)
    log_file = config.LOG_DIR / "app.log"
    # Use 'a' mode (append) and delay=True to minimize lock time
    try:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8', delay=True)
    except PermissionError:
        # Fallback to unique file if locked
        import os
        file_handler = logging.FileHandler(config.LOG_DIR / f"app_{os.getpid()}.log", mode='a', encoding='utf-8')
        
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 3. Error File Handler (ERROR)
    error_log_file = config.LOG_DIR / "error.log"
    try:
        error_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8', delay=True)
    except PermissionError:
        import os
        error_handler = logging.FileHandler(config.LOG_DIR / f"error_{os.getpid()}.log", mode='a', encoding='utf-8')

    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger
