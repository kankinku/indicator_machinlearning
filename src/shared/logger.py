import logging
import sys
import multiprocessing
from logging.handlers import QueueHandler, QueueListener
from typing import Optional
from src.config import config

# Global state for logging
_log_queue: Optional[multiprocessing.Queue] = None
_listener: Optional[QueueListener] = None

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance. 
    Standard usage: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers and name != "":
        # Ensure we don't add duplicate handlers if root is already configured
        logger.propagate = True
    return logger

def setup_main_logging():
    """
    Initializes the master logging system in the main process.
    Uses QueueListener to aggregate logs from all processes into a single writer.
    """
    global _log_queue, _listener
    
    # Root logger configuration
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    # Standard format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | [%(process)d] | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handlers (Only in Main Process)
    # 1. Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    
    # 2. Main Log File
    log_file = config.LOG_DIR / "app.log"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_h = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)
    
    # 3. Error Log File
    error_file = config.LOG_DIR / "error.log"
    error_h = logging.FileHandler(error_file, mode='a', encoding='utf-8')
    error_h.setLevel(logging.ERROR)
    error_h.setFormatter(formatter)
    
    # Setup Queue and Listener
    _log_queue = multiprocessing.Manager().Queue(-1)
    _listener = QueueListener(_log_queue, console, file_h, error_h, respect_handler_level=True)
    _listener.start()
    
    # Set the root logger to use QueueHandler as well (consistency)
    # This way even logs in the main process go through the queue
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(QueueHandler(_log_queue))
    
    return _log_queue

def setup_worker_logging(queue: multiprocessing.Queue):
    """
    Initializes logging in a worker process.
    Removes all existing handlers and replaces them with a single QueueHandler.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    # Clear any existing (inherited) handlers
    for h in root.handlers[:]:
        root.removeHandler(h)
        
    # Append the worker context (PID) to messages if needed, 
    # but QueueListener handles the actual writing.
    root.addHandler(QueueHandler(queue))

def stop_main_logging():
    """Stops the logging listener."""
    global _listener
    if _listener:
        _listener.stop()
        _listener = None
