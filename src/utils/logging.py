"""Lightweight logger + seeding."""
import json
import random
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
import os

def setup_logger(name: str = __name__, log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger

def get_project_logger(module_name: str = None, log_dir: str = "logs") -> logging.Logger:
    """
    Get a pre-configured logger for the project.
    
    Args:
        module_name: Name of the module requesting the logger
        log_dir: Directory to store log files
    
    Returns:
        Configured logger instance
    """
    if module_name is None:
        # Use caller's module name
        import inspect
        caller_frame = inspect.stack()[1]
        module = inspect.getmodule(caller_frame[0])
        module_name = module.__name__ if module else "unknown"
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/project_{timestamp}.log"
    
    return setup_logger(module_name, "INFO", log_filename)

logger = get_project_logger("utils.logging")

def log_json(d: dict, path: str):
    with open(path, "w") as f: json.dump(d, f, indent=2)
def seed_everything(s: int = 42):
    random.seed(s); np.random.seed(s)
