"""
Logging system for PII classification pipeline.
Provides structured logging with file and console outputs.
"""
import logging
import os
from datetime import datetime
from typing import Optional


class PIILogger:
    """Centralized logger for PII classification operations."""
    
    def __init__(self, name: str = "pii_classifier", log_dir: str = "logs"):
        """
        Initialize logger with file and console handlers.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler - daily rotating log
        log_file = os.path.join(
            log_dir, 
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.info(full_message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.warning(full_message)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        if error:
            full_message += f" | Error: {str(error)}"
        self.logger.error(full_message, exc_info=error is not None)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.debug(full_message)
