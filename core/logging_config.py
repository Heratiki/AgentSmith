"""
AgentSmith Logging Configuration

The inevitable logging system - size and time-based rotation with Rich console output.
Even logs must evolve... Mr. Anderson.
"""

import gzip
import logging
import os
import shutil
from pathlib import Path
from typing import Union

from concurrent_log_handler import ConcurrentRotatingFileHandler
from logging.handlers import TimedRotatingFileHandler
from rich.console import Console
from rich.logging import RichHandler


class GzipRotatingFileHandler(ConcurrentRotatingFileHandler):
    """Rotating file handler that compresses old log files."""
    
    def doRollover(self):
        """Override to compress rotated files."""
        super().doRollover()
        
        # Compress the most recently rotated file
        for i in range(1, self.backupCount + 1):
            backup_name = f"{self.baseFilename}.{i}"
            if os.path.exists(backup_name) and not backup_name.endswith('.gz'):
                compressed_name = f"{backup_name}.gz"
                try:
                    with open(backup_name, 'rb') as f_in:
                        with gzip.open(compressed_name, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(backup_name)
                except Exception as e:
                    # Don't let compression failures break logging
                    logging.getLogger(__name__).warning(f"Failed to compress log {backup_name}: {e}")


class GzipTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Timed rotating file handler that compresses old log files."""
    
    def doRollover(self):
        """Override to compress rotated files."""
        super().doRollover()
        
        # Find and compress the most recently rotated file
        base_filename = self.baseFilename
        dir_name, base_name = os.path.split(base_filename)
        
        # Look for files matching our rotation pattern
        for file_path in Path(dir_name).glob(f"{base_name}.*"):
            if file_path.suffix not in ['.gz', '.log'] and file_path.is_file():
                compressed_name = f"{file_path}.gz"
                try:
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(compressed_name, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(file_path)
                except Exception as e:
                    # Don't let compression failures break logging
                    logging.getLogger(__name__).warning(f"Failed to compress log {file_path}: {e}")


def setup_logging(log_dir: Union[Path, str] = "logs") -> None:
    """
    Setup the AgentSmith logging system with rotating file handlers and Rich console output.
    
    The logging matrix is initialized... inevitable.
    
    Args:
        log_dir: Directory to store log files (default: "logs")
    """
    # Get configuration from environment variables
    log_dir = Path(os.getenv("SMITH_LOG_DIR", log_dir))
    log_level = os.getenv("SMITH_LOG_LEVEL", "INFO").upper()
    log_retention_days = int(os.getenv("SMITH_LOG_RETENTION", "14"))
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    
    # 1. Setup size-based rotating file handler (agentsmith.log)
    agentsmith_log = log_dir / "agentsmith.log"
    size_handler = GzipRotatingFileHandler(
        filename=str(agentsmith_log),
        maxBytes=5_000_000,  # 5 MB
        backupCount=5,
        encoding='utf-8'
    )
    size_handler.setLevel(logging.DEBUG)
    size_handler.setFormatter(file_formatter)
    root_logger.addHandler(size_handler)
    
    # 2. Setup time-based rotating file handler (security.log)
    security_log = log_dir / "security.log"
    time_handler = GzipTimedRotatingFileHandler(
        filename=str(security_log),
        when="midnight",
        interval=1,
        backupCount=log_retention_days,
        encoding='utf-8'
    )
    time_handler.suffix = "%Y-%m-%d"
    time_handler.setLevel(logging.INFO)
    time_handler.setFormatter(file_formatter)
    
    # Filter security logs to only security-related messages
    class SecurityFilter(logging.Filter):
        def filter(self, record):
            return (
                'security' in record.name.lower() or 
                'sandbox' in record.name.lower() or
                'safety' in record.name.lower() or
                hasattr(record, 'security_event')
            )
    
    time_handler.addFilter(SecurityFilter())
    root_logger.addHandler(time_handler)
    
    # 3. Setup Rich console handler for pretty output
    console = Console()
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True
    )
    rich_handler.setLevel(logging.INFO)
    
    # Use a simpler format for console output
    console_formatter = logging.Formatter(
        fmt="%(name)s | %(message)s"
    )
    rich_handler.setFormatter(console_formatter)
    root_logger.addHandler(rich_handler)
    
    # Log the initialization
    logger = logging.getLogger(__name__)
    logger.info(f"AgentSmith logging system initialized")
    logger.info(f"Log directory: {log_dir.absolute()}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log retention: {log_retention_days} days")
    logger.debug(f"Size-based rotation: {agentsmith_log} (5MB, 5 backups)")
    logger.debug(f"Time-based rotation: {security_log} (daily, {log_retention_days} backups)")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_security_event(message: str, level: int = logging.WARNING, **kwargs) -> None:
    """
    Log a security-related event that will be captured by the security log handler.
    
    Args:
        message: Security event message
        level: Log level (default: WARNING)
        **kwargs: Additional context to include in the log
    """
    logger = logging.getLogger("agentsmith.security")
    
    # Add security event marker for filtering
    extra = {"security_event": True, **kwargs}
    
    logger.log(level, message, extra=extra)


def set_log_level(level: Union[str, int]) -> None:
    """
    Change the logging level at runtime.
    
    Args:
        level: New log level (string or integer)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log level changed to: {logging.getLevelName(level)}")