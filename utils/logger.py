import logging
import os
from datetime import datetime


def setup_logging():
    """Setup logging to both console and file"""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/agent_conversation_{timestamp}.log"
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return log_filename
