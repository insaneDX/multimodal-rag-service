import sys
from loguru import logger

# Remove default handler
logger.remove()

# Add console handler
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
)

# Add file handler
logger.add(
    "app_logs.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

# Prevent duplicate logging
logger.configure(extra={"app_name": "multimodal-rag"})