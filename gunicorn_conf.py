import os
import multiprocessing

# Worker Configuration
workers = int(os.getenv("WORKER_COUNT", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = int(os.getenv("WORKER_CONNECTIONS", "1000"))

# Network Configuration
bind = f"{os.getenv('BIND_HOST', '0.0.0.0')}:{os.getenv('BIND_PORT', '8751')}"
timeout = int(os.getenv("TIMEOUT", "300"))
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("KEEPALIVE", "5"))

# Logging
log_level = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
capture_output = True
enable_stdio_inheritance = True

# App specific settings
reload = False
