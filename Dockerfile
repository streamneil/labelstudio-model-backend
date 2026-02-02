# ============================================================================ 
# Label Studio ML Backend - Production Docker Image
# ============================================================================ 

FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r labelstudio && useradd -r -g labelstudio labelstudio

# Install dependencies
# Using standard PyPI, can be overridden by build args if needed
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=labelstudio:labelstudio main.py gunicorn_conf.py ./

# Create log directory and ensure permissions
RUN mkdir -p /app/logs && \
    chown -R labelstudio:labelstudio /app

# Switch to non-root user
USER labelstudio

# Expose port
EXPOSE 8751

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8751/health')" || exit 1

# Default environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WORKER_COUNT=4 \
    WORKER_CONNECTIONS=1000 \
    BIND_HOST=0.0.0.0 \
    BIND_PORT=8751 \
    TIMEOUT=300 \
    GRACEFUL_TIMEOUT=30 \
    KEEPALIVE=5 \
    LOG_LEVEL=INFO \
    MAX_CONCURRENT=10

# Start Gunicorn with configuration file
CMD ["gunicorn", "main:app", "-c", "gunicorn_conf.py"]