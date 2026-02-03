# ============================================================================ 
# Label Studio ML Backend - Production Docker Image
# ============================================================================ 

FROM python:3.11-slim

WORKDIR /app

# Create non-root user with home directory
RUN groupadd -r labelstudio && useradd -r -g labelstudio -m -d /home/labelstudio labelstudio

# Set HOME environment variable
ENV HOME=/home/labelstudio

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=labelstudio:labelstudio main.py .

# Create log directory
RUN mkdir -p /app/logs && \
    chown -R labelstudio:labelstudio /app

# Switch to non-root user
USER labelstudio

# Expose port
EXPOSE 8751

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    WORKER_COUNT=4 \
    THREADS=8 \
    PORT=8751

# Start using gunicorn with Flask app
# label-studio-ml SDK exposes the Flask app as 'app' in main.py
CMD exec gunicorn --bind :$PORT --workers $WORKER_COUNT --threads $THREADS --timeout 0 main:app
