# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY test_08 .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/apify/health || exit 1


# Start command
CMD ["python", "gcp_app.py"] "
import os
from test0812 import app
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio

config = Config()
config.bind = [f'0.0.0.0:{os.getenv(\"PORT\", 8080)}']
config.use_reloader = False

asyncio.run(serve(app, config))
"
