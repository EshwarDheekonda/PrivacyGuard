# Use the official Python image as a parent image.
FROM python:3.11-slim

# Set the working directory in the container.
WORKDIR /app

# Install system dependencies with error handling
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first
COPY requirements.txt .

# Install Python packages with timeout and retries
RUN pip install --no-cache-dir --timeout=120 --retries=3 -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8080

# Simplified health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/apify/health || exit 1

# Corrected CMD to run Hypercorn
CMD ["hypercorn", "-b", "0.0.0.0:8080", "test0812:app"]
