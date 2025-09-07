# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install Python packages and create a "virtual environment" in /install
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# Stage 2: Final image for the application
FROM python:3.11-slim

# Create and switch to the non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app
WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local
# Copy the application code
COPY . .

# Use the PORT environment variable provided by Cloud Run
ENV PORT 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/apify/health || exit 1

# Start the application with Hypercorn
CMD ["hypercorn", "-b", "0.0.0.0:$PORT", "test0812:app"]
