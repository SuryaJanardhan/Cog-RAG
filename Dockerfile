FROM python:3.12-slim

# Prevent Python from writing pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for building packages (e.g. pg_config, gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and configurations
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py .

# Create cache and data persistence mountpoints
RUN mkdir -p cache data

EXPOSE 8000

# Start server
CMD ["python", "scripts/run_server.py"]
