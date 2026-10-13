# InferPath Research Prototype Container
# Python 3.11-slim lightweight base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app \
    ADAPTER_MODE=simulated \
    ENVIRONMENT=production

WORKDIR /app

# Install minimal OS dependencies for network health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration and install Python package
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy application code, configurations, data, and dashboard
COPY configs/ /app/configs/
COPY data/ /app/data/
COPY src/ /app/src/
COPY app/ /app/app/

# Expose ports: 8000 (FastAPI), 8501 (Streamlit Dashboard)
EXPOSE 8000 8501

# Default command: launch the FastAPI service
CMD ["uvicorn", "inferpath.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
