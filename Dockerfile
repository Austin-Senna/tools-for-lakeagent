FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY pyproject.toml README.md ./
COPY aurum_v2/ aurum_v2/

# Install the package
# RUN pip install --no-cache-dir .

# Volumes for persistent data and model output
VOLUME ["/app/data", "/app/model"]

# AWS credentials via environment (set at runtime)
ENV AWS_DEFAULT_REGION=us-east-1

ENTRYPOINT ["python", "-m", "aurum_v2.pipeline"]
CMD ["--help"]
