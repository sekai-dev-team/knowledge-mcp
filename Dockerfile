FROM python:3.12-slim

# Build dependencies for sqlite-vec and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# sqlite-vec must be installed first so its native extension is available
RUN pip install --no-cache-dir sqlite-vec

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py entrypoint.sh .

RUN chmod +x entrypoint.sh

# Runtime directories
RUN mkdir -p /vault /data
VOLUME ["/vault", "/data"]

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
