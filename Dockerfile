FROM python:3.12-slim

# Install system dependencies: git for backup, ca-certificates for HTTPS
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install Syncthing (static binary, ~11 MB)
ARG SYNCTHING_VERSION=2.1.0
RUN curl -fsSL \
    "https://github.com/syncthing/syncthing/releases/download/v${SYNCTHING_VERSION}/syncthing-linux-amd64-v${SYNCTHING_VERSION}.tar.gz" \
    -o /tmp/st.tar.gz \
    && tar xzf /tmp/st.tar.gz -C /tmp \
    && cp /tmp/syncthing-linux-amd64-v${SYNCTHING_VERSION}/syncthing /usr/local/bin/syncthing \
    && chmod +x /usr/local/bin/syncthing \
    && rm -rf /tmp/st.tar.gz /tmp/syncthing-linux-amd64-v${SYNCTHING_VERSION}

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: The Qwen3-Embedding-0.6B ONNX model (~625 MB) is downloaded on first
# use via embed.py and cached in /root/.cache/huggingface/hub/.  It is NOT
# pre-downloaded here to keep the Docker image size under 1 GB.

WORKDIR /app

# Copy application code
COPY *.py entrypoint.sh .

RUN chmod +x entrypoint.sh

# Runtime directories
RUN mkdir -p /vault /data
VOLUME ["/vault", "/data"]

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
