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

# Pre-download the embedding model weights so runtime does not need
# internet access.  ~80 MB, cached in /root/.cache/fastembed/.
RUN python -c "\
from fastembed import TextEmbedding; \
m = TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2'); \
list(m.embed('warmup')) \
"

WORKDIR /app

# Copy application code
COPY *.py entrypoint.sh .

RUN chmod +x entrypoint.sh

# Runtime directories
RUN mkdir -p /vault /data
VOLUME ["/vault", "/data"]

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
