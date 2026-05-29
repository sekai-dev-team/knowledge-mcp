FROM python:3.12-slim

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
