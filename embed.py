"""Lightweight CPU-only embedding using fastembed (ONNX Runtime).

No PyTorch, no CUDA.  Generates 384-dimensional L2-normalized vectors
using all-MiniLM-L6-v2, matching the sqlite-vec ``chunks_vec`` schema.

The model weights (~80 MB) are downloaded on first use and cached at
``~/.cache/fastembed/``.
"""

import logging
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Lazy-loaded model singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the fastembed ``TextEmbedding`` model (cached).

    The Hugging Face model weights are downloaded on the first call and
    subsequently served from the local cache.
    """
    from fastembed import TextEmbedding

    logger.info(
        "Loading all-MiniLM-L6-v2 embedding model "
        "(first load downloads ~80 MB of model weights) ..."
    )
    return TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def embed(text: str) -> List[float]:
    """Return a 384‑dimensional embedding vector for *text*.

    Uses ``sentence-transformers/all-MiniLM-L6-v2`` via
    fastembed / ONNX Runtime (CPU only, no GPU required).

    The returned vector is L2‑normalized so that cosine similarity between
    two vectors is equivalent to their dot product.
    """
    model = _get_model()
    # ``model.embed()`` yields numpy ``ndarray`` objects; consume the iterator.
    vec = list(model.embed(text))[0]
    return vec.tolist()
