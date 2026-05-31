"""Lightweight CPU-only embedding using ONNX Runtime (no PyTorch).

Generates 384-dimensional L2-normalized vectors using the quantized
Qwen3-Embedding-0.6B model (``electroglyph/Qwen3-Embedding-0.6B-onnx-uint8``),
matching the sqlite-vec ``chunks_vec`` schema.

The uint8 ONNX model outputs a 1024-dim pooled sentence embedding;
we center (subtract 127.5), truncate to 384 dims, and L2-normalize.

Model weights (~625 MB) are downloaded on first use and cached at
``~/.cache/huggingface/hub/``.

HARDENING (2026-05-31):
  - ONNX Runtime thread count clamped to 2 to bound per-inference memory.
  - Input text is silently truncated to ~2000 tokens (MAX_CHARS=8000) to
    prevent attention-matrix memory explosion on oversized chunks.
  - MemoryError / RuntimeError during inference is caught and re-raised
    with a clear diagnostic so the caller can retry with smaller input.
"""

import logging
import os
from functools import lru_cache
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# HF repo and filenames for the quantized ONNX model
_HF_REPO = "electroglyph/Qwen3-Embedding-0.6B-onnx-uint8"
_ONNX_FILE = "dynamic_uint8.onnx"
_TOKENIZER_FILE = "tokenizer.json"

# The ONNX output is a 1024-dim uint8 vector
_HIDDEN_DIM = 1024
_OUTPUT_DIM = 384

# Safety bounds — a ~2000-token input is roughly 8000 chars.
# Beyond this the attention matrix (O(n²)) risks OOM on 2 GB containers.
_MAX_CHARS = 8000

# ONNX intra-op thread count — clamped to 2 to bound memory.
# Default (CPU core count, often 4+) multiplies per-inference buffers
# and can push the container over its memory limit.
_ONNX_INTRA_THREADS = 2
_ONNX_INTER_THREADS = 1


# ---------------------------------------------------------------------------
#  Lazy-loaded model singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the ONNX session and tokenizer (cached).

    Model weights are downloaded from Hugging Face on the first call and
    subsequently served from ``~/.cache/huggingface/hub/``.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    logger.info(
        "Downloading Qwen3-Embedding-0.6B ONNX model "
        "(first load downloads ~625 MB of model weights) ..."
    )

    model_path = hf_hub_download(_HF_REPO, _ONNX_FILE)
    tok_path = hf_hub_download(_HF_REPO, _TOKENIZER_FILE)

    import onnxruntime as ort

    # --- Memory-hardened session options ---
    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = _ONNX_INTRA_THREADS
    session_opts.inter_op_num_threads = _ONNX_INTER_THREADS
    # Disable graph optimisations that may increase memory (we trade speed
    # for stability on a 2 GB container).
    session_opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )

    session = ort.InferenceSession(
        model_path,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = Tokenizer.from_file(tok_path)

    logger.info(
        "Model loaded successfully (ONNX Runtime, CPU, %d intra / %d inter threads).",
        _ONNX_INTRA_THREADS,
        _ONNX_INTER_THREADS,
    )
    return session, tokenizer


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def embed(text: str) -> List[float]:
    """Return a 384-dimensional embedding vector for *text*.

    Uses ``electroglyph/Qwen3-Embedding-0.6B-onnx-uint8`` via ONNX Runtime
    (CPU only, no GPU required).  Input longer than *_MAX_CHARS* (8 000)
    is silently truncated to avoid attention-matrix OOM.

    The returned vector is L2-normalized so that cosine similarity between
    two vectors is equivalent to their dot product.

    Raises:
        RuntimeError: if ONNX inference fails (e.g. out-of-memory).  The
            error message includes the input length so the caller can retry
            with a smaller chunk.
    """
    # --- Truncate oversized input ---
    if len(text) > _MAX_CHARS:
        logger.warning(
            "Truncating input from %d to %d chars to prevent OOM.",
            len(text),
            _MAX_CHARS,
        )
        text = text[:_MAX_CHARS]

    session, tokenizer = _get_model()

    # Tokenize
    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([[1] * len(encoded.ids)], dtype=np.int64)

    # Inference with memory guard
    try:
        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
    except (MemoryError, RuntimeError) as exc:
        raise RuntimeError(
            f"ONNX inference failed for {len(text)}-char input "
            f"({len(encoded.ids)} tokens): {exc}. "
            f"Try reducing chunk size below {_MAX_CHARS}."
        ) from exc

    # ``sentence_embedding_quantized``: uint8 (1, 1024)
    vec = outputs[0][0].astype(np.float32)

    # Center the uint8 [0, 255] range around zero so that L2-normalisation
    # produces meaningful directions for cosine-similarity comparison.
    vec = vec - 127.5

    # Truncate to 384 dimensions
    vec = vec[:_OUTPUT_DIM]

    # L2-normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()
