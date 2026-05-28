"""Shared test fixtures for knowledge-mcp tests."""

import hashlib
import os
import struct
import pytest


def deterministic_embed(text: str) -> list[float]:
    """Deterministic 384-dim embedding for tests.

    Produces unit-normalized vectors by hashing the input text,
    allowing deterministic vector search testing without
    sentence-transformers.
    """
    raw = hashlib.md5(text.encode("utf-8")).digest()
    vec = []
    for i in range(384):
        h = hashlib.md5(raw + str(i).encode()).digest()
        val = struct.unpack("I", h[:4])[0] / 4294967295.0
        vec.append(val * 2.0 - 1.0)
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


@pytest.fixture
def db_path(tmp_path):
    """Path to a temporary SQLite database file."""
    return str(tmp_path / "test.db")


@pytest.fixture
def vault_path():
    """Path to the sample vault fixtures directory."""
    return os.path.join(
        os.path.dirname(__file__), "fixtures", "sample_vault"
    )


@pytest.fixture
def mock_embed():
    """Deterministic embedding function for tests."""
    return deterministic_embed


@pytest.fixture
def indexer(db_path, vault_path, mock_embed):
    """Indexer instance configured with test database and vault."""
    from indexer import Indexer

    return Indexer(db_path, vault_path, mock_embed)
