"""Tests for the MCP HTTP server (mcp_server.py).

We test the FastAPI /health endpoint via TestClient and the MCP tool
helper functions directly.  The tool handlers are thin wrappers around
Indexer methods (already tested in test_indexer.py / test_search.py),
so we focus on:
  - Health endpoint
  - Tool registration
  - _read_note  (get_note logic)
  - _run_reindex (reindex logic)
  - search and index_status dispatched through handle_call_tool
"""

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

from mcp_server import (
    _read_note,
    _run_reindex,
    _write_note,
    app,
    handle_call_tool,
    handle_list_tools,
    mcp_server,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def configured_app(db_path, vault_path, mock_embed):
    """Set app state so the lifespan event can create an Indexer."""
    app.state.vault_path = vault_path
    app.state.db_path = db_path
    app.state.embed_fn = mock_embed
    return app


# ---------------------------------------------------------------------------
#  Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint(configured_app):
    """GET /health returns {"status": "ok"}."""
    with TestClient(configured_app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
#  Tool registration
# ---------------------------------------------------------------------------


def test_tools_registered():
    """All six MCP tools should be registered on the server."""
    tools = asyncio.run(handle_list_tools())
    names = {t.name for t in tools}
    assert names == {"search", "get_note", "reindex", "index_status", "write_note", "update_note", "list_notes"}


# ---------------------------------------------------------------------------
#  Search tool
# ---------------------------------------------------------------------------


def test_search_tool(indexer):
    """Search tool dispatches to Indexer.search and returns results."""
    indexer.full_index()
    import mcp_server as ms

    ms._indexer = indexer

    result = asyncio.run(ms.handle_call_tool("search", {"query": "machine learning", "limit": 3}))
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert len(data) > 0
    assert any(r["path"] == "machine-learning.md" for r in data)
    # Verify result shape
    for key in ("path", "section_title", "snippet", "bm25_score", "vec_score", "combined_score"):
        assert key in data[0]


# ---------------------------------------------------------------------------
#  get_note tool
# ---------------------------------------------------------------------------


def test_get_note_tool(indexer):
    """_read_note returns content, frontmatter, and backlinks for a valid file."""
    indexer.full_index()
    result = _read_note(indexer, "machine-learning.md")

    assert result["path"] == "machine-learning.md"
    assert "Supervised Learning" in result["content"]
    assert result["frontmatter"]["title"] == "Machine Learning Fundamentals"
    assert "ml" in result["frontmatter"]["tags"]
    assert "ResNet" in result["backlinks"]
    assert "BERT" in result["backlinks"]


def test_get_note_tool_with_frontmatter(indexer):
    """get_note parses frontmatter with various types (list, string)."""
    indexer.full_index()
    result = _read_note(indexer, "python-async.md")

    assert result["path"] == "python-async.md"
    assert result["frontmatter"]["title"] == "Python Async Programming"
    assert "python" in result["frontmatter"]["tags"]
    assert "Concurrency Patterns" in result["backlinks"]


def test_get_note_tool_not_found(indexer):
    """_read_note returns an error dict when the file does not exist."""
    result = _read_note(indexer, "nonexistent-file.md")
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_get_note_returns_wiki_links(indexer):
    """_read_note extracts [[wiki links]] from the note body."""
    indexer.full_index()
    result = _read_note(indexer, "architecture.md")
    assert len(result["backlinks"]) == 1
    assert "Machine Learning Fundamentals" in result["backlinks"]


def test_get_note_date_frontmatter_serializes_to_json(indexer):
    """Date values in frontmatter are serialized as ISO strings through json.dumps.

    Regression test: yaml.safe_load parses ``date: 2025-01-15`` as a
    ``datetime.date`` object, which would crash json.dumps without the
    _serialize_value helper.
    """
    import json
    import mcp_server as ms

    ms._indexer = indexer
    indexer.full_index()

    # Go through handle_call_tool so json.dumps is exercised
    result = asyncio.run(
        ms.handle_call_tool("get_note", {"path": "machine-learning.md"})
    )
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "error" not in data
    assert data["path"] == "machine-learning.md"
    # The date frontmatter should be an ISO string, not a datetime.date
    assert data["frontmatter"]["date"] == "2025-01-15"


# ---------------------------------------------------------------------------
#  Reindex tool
# ---------------------------------------------------------------------------


def test_reindex_full_rebuild(indexer):
    """_run_reindex with no path triggers full rebuild and returns status."""
    indexer.full_index()
    result = _run_reindex(indexer)
    assert result["status"] == "ok"
    assert result["files_processed"] >= 1
    assert result["elapsed"] >= 0


def test_reindex_single_file(indexer, vault_path):
    """_run_reindex with a path reindexes a single file."""
    indexer.full_index()
    result = _run_reindex(indexer, "machine-learning.md")
    assert result["status"] == "ok"
    assert result["files_processed"] == 1


def test_reindex_file_not_found(indexer):
    """_run_reindex returns error for a non-existent file."""
    result = _run_reindex(indexer, "does-not-exist.md")
    assert "error" in result


# ---------------------------------------------------------------------------
#  Index status tool
# ---------------------------------------------------------------------------


def test_index_status_tool(indexer):
    """Index status dispatches to Indexer.index_status and returns fields."""
    indexer.full_index()
    import mcp_server as ms

    ms._indexer = indexer

    result = asyncio.run(ms.handle_call_tool("index_status", {}))
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert data["total_files"] == 4
    assert data["total_chunks"] == 4
    assert data["db_size_mb"] > 0
    assert data["last_indexed"] is not None


# ---------------------------------------------------------------------------
#  Error handling
# ---------------------------------------------------------------------------


def test_unknown_tool_returns_error(indexer):
    """Calling a non-existent tool returns an error message."""
    import mcp_server as ms

    ms._indexer = indexer
    result = asyncio.run(ms.handle_call_tool("nonexistent", {}))
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "error" in data
    assert "Unknown tool" in data["error"]


def test_missing_required_argument_returns_error(indexer):
    """Calling get_note without 'path' should raise, caught by handler."""
    import mcp_server as ms

    ms._indexer = indexer
    result = asyncio.run(ms.handle_call_tool("get_note", {}))
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "error" in data


# ---------------------------------------------------------------------------
#  write_note tool — semantic dedup
# ---------------------------------------------------------------------------


def test_write_note_no_duplicate(indexer):
    """Writing a completely new note with no similar content succeeds."""
    import mcp_server as ms

    ms._indexer = indexer
    indexer.full_index()

    result = _write_note(
        indexer,
        "_tmp_unique_note.md",
        "This is completely unique content about something different.",
    )
    assert result["path"] == "_tmp_unique_note.md"
    assert result["updated"] is True

    # Clean up
    os.remove(os.path.join(indexer.vault_path, "_tmp_unique_note.md"))


def test_write_note_duplicate_found(tmp_path, mock_embed):
    """Writing content identical to an existing note returns duplicate warning."""
    from indexer import Indexer
    import mcp_server as ms

    vault = tmp_path / "vault"
    vault.mkdir()
    db = tmp_path / "test.db"

    existing_content = (
        "## Test Section\n\nThis is the content of an existing note."
    )
    (vault / "existing.md").write_text(existing_content, encoding="utf-8")

    indexer = Indexer(str(db), str(vault), mock_embed)
    indexer.full_index()
    ms._indexer = indexer

    # Try to write a new note with the same content
    result = _write_note(indexer, "new-note.md", existing_content)

    assert result["status"] == "duplicate_found"
    assert "query" in result
    assert result["similar_note"]["path"] == "existing.md"
    assert result["similar_note"]["vec_score"] > 0.85
    assert "suggestion" in result

    # Verify the file was NOT written
    assert not (vault / "new-note.md").exists()


def test_write_note_force_bypasses_dedup(tmp_path, mock_embed):
    """force=True bypasses the semantic dedup check and writes normally."""
    from indexer import Indexer
    import mcp_server as ms

    vault = tmp_path / "vault"
    vault.mkdir()
    db = tmp_path / "test.db"

    existing_content = (
        "## Test Section\n\nThis is the content of an existing note."
    )
    (vault / "existing.md").write_text(existing_content, encoding="utf-8")

    indexer = Indexer(str(db), str(vault), mock_embed)
    indexer.full_index()
    ms._indexer = indexer

    # With force=True, even identical content should write
    result = _write_note(
        indexer, "forced-note.md", existing_content, force=True
    )

    assert result["path"] == "forced-note.md"
    assert result["updated"] is True
    assert (vault / "forced-note.md").exists()


def test_write_note_filename_conflict_checked_first(indexer):
    """Filename conflict error takes priority over dedup check."""
    import mcp_server as ms

    ms._indexer = indexer
    indexer.full_index()

    # Try to write to an existing file path
    result = _write_note(
        indexer,
        "machine-learning.md",
        "Some content that might be a duplicate.",
    )
    assert "error" in result
    assert "already exists" in result["error"]
