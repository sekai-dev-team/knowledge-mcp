"""Tests for the Indexer class indexing operations."""

import os
import sqlite3

import sqlite_vec


class TestIndexer:
    """Tests for indexer full_index, incremental_index, rebuild,
    and change detection."""

    def _get_tables(self, db_path):
        """Return set of all table and virtual table names in the database."""
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        return tables

    def test_full_index_builds_all_tables(self, indexer, db_path):
        """Full index should create all 4 schema tables."""
        indexer.full_index()

        tables = self._get_tables(db_path)
        expected = {"fts_chunks", "chunks_vec", "file_hashes", "chunk_map"}
        missing = expected - tables
        assert not missing, f"Missing tables: {missing}"

    def test_full_index_processes_all_files(self, indexer):
        """All 4 markdown files in sample_vault should be indexed."""
        result = indexer.full_index()

        assert result["new"] == 4
        assert result["skipped"] == 0
        assert result["total"] == 4

    def test_incremental_index_new_file(self, indexer, vault_path):
        """Incremental index should add a new .md file to the index."""
        indexer.full_index()

        new_file = os.path.join(vault_path, "test-new.md")
        try:
            with open(new_file, "w") as f:
                f.write("## Test Section\nThis is a test file.")

            result = indexer.incremental_index(new_file)
            assert result["chunks"] == 1

            status = indexer.index_status()
            assert status["total_files"] == 5  # 4 original + 1 new
        finally:
            if os.path.exists(new_file):
                os.unlink(new_file)

    def test_incremental_index_updates_existing_file(self, indexer, vault_path):
        """Incremental index should update an already-indexed file."""
        indexer.full_index()

        ml_file = os.path.join(vault_path, "machine-learning.md")
        result = indexer.incremental_index(ml_file)
        assert result["chunks"] >= 1
        assert "machine-learning" in result["file"]

    def test_rebuild_drops_and_recreates(self, indexer, db_path):
        """Rebuild should drop old data and recreate the index."""
        indexer.full_index()
        status_before = indexer.index_status()
        assert status_before["total_files"] == 4

        result = indexer.rebuild()

        tables = self._get_tables(db_path)
        expected = {"fts_chunks", "chunks_vec", "file_hashes", "chunk_map"}
        missing = expected - tables
        assert not missing, f"Missing tables after rebuild: {missing}"

        status_after = indexer.index_status()
        assert status_after["total_files"] == 4

    def test_change_detection_skips_unchanged(self, indexer):
        """Second full index should skip all unchanged files."""
        first = indexer.full_index()
        assert first["new"] == 4

        second = indexer.full_index()
        assert second["new"] == 0
        assert second["skipped"] == 4

    def test_file_modification_triggers_update(self, indexer, vault_path):
        """Modifying a file should cause 'updated' count on full index."""
        indexer.full_index()

        # Modify an existing file
        ml_file = os.path.join(vault_path, "machine-learning.md")
        original = ""
        with open(ml_file, "r") as f:
            original = f.read()

        try:
            with open(ml_file, "a") as f:
                f.write("\n\n## New Section\nAdded content for update test.")

            result = indexer.full_index()
            assert result["updated"] >= 1
            assert result["skipped"] >= 3
        finally:
            with open(ml_file, "w") as f:
                f.write(original)

    def test_index_status_returns_all_fields(self, indexer):
        """index_status should return all required fields with valid values."""
        indexer.full_index()

        status = indexer.index_status()
        assert status["total_files"] == 4
        assert status["total_chunks"] == 4
        assert status["db_size_mb"] > 0
        assert status["last_indexed"] is not None

    def test_connection_uses_wal_journal_mode(self, indexer):
        """Connections should use WAL journal mode to allow concurrent access."""
        conn = indexer._get_connection()
        try:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode == "wal", f"Expected 'wal', got '{mode}'"
        finally:
            conn.close()

    def test_connection_has_busy_timeout(self, indexer):
        """Connections should have a busy timeout for concurrent writes."""
        conn = indexer._get_connection()
        try:
            cursor = conn.execute("PRAGMA busy_timeout")
            timeout = cursor.fetchone()[0]
            assert timeout >= 5000, f"busy_timeout={timeout} too low"
        finally:
            conn.close()
