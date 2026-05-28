"""Hybrid search indexer using SQLite FTS5 + sqlite-vec.

Supports BM25 full-text search and vector cosine similarity search
over Markdown knowledge bases.
"""

import hashlib
import os
import re
import struct
import time


class Indexer:
    """Core indexing engine for hybrid search over Markdown knowledge bases.

    Uses SQLite FTS5 for BM25 full-text search and sqlite-vec for vector
    similarity search. Supports full index, incremental index, rebuild,
    and hybrid (RRF) search.

    Args:
        db_path: Path to SQLite database file.
        vault_path: Path to markdown vault directory.
        embed_fn: Callable ``(text: str) -> list[float]`` returning a
            384-dimensional embedding vector.
    """

    def __init__(self, db_path: str, vault_path: str, embed_fn):
        self.db_path = db_path
        self.vault_path = vault_path
        self.embed_fn = embed_fn

    # ------------------------------------------------------------------
    #  Connection helpers
    # ------------------------------------------------------------------

    def _get_connection(self):
        """Open a new SQLite connection with the sqlite-vec extension loaded."""
        import sqlite3
        import sqlite_vec

        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return conn

    def _init_schema(self, conn):
        """Create all schema tables if they do not already exist."""
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                path, section_title, content,
                tokenize='porter unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                embedding float[384]
            );

            CREATE TABLE IF NOT EXISTS file_hashes (
                path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunk_map (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                section_title TEXT,
                content_preview TEXT,
                fts_rowid INTEGER,
                vec_rowid INTEGER
            );
        """)

    # ------------------------------------------------------------------
    #  File chunking
    # ------------------------------------------------------------------

    def _chunk_file(self, filepath: str) -> list[dict]:
        """Split a markdown file into chunks by ``##`` headings.

        Returns a list of dicts with keys *path* (relative to vault),
        *section_title*, and *content*.
        """
        relpath = os.path.relpath(filepath, self.vault_path)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Strip YAML front matter
        content = re.sub(r"^---.*?---\n*", "", content, flags=re.DOTALL)

        # Short files are kept as a single chunk
        if len(content) < 2000:
            return [
                {
                    "path": relpath,
                    "section_title": "",
                    "content": content.strip(),
                }
            ]

        # Split on "## " heading boundaries
        chunks = []
        sections = re.split(r"\n(?=## )", content)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            title = ""
            for line in section.split("\n"):
                m = re.match(r"^##\s+(.+)$", line)
                if m:
                    title = m.group(1).strip()
                    break

            chunks.append(
                {
                    "path": relpath,
                    "section_title": title,
                    "content": section,
                }
            )

        return chunks

    # ------------------------------------------------------------------
    #  Hashing & change detection
    # ------------------------------------------------------------------

    def _compute_hash(self, filepath: str) -> str:
        """Return the SHA-256 hex digest of *filepath*."""
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _is_unchanged(self, conn, filepath: str, sha256: str) -> bool:
        """Return True when the stored hash matches *sha256*."""
        relpath = os.path.relpath(filepath, self.vault_path)
        cursor = conn.execute(
            "SELECT sha256 FROM file_hashes WHERE path = ?", (relpath,)
        )
        row = cursor.fetchone()
        return row is not None and row[0] == sha256

    # ------------------------------------------------------------------
    #  Data deletion
    # ------------------------------------------------------------------

    def _delete_file_data(self, conn, relpath: str):
        """Remove every trace of *relpath* from the index tables."""
        cursor = conn.execute(
            "SELECT fts_rowid, vec_rowid FROM chunk_map WHERE path = ?",
            (relpath,),
        )
        rows = cursor.fetchall()

        for fts_rid, _ in rows:
            if fts_rid is not None:
                conn.execute(
                    "DELETE FROM fts_chunks WHERE rowid = ?", (fts_rid,)
                )

        for _, vec_rid in rows:
            if vec_rid is not None:
                conn.execute(
                    "DELETE FROM chunks_vec WHERE rowid = ?", (vec_rid,)
                )

        conn.execute("DELETE FROM chunk_map WHERE path = ?", (relpath,))
        conn.execute("DELETE FROM file_hashes WHERE path = ?", (relpath,))

    # ------------------------------------------------------------------
    #  Single file indexing
    # ------------------------------------------------------------------

    def _index_file(self, conn, filepath: str) -> int:
        """Index the chunks of *filepath*.

        Returns the number of chunks indexed.
        """
        chunks = self._chunk_file(filepath)

        for chunk in chunks:
            content = chunk["content"]

            # --- FTS5 ---
            cursor = conn.execute(
                "INSERT INTO fts_chunks(path, section_title, content) "
                "VALUES (?, ?, ?)",
                (chunk["path"], chunk["section_title"], content),
            )
            fts_rowid = cursor.lastrowid

            # --- Vector ---
            vec = self.embed_fn(content)
            vec_bytes = struct.pack(f"{len(vec)}f", *vec)
            conn.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                (fts_rowid, vec_bytes),
            )

            # --- Mapping ---
            conn.execute(
                "INSERT INTO chunk_map"
                "(path, section_title, content_preview, fts_rowid, vec_rowid) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    chunk["path"],
                    chunk["section_title"],
                    content[:100],
                    fts_rowid,
                    fts_rowid,
                ),
            )

        return len(chunks)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def full_index(self) -> dict:
        """Index all ``.md`` files in the vault.

        Returns:
            A dict with keys ``new``, ``updated``, ``skipped``, ``total``,
            and ``time`` (seconds).
        """
        start = time.time()
        conn = self._get_connection()
        try:
            self._init_schema(conn)

            stats = {"new": 0, "updated": 0, "skipped": 0, "total": 0}

            for root, dirs, files in os.walk(self.vault_path):
                # Skip hidden directories and .obsidian
                dirs[:] = [
                    d
                    for d in dirs
                    if not d.startswith(".") and d != ".obsidian"
                ]

                for fname in files:
                    if not fname.endswith(".md"):
                        continue

                    filepath = os.path.join(root, fname)
                    relpath = os.path.relpath(filepath, self.vault_path)
                    sha256 = self._compute_hash(filepath)

                    stats["total"] += 1

                    if self._is_unchanged(conn, filepath, sha256):
                        stats["skipped"] += 1
                    else:
                        is_update = (
                            conn.execute(
                                "SELECT 1 FROM file_hashes WHERE path = ?",
                                (relpath,),
                            ).fetchone()
                            is not None
                        )

                        self._delete_file_data(conn, relpath)
                        self._index_file(conn, filepath)

                        conn.execute(
                            "INSERT OR REPLACE INTO file_hashes(path, sha256) "
                            "VALUES (?, ?)",
                            (relpath, sha256),
                        )

                        if is_update:
                            stats["updated"] += 1
                        else:
                            stats["new"] += 1

            conn.commit()

            return {
                "new": stats["new"],
                "updated": stats["updated"],
                "skipped": stats["skipped"],
                "total": stats["total"],
                "time": round(time.time() - start, 3),
            }
        finally:
            conn.close()

    def incremental_index(self, filepath: str) -> dict:
        """Index a single file, removing any previously indexed data.

        Args:
            filepath: Absolute or relative (to vault) path to the file.

        Returns:
            A dict with keys ``file``, ``chunks``, and ``time``.
        """
        start = time.time()

        if not os.path.isabs(filepath):
            filepath = os.path.join(self.vault_path, filepath)

        conn = self._get_connection()
        try:
            self._init_schema(conn)

            relpath = os.path.relpath(filepath, self.vault_path)
            self._delete_file_data(conn, relpath)

            num_chunks = self._index_file(conn, filepath)

            sha256 = self._compute_hash(filepath)
            conn.execute(
                "INSERT OR REPLACE INTO file_hashes(path, sha256) "
                "VALUES (?, ?)",
                (relpath, sha256),
            )
            conn.commit()

            return {
                "file": relpath,
                "chunks": num_chunks,
                "time": round(time.time() - start, 3),
            }
        finally:
            conn.close()

    def rebuild(self) -> dict:
        """Drop all tables and re-run a full index.

        Returns:
            The result dict from :meth:`full_index`.
        """
        conn = self._get_connection()
        try:
            conn.executescript("""
                DROP TABLE IF EXISTS fts_chunks;
                DROP TABLE IF EXISTS chunks_vec;
                DROP TABLE IF EXISTS file_hashes;
                DROP TABLE IF EXISTS chunk_map;
            """)
            conn.commit()
        finally:
            conn.close()

        return self.full_index()

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Hybrid search: BM25 + vector cosine similarity with RRF merge.

        Args:
            query: Search terms.
            limit: Maximum number of results.

        Returns:
            A list of dicts, each with keys ``path``, ``section_title``,
            ``snippet``, ``bm25_score``, ``vec_score``, and
            ``combined_score``.
        """
        if not query or not query.strip():
            return []

        conn = self._get_connection()
        try:
            self._init_schema(conn)

            bm25_results = self._fts5_search(conn, query, limit * 2)
            query_vec = self.embed_fn(query)
            vec_results = self._vec_search(conn, query_vec, limit * 2)

            combined = self._rrf_merge(bm25_results, vec_results, k=60)

            for r in combined[:limit]:
                r["snippet"] = self._fts5_snippet(conn, r["path"], query)

            return combined[:limit]
        finally:
            conn.close()

    def index_status(self) -> dict:
        """Return the current index size.

        Returns:
            A dict with keys ``files`` and ``chunks``.
        """
        conn = self._get_connection()
        try:
            self._init_schema(conn)

            cursor = conn.execute("SELECT COUNT(*) FROM file_hashes")
            file_count = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM chunk_map")
            chunk_count = cursor.fetchone()[0]

            return {"files": file_count, "chunks": chunk_count}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    #  Internal search helpers
    # ------------------------------------------------------------------

    def _fts5_search(self, conn, query: str, limit: int) -> list[dict]:
        """BM25 full-text search via FTS5."""
        terms = re.findall(r"\w+", query)
        if not terms:
            return []

        fts_query = " OR ".join(terms)

        cursor = conn.execute(
            """SELECT rowid, path, section_title, content,
                      bm25(fts_chunks) AS score
               FROM fts_chunks
               WHERE fts_chunks MATCH ?
               ORDER BY score
               LIMIT ?""",
            (fts_query, limit),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "path": row[1],
                    "section_title": row[2],
                    "bm25_score": -row[4],  # negate FTS5 negation
                    "vec_score": 0.0,
                }
            )
        return results

    def _vec_search(self, conn, query_vec: list[float], limit: int) -> list[dict]:
        """Vector similarity search via sqlite-vec."""
        vec_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)

        cursor = conn.execute(
            """SELECT v.rowid, cm.path, cm.section_title, v.distance
               FROM chunks_vec AS v
               LEFT JOIN chunk_map AS cm ON v.rowid = cm.vec_rowid
               WHERE v.embedding MATCH ?
                 AND k = ?""",
            (vec_bytes, limit),
        )

        results = []
        for row in cursor.fetchall():
            # L2 distance → similarity in [0, 1]
            similarity = 1.0 / (1.0 + float(row[3]))
            results.append(
                {
                    "path": row[1],
                    "section_title": row[2],
                    "bm25_score": 0.0,
                    "vec_score": similarity,
                }
            )
        return results

    def _rrf_merge(
        self,
        bm25_results: list[dict],
        vec_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion merge of two ranked result lists."""
        merged: dict[tuple, dict] = {}

        for rank, r in enumerate(bm25_results):
            key = (r["path"], r.get("section_title", ""))
            if key not in merged:
                merged[key] = {
                    "path": r["path"],
                    "section_title": r.get("section_title", ""),
                    "bm25_score": r.get("bm25_score", 0.0),
                    "vec_score": 0.0,
                    "rrf_score": 0.0,
                }
            merged[key]["rrf_score"] += 1.0 / (k + rank + 1)

        for rank, r in enumerate(vec_results):
            key = (r["path"], r.get("section_title", ""))
            if key not in merged:
                merged[key] = {
                    "path": r["path"],
                    "section_title": r.get("section_title", ""),
                    "bm25_score": 0.0,
                    "vec_score": r.get("vec_score", 0.0),
                    "rrf_score": 0.0,
                }
            else:
                merged[key]["vec_score"] = r.get("vec_score", 0.0)
            merged[key]["rrf_score"] += 1.0 / (k + rank + 1)

        sorted_results = sorted(
            merged.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return [
            {
                "path": item["path"],
                "section_title": item["section_title"],
                "snippet": "",
                "bm25_score": item["bm25_score"],
                "vec_score": item["vec_score"],
                "combined_score": item["rrf_score"],
            }
            for item in sorted_results
        ]

    def _fts5_snippet(self, conn, path: str, query: str) -> str:
        """Generate a snippet with ``<mark>`` tags around query terms."""
        terms = re.findall(r"\w+", query)
        if not terms:
            return ""

        fts_query = " OR ".join(terms)

        try:
            cursor = conn.execute(
                """SELECT snippet(fts_chunks, 2, '<mark>', '</mark>', '...', 40)
                   FROM fts_chunks
                   WHERE fts_chunks MATCH ?
                     AND path = ?
                   LIMIT 1""",
                (fts_query, path),
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] else ""
        except Exception:
            return ""


# ------------------------------------------------------------------
#  CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <db_path> <vault_path> [command]")
        print("Commands: full_index (default), rebuild, search <query>, status")
        sys.exit(1)

    db_path = sys.argv[1]
    vault_path = sys.argv[2]

    def _fallback_embed(_text: str) -> list[float]:
        """In production, replace with sentence-transformers all-MiniLM-L6-v2."""
        return [0.0] * 384

    indexer = Indexer(db_path, vault_path, _fallback_embed)

    if len(sys.argv) >= 4:
        cmd = sys.argv[3]
        if cmd == "rebuild":
            result = indexer.rebuild()
        elif cmd == "search" and len(sys.argv) >= 5:
            query = sys.argv[4]
            results = indexer.search(query)
            for r in results:
                print(f'  [{r["path"]}] {r["section_title"]}')
                print(f'    Score: {r["combined_score"]:.3f}')
                if r["snippet"]:
                    print(f"    {r['snippet']}")
            sys.exit(0)
        elif cmd == "status":
            status = indexer.index_status()
            print(f"Files: {status['files']}, Chunks: {status['chunks']}")
            sys.exit(0)
        else:
            result = indexer.full_index()
    else:
        result = indexer.full_index()

    print(f"Indexed: {result}")
