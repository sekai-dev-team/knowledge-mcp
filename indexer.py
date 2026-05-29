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
        """Open a new SQLite connection with the sqlite-vec extension loaded.

        WAL journal mode is enabled so that concurrent readers (search /
        index_status) do not block writers (incremental_index / rebuild),
        and a 10-second busy timeout prevents immediate ``database is locked``
        errors when the change_watcher and the MCP server both need to write.
        """
        import sqlite3
        import sqlite_vec

        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
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

        Enforces minimum chunk size (500 chars) by merging adjacent small
        sections, and maximum chunk size (4000 chars) by subdividing
        oversized sections on paragraph boundaries.

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
        raw_sections = re.split(r"\n(?=## )", content)
        sections = []

        for section in raw_sections:
            section = section.strip()
            if not section:
                continue

            title = ""
            for line in section.split("\n"):
                m = re.match(r"^##\s+(.+)$", line)
                if m:
                    title = m.group(1).strip()
                    break

            sections.append(
                {
                    "path": relpath,
                    "section_title": title,
                    "content": section,
                }
            )

        # Merge adjacent chunks smaller than MIN_CHUNK_SIZE
        MIN_CHUNK_SIZE = 500
        merged = []
        for chunk in sections:
            if (
                merged
                and len(chunk["content"]) < MIN_CHUNK_SIZE
                and len(merged[-1]["content"]) < MIN_CHUNK_SIZE
            ):
                merged[-1]["content"] += "\n\n" + chunk["content"]
                if not merged[-1]["section_title"] and chunk["section_title"]:
                    merged[-1]["section_title"] = chunk["section_title"]
            else:
                merged.append(chunk)
        sections = merged

        # Subdivide chunks larger than MAX_CHUNK_SIZE on paragraph boundaries
        MAX_CHUNK_SIZE = 4000
        subdivided = []
        for chunk in sections:
            if len(chunk["content"]) <= MAX_CHUNK_SIZE:
                subdivided.append(chunk)
                continue

            paragraphs = re.split(r"\n\n+", chunk["content"])
            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) + 2 > MAX_CHUNK_SIZE and buffer:
                    subdivided.append(
                        {
                            "path": relpath,
                            "section_title": chunk["section_title"],
                            "content": buffer.strip(),
                        }
                    )
                    buffer = para
                else:
                    if buffer:
                        buffer += "\n\n"
                    buffer += para
            if buffer:
                subdivided.append(
                    {
                        "path": relpath,
                        "section_title": chunk["section_title"],
                        "content": buffer.strip(),
                    }
                )

        return subdivided

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
                    content[:500],
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
        """Return the current index status.

        Returns:
            A dict with keys ``total_files``, ``total_chunks``,
            ``db_size_mb``, and ``last_indexed``.
        """
        conn = self._get_connection()
        try:
            self._init_schema(conn)

            cursor = conn.execute("SELECT COUNT(*) FROM file_hashes")
            file_count = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM chunk_map")
            chunk_count = cursor.fetchone()[0]

            # Database file size
            page_count = conn.execute("PRAGMA page_count").fetchone()[0]
            page_size = conn.execute("PRAGMA page_size").fetchone()[0]
            db_size_mb = round(page_count * page_size / (1024.0 * 1024.0), 3)

            # Last indexed timestamp
            cursor = conn.execute(
                "SELECT MAX(indexed_at) FROM file_hashes"
            )
            row = cursor.fetchone()
            last_indexed = row[0] if row and row[0] else None

            return {
                "total_files": file_count,
                "total_chunks": chunk_count,
                "db_size_mb": db_size_mb,
                "last_indexed": last_indexed,
            }
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
            # L2 distance on unit-normalized vectors → cosine similarity in [0, 1]
            distance = float(row[3])
            cos_sim = 1.0 - (distance * distance) / 2.0
            similarity = max(0.0, cos_sim)
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
        """Generate a snippet with ``[`` ``]`` bracket markers around matches."""
        terms = re.findall(r"\w+", query)
        if not terms:
            return ""

        fts_query = " OR ".join(terms)

        try:
            cursor = conn.execute(
                """SELECT snippet(fts_chunks, 2, '[', ']', '...', 40)
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
    import argparse

    raw_args = sys.argv[1:]

    # --- Heuristic: detect backward-compatible positional form ---
    # Positional:  <db_path> <vault_path> [command [args...]]
    # Flag-based:  --vault PATH --db PATH [--full | subcommand ...]
    # If --vault or --db appears anywhere, use the argparse form.
    _is_flag_form = any(
        a.startswith("--") and a.split("=")[0] in ("--vault", "--db")
        for a in raw_args
    )

    if (
        not _is_flag_form
        and len(raw_args) >= 2
        and not raw_args[0].startswith("-")
        and not raw_args[1].startswith("-")
    ):
        # ---- backward-compatible positional form ---------------------------------
        db_path = raw_args[0]
        vault_path = raw_args[1]
        _rest = raw_args[2:]

        command = _rest[0] if _rest else None
        cmd_query = (
            _rest[1] if _rest and _rest[0] == "search" and len(_rest) >= 2 else ""
        )
        is_full = False
    else:
        # ---- standard argparse form -----------------------------------------------
        # NOTE: command and query are positional args (not subparsers) so that
        # --vault and --db can appear on either side of the subcommand name
        # without the parent-parser / required-flag conflicts that subparsers
        # introduce.
        _parser = argparse.ArgumentParser(
            description="Hybrid search indexer for Markdown knowledge bases",
        )
        _parser.add_argument("--vault", required=True, help="Path to vault directory")
        _parser.add_argument("--db", required=True, help="Path to SQLite database file")
        _parser.add_argument(
            "--full",
            action="store_true",
            help="Perform full index (default)",
        )
        _parser.add_argument(
            "command",
            nargs="?",
            choices=["full_index", "rebuild", "search", "status"],
            default=None,
            help="Subcommand (default: full_index)",
        )
        _parser.add_argument(
            "query",
            nargs="?",
            default="",
            help="Search query (only used with 'search' subcommand)",
        )

        _parsed = _parser.parse_args(raw_args)
        vault_path = _parsed.vault
        db_path = _parsed.db
        command = _parsed.command
        cmd_query = _parsed.query if command == "search" else ""
        is_full = _parsed.full

    # ------------------------------------------------------------------
    #  Real embedding & Indexer instance
    # ------------------------------------------------------------------

    from embed import embed as _real_embed

    indexer = Indexer(db_path, vault_path, _real_embed)

    # ------------------------------------------------------------------
    #  Dispatch
    # ------------------------------------------------------------------

    if command == "rebuild":
        result = indexer.rebuild()
    elif command == "search":
        results = indexer.search(cmd_query or "")
        for r in results:
            print(f'  [{r["path"]}] {r["section_title"]}')
            print(f'    Score: {r["combined_score"]:.3f}')
            if r["snippet"]:
                print(f"    {r['snippet']}")
        sys.exit(0)
    elif command == "status":
        status = indexer.index_status()
        print(f"Files: {status['total_files']}, Chunks: {status['total_chunks']}")
        print(f"DB size: {status['db_size_mb']} MB")
        print(f"Last indexed: {status['last_indexed']}")
        sys.exit(0)
    elif command in ("full_index", None) or is_full:
        result = indexer.full_index()
    else:
        print(f"Unknown command: {command}")
        print("Commands: full_index (default), rebuild, search <query>, status")
        sys.exit(1)

    print(f"Indexed: {result}")
