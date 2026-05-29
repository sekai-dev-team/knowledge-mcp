"""Watchdog-based file change watcher for the knowledge-mcp vault.

Monitors the vault directory for ``.md`` file changes and triggers
incremental reindexing via :class:`indexer.Indexer`.

Ignores ``.obsidian/``, ``.git/``, and non-``.md`` files.  Debounces
rapid successive changes to the same path within a 2-second window.
"""

import argparse
import os
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from indexer import Indexer

# ---------------------------------------------------------------------------
#  Globals set by the CLI
# ---------------------------------------------------------------------------

_vault_path: str = ""
_db_path: str = ""
_embed_fn = None


# ---------------------------------------------------------------------------
#  Debounce helper
# ---------------------------------------------------------------------------

class Debounce:
    """Tracks the last seen time for each path and answers whether enough
    time has elapsed since the last event for that path."""

    def __init__(self, cooldown: float = 2.0):
        self._cooldown = cooldown
        self._timestamps: dict[str, float] = {}

    def should_process(self, path: str) -> bool:
        now = time.monotonic()
        last = self._timestamps.get(path, 0.0)
        if now - last >= self._cooldown:
            self._timestamps[path] = now
            return True
        self._timestamps[path] = now
        return False


# ---------------------------------------------------------------------------
#  Event handler
# ---------------------------------------------------------------------------

class VaultEventHandler(FileSystemEventHandler):
    """Handles filesystem events by triggering incremental reindexing."""

    def __init__(self, vault_path: str, db_path: str, embed_fn):
        super().__init__()
        self._vault_path = vault_path
        self._db_path = db_path
        self._embed_fn = embed_fn
        self._debounce = Debounce(cooldown=2.0)

    def _reindex(self, path: str):
        """Run incremental indexing on *path* if it is a ``.md`` file
        inside the watched vault."""
        if not path.endswith(".md"):
            return
        if not self._debounce.should_process(path):
            return

        try:
            indexer = Indexer(self._db_path, self._vault_path, self._embed_fn)
            result = indexer.incremental_index(path)
            print(f"[change_watcher] Reindexed: {result}")
        except Exception as exc:
            print(f"[change_watcher] Error reindexing {path}: {exc}")

    def on_created(self, event):
        if not event.is_directory:
            self._reindex(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._reindex(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._reindex(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            # Reindex the destination (new path); the old path will be
            # cleaned up because incremental_index removes stale data.
            self._reindex(event.dest_path)


# ---------------------------------------------------------------------------
#  Ignored directory filter
# ---------------------------------------------------------------------------

#: Directories whose events are ignored by the observer.
IGNORED_DIRS = frozenset({".obsidian", ".git"})


def _should_ignore(path: str) -> bool:
    """Return True when *path* is inside an ignored directory."""
    for part in path.split(os.sep):
        if part in IGNORED_DIRS:
            return True
    return False


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    global _vault_path, _db_path, _embed_fn

    parser = argparse.ArgumentParser(
        description="Watch a vault directory and reindex changed files"
    )
    parser.add_argument(
        "--vault", required=True, help="Path to the markdown vault directory"
    )
    parser.add_argument(
        "--db", required=True, help="Path to the SQLite index database"
    )
    args = parser.parse_args()

    _vault_path = args.vault
    _db_path = args.db

    from embed import embed
    _embed_fn = embed

    event_handler = VaultEventHandler(_vault_path, _db_path, _embed_fn)
    observer = Observer()
    observer.schedule(
        event_handler,
        _vault_path,
        recursive=True,
    )

    print(f"[change_watcher] Watching {_vault_path} for changes ...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
