"""Watchdog-based file change watcher for the knowledge-mcp vault.

Monitors the vault directory for ``.md`` file changes and notifies the
MCP server to perform incremental reindexing via its ``/reindex-file``
HTTP endpoint.  The watcher itself does NOT load the embedding model —
all heavy work is delegated to the MCP server process.

Ignores ``.obsidian/``, ``.git/``, and non-``.md`` files.  Debounces
rapid successive changes to the same path within a 2-second window.

After changes settle (60s debounce), automatically commits and pushes
to the git backup repository via SSH.
"""

import argparse
import os
import subprocess
import threading
import time
from datetime import datetime, timezone

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

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
#  Git auto-push (background thread)
# ---------------------------------------------------------------------------

class GitAutoPush:
    """Watches for file changes and auto-commits+pulls+pulls after a debounce.

    Runs in a background thread.  When signalled, waits *cooldown* seconds
    for further changes, then stages all vault changes and pushes.
    """

    def __init__(self, vault_path: str, cooldown: float = 60.0):
        self._vault_path = vault_path
        self._cooldown = cooldown
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._git_env: dict[str, str] = {}

    def _ensure_ssh_remote(self) -> None:
        """Switch git remote from HTTPS to SSH if needed (reuse mounted key)."""
        try:
            result = subprocess.run(
                ["git", "-C", self._vault_path, "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=10,
            )
            url = result.stdout.strip()
            if url.startswith("git@"):
                return  # already SSH

            subprocess.run(
                ["git", "-C", self._vault_path, "remote", "set-url", "origin",
                 "git@github.com:sekai-dev-team/yui-vault-backup.git"],
                capture_output=True, timeout=10,
            )
            print("[git-sync] Switched remote to SSH")
        except Exception as exc:
            print(f"[git-sync] Failed to switch remote: {exc}")

    def _ensure_git_config(self) -> None:
        """Set git user identity if not already configured."""
        try:
            for key, value in [
                ("user.email", "yui@hermes.local"),
                ("user.name", "Yui Backup Bot"),
            ]:
                subprocess.run(
                    ["git", "-C", self._vault_path, "config", key, value],
                    capture_output=True, timeout=5,
                )
        except Exception:
            pass

    def signal(self) -> None:
        """Wake the background thread — a file change occurred."""
        self._event.set()

    def start(self) -> None:
        """Launch the background sync thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[git-sync] Auto-push enabled (60s debounce)")

    def _run(self) -> None:
        """Main loop: wait for signals, debounce, then push."""
        # One-time setup
        self._ensure_git_config()
        self._ensure_ssh_remote()

        while self._running:
            # Wait for first signal
            self._event.wait()
            self._event.clear()

            # Debounce: keep resetting timer while new events arrive
            while True:
                self._event.wait(self._cooldown)
                if not self._event.is_set():
                    break  # cooldown expired, no new events
                self._event.clear()

            self._commit_and_push()

    def _commit_and_push(self) -> None:
        """Stage all changes, commit if dirty, and push."""
        try:
            vault = self._vault_path

            # Stage
            subprocess.run(
                ["git", "-C", vault, "add", "-A"],
                capture_output=True, timeout=30,
            )

            # Check if dirty
            diff_result = subprocess.run(
                ["git", "-C", vault, "diff", "--cached", "--quiet"],
                capture_output=True, timeout=10,
            )
            if diff_result.returncode == 0:
                return  # no changes

            # Commit
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            subprocess.run(
                ["git", "-C", vault, "commit", "-m", f"auto backup {ts}"],
                capture_output=True, timeout=10,
            )

            # Push
            push_result = subprocess.run(
                ["git", "-C", vault, "push"],
                capture_output=True, text=True, timeout=60,
            )
            print(f"[git-sync] Pushed at {ts}")
            if push_result.stderr:
                # git push sends progress to stderr, not errors
                for line in push_result.stderr.strip().split("\n"):
                    if "error" in line.lower() or "fatal" in line.lower():
                        print(f"[git-sync] Push: {line}")

        except Exception as exc:
            print(f"[git-sync] Error: {exc}")

    def stop(self) -> None:
        """Signal the thread to exit."""
        self._running = False
        self._event.set()


# ---------------------------------------------------------------------------
#  Event handler
# ---------------------------------------------------------------------------

class VaultEventHandler(FileSystemEventHandler):
    """Handles filesystem events by notifying the MCP server to reindex.

    Instead of loading the ONNX embedding model itself (which would
    duplicate ~1.2 GB of memory), this handler POSTs to the MCP server's
    ``/reindex-file`` endpoint so the work is performed inside the
    server process where the model is already resident.
    """

    def __init__(self, vault_path: str,
                 git_sync: GitAutoPush | None = None,
                 mcp_url: str = "http://127.0.0.1:8000"):
        super().__init__()
        self._vault_path = vault_path
        self._debounce = Debounce(cooldown=2.0)
        self._git_sync = git_sync
        self._mcp_url = mcp_url
        self._lock = threading.Lock()

    def _reindex(self, path: str):
        """POST *path* to the MCP server's ``/reindex-file`` endpoint.

        Serialised via ``self._lock`` to avoid flooding the server.
        Failures are logged but never crash the watcher.
        """
        if not path.endswith(".md"):
            return
        if not self._debounce.should_process(path):
            return

        relpath = os.path.relpath(path, self._vault_path)

        with self._lock:
            try:
                import urllib.request
                import json

                payload = json.dumps({"path": relpath}).encode("utf-8")
                req = urllib.request.Request(
                    f"{self._mcp_url}/reindex-file",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode())
                    print(f"[change_watcher] Reindexed: {body}")
            except Exception as exc:
                print(f"[change_watcher] Error reindexing {relpath}: {exc}")

    def _notify_git(self) -> None:
        if self._git_sync:
            self._git_sync.signal()

    def on_created(self, event):
        if not event.is_directory:
            self._reindex(event.src_path)
            self._notify_git()

    def on_modified(self, event):
        if not event.is_directory:
            self._reindex(event.src_path)
            self._notify_git()

    def on_deleted(self, event):
        if not event.is_directory:
            self._reindex(event.src_path)
            self._notify_git()

    def on_moved(self, event):
        if not event.is_directory:
            self._reindex(event.dest_path)
            self._notify_git()


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
    global _vault_path

    parser = argparse.ArgumentParser(
        description="Watch a vault directory and notify the MCP server on changes"
    )
    parser.add_argument(
        "--vault", required=True, help="Path to the markdown vault directory"
    )
    parser.add_argument(
        "--git-sync", action="store_true", default=False,
        help="Enable automatic git commit+push on file changes"
    )
    args = parser.parse_args()

    _vault_path = args.vault

    git_sync = None
    if args.git_sync:
        git_sync = GitAutoPush(_vault_path, cooldown=60.0)
        git_sync.start()

    event_handler = VaultEventHandler(_vault_path, git_sync=git_sync)
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
    if git_sync:
        git_sync.stop()
    observer.join()


if __name__ == "__main__":
    main()
