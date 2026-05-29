"""FastAPI + MCP Streamable HTTP server wrapping Indexer as MCP tools.

Exposes seven MCP tools (search, get_note, reindex, index_status,
write_note, update_note, list_notes) over the Model Context Protocol via
Streamable HTTP transport, alongside a FastAPI /health endpoint.
"""

import argparse
import asyncio
import datetime
import json
import os
import logging
import re
import time
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.types import ServerCapabilities, Tool, TextContent, ToolsCapability
from starlette.types import ASGIApp, Scope, Receive, Send
import uvicorn

from indexer import Indexer

# ---------------------------------------------------------------------------
#  Module-level MCP server and Streamable HTTP transport
# ---------------------------------------------------------------------------

mcp_server = Server("knowledge-mcp")
http_transport = StreamableHTTPServerTransport(
    mcp_session_id=None,
    is_json_response_enabled=True,
)

# The Indexer instance is set during the FastAPI lifespan startup.
_indexer: Indexer | None = None


def _get_indexer() -> Indexer:
    if _indexer is None:
        raise RuntimeError("Indexer not initialized")
    return _indexer


_logger = logging.getLogger(__name__)


def _retry_on_lock(fn, *args, **kwargs):
    """Execute *fn*, retrying up to 3 times on transient SQLite lock errors.

    Catches only ``sqlite3.OperationalError`` whose message contains
    *database is locked*.  Backs off exponentially (1 s, 2 s, 4 s) between
    attempts and logs a warning on the first failure so the frequency of
    lock contention can be monitored in production.
    """
    import sqlite3

    for attempt in range(4):  # initial + up to 3 retries
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "database is locked" not in str(e):
                raise
            if attempt == 0:
                _logger.warning(
                    "Database locked during %s, retrying…", fn.__name__
                )
            if attempt < 3:
                time.sleep(2**attempt)  # 1 s, 2 s, 4 s
            else:
                raise


# ---------------------------------------------------------------------------
#  MCP tool registration
# ---------------------------------------------------------------------------


@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="search",
            description="Hybrid search across the knowledge base",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 5)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_note",
            description="Read a full markdown note with frontmatter and backlinks",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the vault",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="reindex",
            description="Rebuild the entire index, or a single file if a path is provided",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional relative path to a single file to reindex",
                    },
                },
            },
        ),
        Tool(
            name="index_status",
            description="Return current index statistics",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="write_note",
            description="Create or overwrite a markdown note in the vault. "
                        "Automatically checks for semantically similar notes "
                        "before writing unless force=true.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the vault",
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown body content",
                    },
                    "frontmatter": {
                        "type": "object",
                        "description": "Optional YAML frontmatter key-value pairs",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Skip semantic duplicate check (default false)",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="update_note",
            description="Apply a string replacement to an existing note and re-index",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the vault",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Text to be replaced",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        ),
        Tool(
            name="list_notes",
            description="Return all .md filenames in the vault root directory",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    indexer = _get_indexer()
    try:
        if name == "search":
            query = arguments.get("query", "")
            limit = int(arguments.get("limit", 5))
            results = indexer.search(query, limit)
            return [TextContent(type="text", text=json.dumps(results))]
        elif name == "get_note":
            path = arguments["path"]
            return [TextContent(type="text", text=json.dumps(_read_note(indexer, path)))]
        elif name == "reindex":
            path = arguments.get("path")
            return [TextContent(type="text", text=json.dumps(_run_reindex(indexer, path)))]
        elif name == "index_status":
            return [TextContent(type="text", text=json.dumps(indexer.index_status()))]
        elif name == "write_note":
            path = arguments["path"]
            content = arguments["content"]
            frontmatter = arguments.get("frontmatter")
            force = bool(arguments.get("force", False))
            # Fallback: allow force via frontmatter (Hermes MCP adapter limitation)
            if frontmatter and frontmatter.get("force"):
                force = True
                # Strip internal key from written frontmatter
                fm_write = {k: v for k, v in frontmatter.items() if k != "force"}
                frontmatter = fm_write if fm_write else None
            return [TextContent(type="text", text=json.dumps(_write_note(indexer, path, content, frontmatter, force)))]
        elif name == "update_note":
            path = arguments["path"]
            old_string = arguments["old_string"]
            new_string = arguments["new_string"]
            return [TextContent(type="text", text=json.dumps(_update_note(indexer, path, old_string, new_string)))]
        elif name == "list_notes":
            return [TextContent(type="text", text=json.dumps(_list_notes(indexer)))]
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


# ---------------------------------------------------------------------------
#  Helper functions backing each tool
# ---------------------------------------------------------------------------


def _serialize_value(value):
    """Recursively convert ``datetime.date`` / ``datetime.datetime`` objects
    in a mixed structure (dict / list / scalar) to ISO-format strings so the
    result is JSON-safe.

    ``yaml.safe_load`` parses ``date: 2026-05-29`` as a ``datetime.date``
    object, which ``json.dumps`` cannot handle natively.
    """
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    return value


def _read_note(indexer: Indexer, path: str) -> dict:
    """Read a markdown file from the vault, parse frontmatter and wiki links."""
    full_path = os.path.join(indexer.vault_path, path)
    if not os.path.isfile(full_path):
        return {"error": f"File not found: {path}"}

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    frontmatter: dict = {}
    body = content
    fm_match = re.match(r"^---\s*\n(.*?)\n---\n*", content, re.DOTALL)
    if fm_match:
        try:
            frontmatter = yaml.safe_load(fm_match.group(1)) or {}
            frontmatter = _serialize_value(frontmatter)  # JSON-safe
        except Exception:
            frontmatter = {"_parse_error": True}
        body = content[fm_match.end() :]

    backlinks = re.findall(r"\[\[([^\]]+)\]\]", body)

    return {
        "path": path,
        "content": body.strip(),
        "frontmatter": frontmatter,
        "backlinks": backlinks,
    }


def _run_reindex(indexer: Indexer, path: str | None = None) -> dict:
    """Reindex the whole vault (path is None) or a single file."""
    start = time.time()
    if path:
        full_path = os.path.join(indexer.vault_path, path)
        if not os.path.isfile(full_path):
            return {"error": f"File not found in vault: {path}"}
        indexer.incremental_index(path)
        return {
            "status": "ok",
            "files_processed": 1,
            "elapsed": round(time.time() - start, 3),
        }
    else:
        result = indexer.rebuild()
        return {
            "status": "ok",
            "files_processed": result["total"],
            "elapsed": round(time.time() - start, 3),
        }


def _list_notes(indexer: Indexer) -> list[str]:
    """Return all .md filenames in the vault root directory."""
    files = []
    for f in sorted(os.listdir(indexer.vault_path)):
        if f.endswith(".md"):
            files.append(f)
    return files


def _write_note(
    indexer: Indexer,
    path: str,
    content: str,
    frontmatter: dict | None = None,
    force: bool = False,
) -> dict:
    """Create or overwrite a markdown note, then re-index it.

    If ``force`` is False (the default) the method first runs a semantic
    search against *content*.  When the top result's ``vec_score`` exceeds
    0.85 the note is **not** written — a *duplicate_found* warning with
    action suggestions is returned instead.  Pass ``force=True`` to skip
    this check.
    """
    full_path = os.path.join(indexer.vault_path, path)

    # Reject if file already exists to prevent accidental overwrites
    if os.path.exists(full_path):
        return {"error": f"File already exists: {path}. Use update_note to modify."}

    # Semantic duplicate check (skip when force=True)
    if not force:
        query = content[:500]
        results = indexer.search(query, limit=3)
        if results and results[0].get("vec_score", 0) > 0.85:
            top = results[0]
            return {
                "status": "duplicate_found",
                "query": query[:200],
                "similar_note": {
                    "path": top["path"],
                    "section_title": top["section_title"],
                    "snippet": top["snippet"],
                    "vec_score": top["vec_score"],
                    "bm25_score": top["bm25_score"],
                },
                "suggestion": (
                    f"A semantically similar note already exists "
                    f"(vec_score: {top['vec_score']:.3f}).\n"
                    "Choose one:\n"
                    "1. MERGE — use update_note() to add the new content "
                    "to the existing note\n"
                    "2. SPLIT — if the combined content would be too large "
                    "(>3000 words), split both into smaller atomic notes "
                    "and write separately\n"
                    "3. CREATE — if this represents a genuinely different "
                    "concept, call write_note() again with force=true to "
                    "bypass this check"
                ),
            }

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if frontmatter:
        fm_lines = ["---"]
        for k, v in frontmatter.items():
            fm_lines.append(f"{k}: {v}")
        fm_lines.append("---")
        body = "\n".join(fm_lines) + "\n\n" + content
    else:
        body = content

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(body)

    _retry_on_lock(indexer.incremental_index, path)

    return {"path": path, "updated": True}


def _update_note(indexer: Indexer, path: str, old_string: str, new_string: str) -> dict:
    """Apply a string replacement to an existing note and re-index."""
    full_path = os.path.join(indexer.vault_path, path)

    if not os.path.isfile(full_path):
        return {"error": f"File not found: {path}"}

    with open(full_path, "r", encoding="utf-8") as f:
        original = f.read()

    updated = original.replace(old_string, new_string)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(updated)

    _retry_on_lock(indexer.incremental_index, path)

    return {"path": path, "updated": True}


# ---------------------------------------------------------------------------
#  ASGI middleware to intercept /mcp/ requests for Streamable HTTP
# ---------------------------------------------------------------------------


class MCPHTTPMiddleware:
    """Intercepts requests at ``/mcp/`` and delegates to the StreamableHTTP transport.

    The transport handles all HTTP methods (POST for JSON-RPC, GET for SSE,
    DELETE for session termination) via its ``handle_request`` ASGI interface.
    """

    def __init__(self, app: ASGIApp, transport: StreamableHTTPServerTransport):
        self.app = app
        self.transport = transport

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if (
            scope["type"] == "http"
            and scope.get("path", "").rstrip("/") in ("/mcp", "/mcp/")
        ):
            await self.transport.handle_request(scope, receive, send)
        else:
            await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
#  FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the Indexer on startup and run the MCP server in background."""
    global _indexer
    _indexer = Indexer(
        db_path=app.state.db_path,
        vault_path=app.state.vault_path,
        embed_fn=app.state.embed_fn,
    )
    app.state.indexer = _indexer

    # Connect the StreamableHTTP transport and run the MCP server loop
    # in a background task so the FastAPI app can serve requests.
    async with http_transport.connect() as (read_stream, write_stream):
        mcp_task = asyncio.create_task(
            mcp_server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="knowledge-mcp",
                    server_version="0.1.0",
                    capabilities=ServerCapabilities(
                        tools=ToolsCapability(listChanged=False),
                    ),
                ),
            )
        )
        try:
            yield
        finally:
            mcp_task.cancel()
            try:
                await mcp_task
            except asyncio.CancelledError:
                pass


app = FastAPI(lifespan=lifespan, title="Knowledge MCP Server", version="0.1.0")
app.add_middleware(MCPHTTPMiddleware, transport=http_transport)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Knowledge MCP Server")
    parser.add_argument("--vault", required=True, help="Path to the markdown vault")
    parser.add_argument(
        "--db", required=True, help="Path to the SQLite index database"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind port"
    )
    args = parser.parse_args()

    app.state.vault_path = args.vault
    app.state.db_path = args.db
    from embed import embed as _real_embed

    app.state.embed_fn = _real_embed

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
