"""FastAPI + MCP Streamable HTTP server wrapping Indexer as MCP tools.

Exposes seven MCP tools (search, get_note, reindex, index_status,
write_note, update_note, list_notes) over the Model Context Protocol via
Streamable HTTP transport, alongside a FastAPI /health endpoint.
"""

import argparse
import asyncio
import json
import os
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
            description="Create or overwrite a markdown note in the vault",
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
            return [TextContent(type="text", text=json.dumps(_write_note(indexer, path, content, frontmatter)))]
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


def _write_note(indexer: Indexer, path: str, content: str, frontmatter: dict | None = None) -> dict:
    """Create or overwrite a markdown note, then re-index it."""
    full_path = os.path.join(indexer.vault_path, path)

    # Reject if file already exists to prevent accidental overwrites
    if os.path.exists(full_path):
        return {"error": f"File already exists: {path}. Use update_note to modify."}

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

    indexer.incremental_index(path)

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

    indexer.incremental_index(path)

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
