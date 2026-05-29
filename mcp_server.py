"""FastAPI + MCP SSE server wrapping Indexer as MCP tools.

Exposes six MCP tools (search, get_note, reindex, index_status,
write_note, update_note) over the Model Context Protocol via SSE
transport, alongside a FastAPI /health endpoint.
"""

import argparse
import json
import os
import re
import time
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, ServerCapabilities
from starlette.requests import Request
from starlette.types import ASGIApp, Scope, Receive, Send
import uvicorn

from indexer import Indexer

# ---------------------------------------------------------------------------
#  Module-level MCP server and SSE transport
# ---------------------------------------------------------------------------

mcp_server = Server("knowledge-mcp")
sse_transport = SseServerTransport("/messages/")

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


def _write_note(indexer: Indexer, path: str, content: str, frontmatter: dict | None = None) -> dict:
    """Create or overwrite a markdown note, then re-index it."""
    full_path = os.path.join(indexer.vault_path, path)

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
#  ASGI middleware to intercept POST /messages/ for MCP SSE
# ---------------------------------------------------------------------------


class SSEPostMiddleware:
    """Intercepts ``POST /messages/`` and delegates to the SSE transport.

    This is necessary because ``SseServerTransport.handle_post_message``
    sends the HTTP response itself via the raw ASGI *send* channel, which
    conflicts with FastAPI's normal response pipeline.
    """

    def __init__(self, app: ASGIApp, transport: SseServerTransport):
        self.app = app
        self.transport = transport

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope.get("path", "").rstrip("/") == "/messages"
        ):
            await self.transport.handle_post_message(scope, receive, send)
        else:
            await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
#  FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the Indexer on startup and store it globally."""
    global _indexer
    _indexer = Indexer(
        db_path=app.state.db_path,
        vault_path=app.state.vault_path,
        embed_fn=app.state.embed_fn,
    )
    app.state.indexer = _indexer
    yield


app = FastAPI(lifespan=lifespan, title="Knowledge MCP Server", version="0.1.0")
app.add_middleware(SSEPostMiddleware, transport=sse_transport)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/sse")
async def handle_sse(request: Request):
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp_server.run(
            streams[0],
            streams[1],
            InitializationOptions(
                server_name="knowledge-mcp",
                server_version="0.1.0",
                capabilities=ServerCapabilities(),
            ),
        )


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
