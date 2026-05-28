---
name: obsidian-fts
description: |
  Query KONA's Obsidian knowledge base via the knowledge-mcp MCP server.
  Provides hybrid search (BM25 + semantic) + note retrieval + index management.
triggers:
  - User asks "what do I know about..." / "search my notes for..."
  - User references a concept that might be in the knowledge base
  - Before answering domain questions, check the knowledge base first
  - User asks to save/update notes
platforms: all
---

# Obsidian FTS — Knowledge Base Access

## Setup

Uses the `knowledge-mcp` MCP server. If MCP tools are not available:
1. Run `hermes mcp list` to check connection
2. If missing: `hermes mcp add knowledge --url http://knowledge-mcp:8000/sse`
3. `/reload-mcp` to pick up new tools

## Query Protocol

### Searching (查)

```text
1. Call search(query, limit=5)
2. Review snippets and scores
3. If results are relevant, pick top 2-3 and call get_note(path)
4. Synthesize from full note content

Token budget:
  - search results: ~300 tokens (5 results with snippets)
  - get_note: ~1500 tokens per full note
  - Total typical query: ~300 + ~3000 = ~3300 tokens

If search returns poor results:
  - Try rephrasing the query
  - Check if reindex is needed: call index_status()
  - Fallback: use search_files on /opt/data/vault/
```

### Adding Notes (增)

```text
1. Write the note with write_file to /opt/data/vault/<path>.md
2. Must include YAML frontmatter:
   ---
   title: "Note Title"
   tags: [tag1, tag2]
   date: YYYY-MM-DD
   ---
3. Use [[wiki links]] to connect to existing notes
4. Change watcher will auto-index within 2-5 seconds
5. Verify: call search("new note title", limit=1)
```

### Modifying Notes (改)

```text
1. Use patch or read_file + write_file to modify
2. Change watcher auto-reindexes
3. If extensive restructuring, call reindex(path) to force rebuild
```

## Quality Rules

- **Always search before answering domain questions.** Knowledge base > memory > general knowledge.
- If search returns relevant notes, cite them: "According to your note [[path]]..."
- If you add a note, tell the user where you saved it.
- Never modify notes without user confirmation (unless user explicitly asked).
- If index_status shows >24h since last index, suggest reindex.
