---
title: "Python Async Programming"
tags: [python, async, programming]
date: 2025-02-20
---

## Async/Await Basics

Python's `asyncio` library provides a framework for writing concurrent code using the async/await syntax. It's particularly useful for I/O-bound operations.

```python
import asyncio

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

## Event Loop

The event loop is the core of asyncio. It manages and distributes the execution of tasks. You rarely interact with it directly, but understanding it helps:

- Runs in a single thread
- Switches between coroutines at `await` points
- Handles I/O readiness notifications

## Common Patterns

### Gathering multiple tasks
```python
results = await asyncio.gather(
    fetch_data("/api/1"),
    fetch_data("/api/2"),
    fetch_data("/api/3"),
)
```

### Timeouts
```python
try:
    result = await asyncio.wait_for(
        fetch_data("/api/slow"), timeout=5.0
    )
except asyncio.TimeoutError:
    print("Request timed out")
```

See also: [[Concurrency Patterns]] for more advanced usage.
